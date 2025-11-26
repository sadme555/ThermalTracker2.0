import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import RgbtTinyRgbtCoco
from models.simple_motr import SimpleMOTRLikeModel


def box_xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_h: float, img_w: float) -> torch.Tensor:
    """
    boxes: [N,4] 像素坐标 [x1,y1,x2,y2]
    返回: [N,4] 归一化 (cx,cy,w,h)
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return torch.stack([cx, cy, w, h], dim=1)


def greedy_match(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
    """
    用简单的 L1 距离做贪心匹配：
      pred_boxes: [Q,4] (cx,cy,w,h, 0~1)
      gt_boxes:   [G,4] (cx,cy,w,h, 0~1)

    返回:
      - matched_pred_indices: [K]
      - matched_gt_indices:   [K]
    """
    Q = pred_boxes.size(0)
    G = gt_boxes.size(0)
    if Q == 0 or G == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
            torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
        )

    # 计算 L1 距离矩阵 [Q,G]
    diff = pred_boxes[:, None, :] - gt_boxes[None, :, :]  # [Q,G,4]
    dist = diff.abs().sum(dim=-1)  # [Q,G]

    matched_pred = []
    matched_gt = []
    used_pred = set()
    used_gt = set()

    # 贪心：每次找全局最小距离的 (q,g)，如果没被用过，就配对
    while True:
        min_val, min_idx = dist.min(dim=0)  # [G]
        # 全局最小
        g = int(min_val.argmin().item())
        q = int(min_idx[g].item())
        if q in used_pred or g in used_gt:
            # 把这一对的距离设为大数，下次跳过
            dist[q, g] = 1e9
            if (dist.min() >= 1e8):
                break
            continue

        matched_pred.append(q)
        matched_gt.append(g)
        used_pred.add(q)
        used_gt.add(g)

        # 关闭这一对
        dist[q, :] = 1e9
        dist[:, g] = 1e9

        if len(used_pred) == Q or len(used_gt) == G:
            break

        if (dist.min() >= 1e8):
            break

    if len(matched_pred) == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
            torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
        )

    return (
        torch.tensor(matched_pred, dtype=torch.long, device=pred_boxes.device),
        torch.tensor(matched_gt, dtype=torch.long, device=pred_boxes.device),
    )


def compute_detection_loss(pred_boxes, pred_logits, targets, device, cls_weight=1.0):
    """
    pred_boxes:  [B,Q,4]  (cx,cy,w,h, 0~1)
    pred_logits: [B,Q]    (未过 sigmoid)
    targets:     List[dict]，来自 RgbtTinyInfraredCoco

    返回: scalar loss
    """
    B, Q, _ = pred_boxes.shape

    total_bbox_loss = 0.0
    total_cls_loss = 0.0
    total_samples = 0

    for b in range(B):
        tgt = targets[b]
        boxes_xyxy = tgt["boxes"].to(device)  # [G,4]
        if boxes_xyxy.numel() == 0:
            # 没有 GT：所有 query 都是背景
            logits_b = pred_logits[b]  # [Q]
            labels = torch.zeros_like(logits_b, device=device)
            cls_loss = F.binary_cross_entropy_with_logits(logits_b, labels)
            total_cls_loss += cls_loss
            total_samples += 1
            continue

        h, w = tgt["orig_size"].tolist()
        gt_boxes_cxcywh = box_xyxy_to_cxcywh_norm(boxes_xyxy, h, w)  # [G,4]

        pred_boxes_b = pred_boxes[b]   # [Q,4]
        logits_b = pred_logits[b]      # [Q]

        matched_pred_idx, matched_gt_idx = greedy_match(pred_boxes_b, gt_boxes_cxcywh)
        if matched_pred_idx.numel() == 0:
            # 匹配失败，则只训分类：全部当背景
            labels = torch.zeros_like(logits_b, device=device)
            cls_loss = F.binary_cross_entropy_with_logits(logits_b, labels)
            total_cls_loss += cls_loss
            total_samples += 1
            continue

        # 回归 loss：只在匹配到的 query 上
        matched_pred = pred_boxes_b[matched_pred_idx]        # [K,4]
        matched_gt = gt_boxes_cxcywh[matched_gt_idx]         # [K,4]
        bbox_loss = F.l1_loss(matched_pred, matched_gt)

        # 分类 label：匹配到的 query 为 1，其余为 0
        labels = torch.zeros_like(logits_b, device=device)
        labels[matched_pred_idx] = 1.0
        cls_loss = F.binary_cross_entropy_with_logits(logits_b, labels)

        total_bbox_loss += bbox_loss
        total_cls_loss += cls_loss
        total_samples += 1

    if total_samples == 0:
        return torch.tensor(0.0, device=device)

    avg_bbox_loss = total_bbox_loss / total_samples
    avg_cls_loss = total_cls_loss / total_samples

    loss = avg_bbox_loss + cls_weight * avg_cls_loss
    return loss


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def parse_args():
    parser = argparse.ArgumentParser("Train SimpleMOTRLikeModel on RGBT-Tiny")
    parser.add_argument("--data-root", type=str, default="data/RGBT-Tiny")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--save-path", type=str, default="checkpoints/simple_motr.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    dataset = RgbtTinyRgbtCoco(
        root=str(data_root),
        split="train",
        transforms=None,
    )


    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = SimpleMOTRLikeModel(num_queries=32, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"[SimpleMOTR] Start training for {args.epochs} epochs.")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for images, targets in dataloader:
            images = images.to(device)

            pred_boxes, pred_logits = model(images)  # [B,Q,4], [B,Q]

            loss = compute_detection_loss(pred_boxes, pred_logits, targets, device=device, cls_weight=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step {global_step} "
                    f"Loss: {loss.item():.4f}"
                )

        # 每个 epoch 存一次
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args),
            },
            save_path,
        )
        print(f"[SimpleMOTR] Epoch {epoch+1} saved to {save_path}")

    print("[SimpleMOTR] Training finished.")


if __name__ == "__main__":
    main()
