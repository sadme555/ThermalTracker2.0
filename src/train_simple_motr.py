import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import RgbtTinyInfraredCoco
from models.simple_motr import SimpleMOTRLikeModel


def collate_fn(batch):
    """
    batch: list of (image, target)
    image: [3,H,W], target: dict
    返回:
      images: [B,3,H,W]
      targets: list[dict]
    """
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)  # 假设所有图像大小一样
    return images, list(targets)


def boxes_xyxy_to_cxcywh_norm(
    boxes: torch.Tensor, img_h: float, img_w: float
) -> torch.Tensor:
    """
    boxes: [N,4], [x1,y1,x2,y2] (像素)
    返回 [N,4], [cx,cy,w,h] 归一到 0~1
    """
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return torch.stack([cx, cy, w, h], dim=1)


def greedy_match(
    pred_boxes: torch.Tensor, gt_boxes: torch.Tensor
) -> List[Tuple[int, int]]:
    """
    非严格版的 Hungarian：简单贪心匹配。
    pred_boxes: [Q,4]  (cx,cy,w,h 归一化)
    gt_boxes: [G,4]    (cx,cy,w,h 归一化)
    返回一个匹配列表: [(pred_idx, gt_idx), ...]
    """
    Q = pred_boxes.size(0)
    G = gt_boxes.size(0)
    if Q == 0 or G == 0:
        return []

    # L1 代价矩阵 [Q,G]
    # 扩展成 [Q,G,4] 做绝对值和
    diff = pred_boxes[:, None, :] - gt_boxes[None, :, :]  # [Q,G,4]
    cost = diff.abs().sum(dim=-1)  # [Q,G]

    matched = []
    used_pred = set()
    used_gt = set()

    # 最多匹配 min(Q,G) 对
    for _ in range(min(Q, G)):
        # 在剩余的 pred/gt 中找到代价最小的一对
        best_cost = None
        best_p = None
        best_g = None
        for p in range(Q):
            if p in used_pred:
                continue
            for g in range(G):
                if g in used_gt:
                    continue
                c = cost[p, g].item()
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best_p = p
                    best_g = g
        if best_p is None or best_g is None:
            break
        used_pred.add(best_p)
        used_gt.add(best_g)
        matched.append((best_p, best_g))

    return matched


def compute_detection_loss(
    pred_boxes: torch.Tensor,  # [B,Q,4] (cx,cy,w,h 0~1)
    targets: List[dict],
) -> torch.Tensor:
    """
    对一个 batch 的预测和 GT 算一个简单的 L1 匹配损失。
    确保所有参与运算的张量都在 pred_boxes 同一个 device 上。
    """
    device = pred_boxes.device
    B, Q, _ = pred_boxes.shape
    total_loss = torch.tensor(0.0, device=device)
    total_gt = 0

    for b in range(B):
        tgt = targets[b]
        # GT boxes: [G,4] 像素坐标，搬到同一 device
        gt_xyxy = tgt["boxes"].to(device)
        if gt_xyxy.numel() == 0:
            # 没有 GT，就不计算这个样本的 loss
            continue

        # orig_size 一般是 CPU tensor，我们只取数值即可
        h, w = tgt["orig_size"].tolist()  # [H,W] -> python float

        # 转成 [cx,cy,w,h] 归一化到 0~1，保持在 device 上
        gt_cxcywh = boxes_xyxy_to_cxcywh_norm(gt_xyxy, h, w)  # [G,4]

        preds_b = pred_boxes[b]  # [Q,4]

        matches = greedy_match(preds_b, gt_cxcywh)
        if len(matches) == 0:
            continue

        pred_matched = []
        gt_matched = []
        for p_idx, g_idx in matches:
            pred_matched.append(preds_b[p_idx])
            gt_matched.append(gt_cxcywh[g_idx])

        pred_matched = torch.stack(pred_matched, dim=0)  # [M,4]
        gt_matched = torch.stack(gt_matched, dim=0)      # [M,4]

        l1 = F.l1_loss(pred_matched, gt_matched, reduction="sum")
        total_loss = total_loss + l1
        total_gt += gt_matched.size(0)

    if total_gt == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / total_gt



def parse_args():
    parser = argparse.ArgumentParser("Train SimpleMOTRLikeModel on RGBT-Tiny")
    parser.add_argument("--data-root", type=str,
                        default="data/RGBT-Tiny",
                        help="RGBT-Tiny root (包含 annotations_coco 和 images)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/simple_motr.pth")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    data_root = Path(args.data_root)
    dataset = RgbtTinyInfraredCoco(
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

    model = SimpleMOTRLikeModel(
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    print(f"Start training for {args.epochs} epochs.")
    model.train()

    global_step = 0
    for epoch in range(args.epochs):
        for images, targets in dataloader:
            images = images.to(device)

            # forward
            pred_boxes = model(images)  # [B,Q,4] cx,cy,w,h 0~1

            # loss
            loss = compute_detection_loss(pred_boxes, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step {global_step} "
                    f"Loss: {loss.item():.4f}"
                )

        # 每个 epoch 结束后简单保存一次
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
            },
            save_path,
        )
        print(f"[Epoch {epoch+1}] Saved checkpoint to {save_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
