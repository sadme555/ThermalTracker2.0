import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import RgbtTinyInfraredCoco
from tracking.motr_like import extract_roi_feature
from models.track_update import TrackUpdateNet


def box_xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_h: float, img_w: float) -> torch.Tensor:
    """
    boxes: [N,4] 像素坐标 [x1,y1,x2,y2]
    返回: [N,4] 归一化 (cx,cy,w,h)
    """
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return torch.stack([cx, cy, w, h], dim=1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    简单版 IoU，用于在 t+1 帧中找到与 box_t 最像的目标。
    boxes1: [N,4], boxes2: [M,4]，都是 [x1,y1,x2,y2]
    返回 [N,M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device)

    b1 = boxes1[:, None, :]  # [N,1,4]
    b2 = boxes2[None, :, :]  # [1,M,4]

    tl = torch.max(b1[..., :2], b2[..., :2])  # [N,M,2]
    br = torch.min(b1[..., 2:], b2[..., 2:])  # [N,M,2]
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    area1 = area1[:, None]
    area2 = area2[None, :]

    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


class TwoFrameDataset(Dataset):
    """
    构造 (t, t+1) 相邻帧对的数据集。

    每个样本返回：
      - img_t, target_t
      - img_tp1, target_tp1
    """

    def __init__(self, base_dataset: RgbtTinyInfraredCoco):
        super().__init__()
        self.base = base_dataset

        # 先按 seq_name 分组，把同一序列的帧按 frame_id 排一下
        seq_to_frames = defaultdict(list)  # {seq_name: [(frame_id, idx), ...]}
        for idx in range(len(self.base)):
            _, target = self.base[idx]
            seq_name = target["seq_name"]
            frame_id = int(target["frame_id"].item())
            seq_to_frames[seq_name].append((frame_id, idx))

        for k in seq_to_frames:
            seq_to_frames[k] = sorted(seq_to_frames[k], key=lambda x: x[0])

        # 构造 (t, t+1) 对的索引
        self.pairs: List[Tuple[int, int]] = []
        for seq_name, frame_list in seq_to_frames.items():
            for i in range(len(frame_list) - 1):
                _, idx_t = frame_list[i]
                _, idx_tp1 = frame_list[i + 1]
                self.pairs.append((idx_t, idx_tp1))

        print(f"[TwoFrameDataset] Total pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        idx_t, idx_tp1 = self.pairs[idx]
        img_t, tgt_t = self.base[idx_t]
        img_tp1, tgt_tp1 = self.base[idx_tp1]
        return (img_t, tgt_t, img_tp1, tgt_tp1)


def collate_twoframe(batch):
    """
    把一批 (img_t, tgt_t, img_tp1, tgt_tp1) 组合到一起。
    返回：
      images_t: [B,3,H,W]
      targets_t: List[dict]
      images_tp1: [B,3,H,W]
      targets_tp1: List[dict]
    """
    imgs_t, tgts_t, imgs_tp1, tgts_tp1 = zip(*batch)
    imgs_t = torch.stack(imgs_t, dim=0)
    imgs_tp1 = torch.stack(imgs_tp1, dim=0)
    return imgs_t, list(tgts_t), imgs_tp1, list(tgts_tp1)


def build_train_pairs(
    images_t: torch.Tensor,       # [B,3,H,W]
    targets_t: List[dict],
    images_tp1: torch.Tensor,     # [B,3,H,W]
    targets_tp1: List[dict],
    device: torch.device,
    min_iou: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从一个 batch 的 (t, t+1) 中，构造训练样本：
      - 对每个 GT box_t，在 t+1 帧中找 IoU 最高的 box_tp1（作为匹配目标）
      - 如果最大 IoU < min_iou，就丢弃这个 box_t
      - 对留下的 (box_t, box_tp1)，生成：
          feat_t: 从 img_t + box_t 提取 ROI 特征
          box_t_cxcywh: 当前帧 box_t 的归一化坐标
          box_tp1_cxcywh: 下一帧 box_tp1 的归一化坐标（监督信号）

    返回：
      feats: [N, D]
      box_t_cxcywh: [N, 4]
      box_tp1_cxcywh: [N, 4]
    """
    B, C, H, W = images_t.shape
    feats_list = []
    box_t_list = []
    box_tp1_list = []

    for b in range(B):
        img_t = images_t[b]          # [3,H,W]
        tgt_t = targets_t[b]
        img_tp1 = images_tp1[b]
        tgt_tp1 = targets_tp1[b]

        boxes_t = tgt_t["boxes"]     # [Nt,4] (xyxy, 像素)
        boxes_tp1 = tgt_tp1["boxes"] # [Ntp1,4]

        if boxes_t.numel() == 0 or boxes_tp1.numel() == 0:
            continue

        # 计算 t 和 t+1 中所有 box 两两 IoU
        ious = box_iou(boxes_t.to(device), boxes_tp1.to(device))  # [Nt,Ntp1]

        max_iou_vals, max_iou_idx = ious.max(dim=1)  # [Nt]

        # 逐个 box_t 处理
        for i in range(boxes_t.size(0)):
            iou_val = max_iou_vals[i].item()
            if iou_val < min_iou:
                continue

            box_t_xyxy = boxes_t[i].to(device)              # [4]
            box_tp1_xyxy = boxes_tp1[max_iou_idx[i]].to(device)  # [4]

            # ROI 特征（用 tracker 里的函数）
            feat_t = extract_roi_feature(img_t.to(device), box_t_xyxy)  # [D]

            # 坐标归一化
            h, w = tgt_t["orig_size"].tolist()
            box_t_cxcywh = box_xyxy_to_cxcywh_norm(box_t_xyxy.unsqueeze(0), h, w)[0]      # [4]
            box_tp1_cxcywh = box_xyxy_to_cxcywh_norm(box_tp1_xyxy.unsqueeze(0), h, w)[0]  # [4]

            feats_list.append(feat_t)
            box_t_list.append(box_t_cxcywh)
            box_tp1_list.append(box_tp1_cxcywh)

    if len(feats_list) == 0:
        # 没有有效样本，返回空张量
        return (
            torch.zeros((0, 192), device=device),
            torch.zeros((0, 4), device=device),
            torch.zeros((0, 4), device=device),
        )

    feats = torch.stack(feats_list, dim=0)           # [N,D]
    box_t_cxcywh = torch.stack(box_t_list, dim=0)    # [N,4]
    box_tp1_cxcywh = torch.stack(box_tp1_list, dim=0)# [N,4]
    return feats, box_t_cxcywh, box_tp1_cxcywh


def parse_args():
    parser = argparse.ArgumentParser("Train TrackUpdateNet on RGBT-Tiny")
    parser.add_argument("--data-root", type=str, default="data/RGBT-Tiny")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-path", type=str, default="checkpoints/track_update.pth")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    base_dataset = RgbtTinyInfraredCoco(
        root=str(data_root),
        split="train",
        transforms=None,
    )
    twoframe_dataset = TwoFrameDataset(base_dataset)

    dataloader = DataLoader(
        twoframe_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_twoframe,
    )

    # ROI 特征维度：我们在 extract_roi_feature 里默认 out_size=8，C=3 -> 3*8*8 = 192
    feat_dim = 3 * 8 * 8
    model = TrackUpdateNet(feat_dim=feat_dim, hidden_dim=256).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"[TrainTrackUpdate] Start training for {args.epochs} epochs.")
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        for images_t, targets_t, images_tp1, targets_tp1 in dataloader:
            images_t = images_t.to(device)
            images_tp1 = images_tp1.to(device)

            feats, box_t_cxcywh, box_tp1_cxcywh = build_train_pairs(
                images_t, targets_t, images_tp1, targets_tp1, device=device, min_iou=0.1
            )

            if feats.size(0) == 0:
                # 这一 batch 没有有效样本，直接跳过
                continue

            # 预测 Δbox
            delta = model(feats, box_t_cxcywh)  # [N,4]
            box_pred = box_t_cxcywh + delta     # [N,4]

            loss = F.l1_loss(box_pred, box_tp1_cxcywh)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step {global_step} "
                    f"Loss: {loss.item():.4f} "
                    f"(pairs: {feats.size(0)})"
                )

        # 每个 epoch 结束保存一次
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
        print(f"[TrainTrackUpdate] Epoch {epoch+1} saved to {save_path}")

    print("[TrainTrackUpdate] Training finished.")


if __name__ == "__main__":
    main()
