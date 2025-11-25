from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image, ImageDraw

from datasets import RgbtTinyInfraredCoco
from tracking.motr_like import MOTRLikeTracker


def group_indices_by_sequence(dataset):
    seq_to_frames = defaultdict(list)
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        seq_name = target["seq_name"]
        frame_id = int(target["frame_id"].item())
        seq_to_frames[seq_name].append((frame_id, idx))

    for k in seq_to_frames:
        seq_to_frames[k] = sorted(seq_to_frames[k], key=lambda x: x[0])
    return seq_to_frames


def tensor_to_pil(img_tensor):
    """
    img_tensor: [3,H,W], 0~1
    """
    img = (img_tensor * 255.0).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)


def draw_boxes(img: Image.Image, boxes: torch.Tensor, color="red", width=2):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


def main():
    data_root = Path("data/RGBT-Tiny")
    dataset = RgbtTinyInfraredCoco(root=str(data_root), split="train", transforms=None)

    seq_to_frames = group_indices_by_sequence(dataset)
    all_seq = sorted(seq_to_frames.keys())
    seq_name = all_seq[0]  # 先可视化第一个序列

    frames = seq_to_frames[seq_name]
    print(f"Visualize sequence: {seq_name}, total frames: {len(frames)}")

    tracker = MOTRLikeTracker(
        iou_thresh=0.5,
        use_track_update=True,
        track_update_ckpt="checkpoints/track_update.pth",
        use_detector=True,
        detector_ckpt="checkpoints/simple_motr.pth",
        detector_num_queries=16,
        device="cuda",
    )

    tracker.reset(seq_name)

    out_dir = Path("outputs") / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    max_frames = min(30, len(frames))

    for i in range(max_frames):
        frame_id, idx = frames[i]
        img_tensor, target = dataset[idx]

        gt_boxes = target["boxes"]

        tracks = tracker.step(
            img_tensor,
            meta={
                "frame_id": frame_id,
                "gt_boxes": gt_boxes,
            },
        )


        # 收集当前帧 tracker 输出的 box 和 pred_box
        if len(tracks) > 0:
            track_boxes = torch.stack([t["box"] for t in tracks], dim=0)  # [T,4]
            pred_boxes_list = []
            for t in tracks:
                if "pred_box" in t and t["pred_box"] is not None:
                    pred_boxes_list.append(t["pred_box"])
            if len(pred_boxes_list) > 0:
                pred_boxes = torch.stack(pred_boxes_list, dim=0)
            else:
                pred_boxes = torch.zeros((0, 4))
        else:
            track_boxes = torch.zeros((0, 4))
            pred_boxes = torch.zeros((0, 4))

        # 生成可视化：原图 + GT(绿) + 预测(蓝) + 最终轨迹(红)
        img = tensor_to_pil(img_tensor)
        img = draw_boxes(img, gt_boxes, color="green", width=2)       # GT
        img = draw_boxes(img, pred_boxes, color="blue", width=2)      # TrackUpdateNet 预测
        img = draw_boxes(img, track_boxes, color="red", width=2)      # tracker 最终 box（当前是 GT）


        out_path = out_dir / f"{frame_id:05d}.jpg"
        img.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
