from pathlib import Path
from collections import defaultdict

import torch

from datasets import RgbtTinyInfraredCoco
from tracking.motr_like import MOTRLikeTracker


def build_dataset():
    data_root = Path("/root/autodl-tmp/ThermalTracker2.0/data/RGBT-Tiny")
    print(f"RGBT-Tiny root: {data_root}")

    dataset = RgbtTinyInfraredCoco(root=str(data_root), split="train", transforms=None)
    print(f"Total train images: {len(dataset)}")
    return dataset


def group_indices_by_sequence(dataset):
    """
    遍历一遍 dataset，把每个样本的 (seq_name, frame_id, idx) 收集起来，
    返回一个 dict: seq_name -> [(frame_id, idx), ...]
    """
    seq_to_frames = defaultdict(list)

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        seq_name = target["seq_name"]
        frame_id = int(target["frame_id"].item())
        seq_to_frames[seq_name].append((frame_id, idx))

    # 每个序列内部按 frame_id 排序
    for seq_name in seq_to_frames:
        seq_to_frames[seq_name] = sorted(seq_to_frames[seq_name], key=lambda x: x[0])

    print(f"Total sequences: {len(seq_to_frames)}")
    return seq_to_frames


def run_tracking_on_one_sequence(dataset, seq_to_frames, max_frames: int = 50):
    """
    在一个序列上跑我们刚写的 MOTRLikeTracker：
    - 只选第一个序列（按名字排序）
    - 只跑前 max_frames 帧，防止一次性打印太多
    """
    # 选一个序列（第一个）
    all_seq_names = sorted(seq_to_frames.keys())
    seq_name = all_seq_names[0]
    frames = seq_to_frames[seq_name]

    print(f"Run tracking on sequence: {seq_name}, total frames: {len(frames)}")

    tracker = MOTRLikeTracker(
        iou_thresh=0.5,
        use_track_update=True,
        track_update_ckpt="checkpoints/track_update.pth",
        use_detector=True,
        detector_ckpt="checkpoints/simple_motr.pth",
        detector_num_queries=32,
        device="cuda",
    )


    tracker.reset(seq_name)

    num_frames_to_run = min(max_frames, len(frames))

    for i in range(num_frames_to_run):
        frame_id, idx = frames[i]
        img, target = dataset[idx]

        # 这里我们用 GT boxes 当检测结果
        gt_boxes: torch.Tensor = target["boxes"]  # [N,4]

        tracks = tracker.step(
            img,
            meta={
                "frame_id": frame_id,
                "gt_boxes": gt_boxes,
            },
        )

        print(f"[Seq {seq_name}] Frame {frame_id:04d}: "
              f"{len(tracks)} tracks (first few IDs: {[t['track_id'] for t in tracks[:3]]})")


def main():
    dataset = build_dataset()
    seq_to_frames = group_indices_by_sequence(dataset)
    run_tracking_on_one_sequence(dataset, seq_to_frames, max_frames=20)


if __name__ == "__main__":
    main()
