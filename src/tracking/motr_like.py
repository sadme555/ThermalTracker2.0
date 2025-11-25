from typing import List, Dict, Any

import torch
import torch.nn.functional as F

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算 [N,4] 和 [M,4] 两组 bbox 的 IoU，格式都是 [x1,y1,x2,y2]，返回 [N,M].
    兼容老版本 PyTorch（用 torch.max/torch.min，而不是 torch.maximum/torch.minimum）。
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)))

    # [N,1,4] 和 [1,M,4]，利用 broadcast
    b1 = boxes1[:, None, :]  # N,1,4
    b2 = boxes2[None, :, :]  # 1,M,4

    # 交集区域
    tl = torch.max(b1[..., :2], b2[..., :2])  # top-left
    br = torch.min(b1[..., 2:], b2[..., 2:])  # bottom-right
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    # 各自面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    area1 = area1[:, None]  # N,1
    area2 = area2[None, :]  # 1,M

    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou

def extract_roi_feature(
    img: torch.Tensor,
    box: torch.Tensor,
    out_size: int = 8,
) -> torch.Tensor:
    """
    从 img 中裁剪 box 对应的 patch，做一个简单的 ROI 特征：
      1. 裁剪出 [C,h,w]
      2. resize 到 [C,out_size,out_size]
      3. 展平为一维向量 [C*out_size*out_size]

    img: [3,H,W]，数值范围 0~1
    box: [4]，像素坐标 [x1,y1,x2,y2]
    """
    C, H, W = img.shape
    x1, y1, x2, y2 = box

    # 转成 int，并做 clamp，避免越界
    x1 = int(max(0, min(W - 1, float(x1))))
    x2 = int(max(0, min(W,     float(x2))))
    y1 = int(max(0, min(H - 1, float(y1))))
    y2 = int(max(0, min(H,     float(y2))))

    # 保证至少 1 像素
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)

    patch = img[:, y1:y2, x1:x2]  # [C,h,w]

    # 防御：如果裁剪结果尺寸很奇怪，直接返回 0 向量
    if patch.numel() == 0:
        return torch.zeros(C * out_size * out_size, dtype=img.dtype, device=img.device)

    # resize 到固定大小
    patch = patch.unsqueeze(0)  # [1,C,h,w]
    patch = F.interpolate(
        patch,
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
    )[0]  # [C,out_size,out_size]

    feat = patch.reshape(-1)  # [C*out_size*out_size]
    return feat

class MOTRLikeTracker:
    """
    目前版本：完全用 **GT boxes** 当检测结果，只做 IOU 匹配 + 轨迹管理。

    接口：
      - reset(seq_name): 开始一个新视频
      - step(img, meta): 输入一帧图像 + GT 框，输出这一帧的轨迹列表
        meta 里需要：
          - "frame_id": int
          - "gt_boxes": Tensor[N,4] （像素坐标 [x1,y1,x2,y2]）
    """

    def __init__(self, iou_thresh: float = 0.5):
        self.iou_thresh = iou_thresh
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id: int = 1
        self.current_seq_name: str = ""

    def reset(self, seq_name: str):
        """开始一个新序列，清空已有轨迹。"""
        self.active_tracks = {}
        self.next_track_id = 1
        self.current_seq_name = seq_name

    def step(self, img: torch.Tensor, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        :param img: 一帧图像 [C,H,W]（目前没用，将来可以给模型用）
        :param meta: 包含
            - "frame_id": int
            - "gt_boxes": Tensor[N,4]，当前帧的 GT 检测框（像素坐标）
        :return: 这一帧的轨迹列表，每个元素是一个 dict:
            {
              "track_id": int,
              "box": Tensor[4],   # [x1,y1,x2,y2]
              "frame_id": int,
            }
        """
        frame_id: int = meta["frame_id"]
        det_boxes: torch.Tensor = meta["gt_boxes"]  # [N,4]

        # 当前没有任何检测，直接返回
        if det_boxes.numel() == 0:
            return []

        # 把当前 active_tracks 的 box 收集起来
        if len(self.active_tracks) > 0:
            track_ids = list(self.active_tracks.keys())
            track_boxes = torch.stack(
                [self.active_tracks[tid]["box"] for tid in track_ids],
                dim=0,
            )  # [T,4]
            ious = box_iou(track_boxes, det_boxes)  # [T,N]
        else:
            track_ids = []
            ious = torch.zeros((0, det_boxes.size(0)))

        frame_tracks: List[Dict[str, Any]] = []
        num_det = det_boxes.size(0)
        det_assigned = torch.zeros(num_det, dtype=torch.bool)

        for t_idx, track_id in enumerate(track_ids):
            if num_det == 0:
                break
            track_iou = ious[t_idx]  # [N]
            max_iou, det_idx = track_iou.max(dim=0)
            if max_iou >= self.iou_thresh and not det_assigned[det_idx]:
                box = det_boxes[det_idx]

                # === 新增：根据当前帧图像 + 新 box 提取轨迹特征 ===
                feat = extract_roi_feature(img, box)  # [D]

                self.active_tracks[track_id]["box"] = box
                self.active_tracks[track_id]["feat"] = feat
                self.active_tracks[track_id]["last_frame"] = frame_id
                det_assigned[det_idx] = True

                frame_tracks.append(
                    {
                        "track_id": track_id,
                        "box": box,
                        "frame_id": frame_id,
                        "feat": feat,
                    }
                )


        for det_idx in range(num_det):
            if det_assigned[det_idx]:
                continue
            box = det_boxes[det_idx]

            # === 新增：对新轨迹也提一个初始特征 ===
            feat = extract_roi_feature(img, box)  # [D]

            track_id = self.next_track_id
            self.next_track_id += 1

            self.active_tracks[track_id] = {
                "box": box,
                "feat": feat,
                "last_frame": frame_id,
            }

            frame_tracks.append(
                {
                    "track_id": track_id,
                    "box": box,
                    "frame_id": frame_id,
                    "feat": feat,
                }
            )


        return frame_tracks
