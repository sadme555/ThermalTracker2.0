from typing import List, Dict, Any
from pathlib import Path

import torch
import torch.nn.functional as F

from models.track_update import TrackUpdateNet
from models.simple_motr import SimpleMOTRLikeModel

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float):
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # 注意这里用 torch.max / torch.min，兼容旧版本 PyTorch
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        mask = iou <= iou_thresh
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long)



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

def boxes_xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_h: float, img_w: float) -> torch.Tensor:
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


def boxes_cxcywh_norm_to_xyxy(boxes: torch.Tensor, img_h: float, img_w: float) -> torch.Tensor:
    """
    boxes: [N,4] 归一化 (cx,cy,w,h)
    返回: [N,4] 像素坐标 [x1,y1,x2,y2]
    """
    cx, cy, w, h = boxes.unbind(dim=1)
    cx = cx * img_w
    cy = cy * img_h
    w = w * img_w
    h = h * img_h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=1)



class MOTRLikeTracker:
    """
    当前版本：
      - 可以用 GT 当检测结果（use_detector=False）
      - 也可以用 SimpleMOTRLikeModel 当检测器（use_detector=True）
      - 如果 use_track_update=True，会用 TrackUpdateNet 做时序预测
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        use_track_update: bool = True,
        track_update_ckpt: str = "checkpoints/track_update.pth",
        use_detector: bool = False,
        detector_ckpt: str = "checkpoints/simple_motr.pth",
        detector_num_queries: int = 16,
        device: str = "cuda",
    ):
        # 基本参数
        self.iou_thresh = iou_thresh
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id: int = 1
        self.current_seq_name: str = ""

        # 设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # === TrackUpdateNet ===
        self.use_track_update = use_track_update
        self.track_update = None

        if self.use_track_update:
            feat_dim = 3 * 8 * 8  # 和我们训练时的 ROI 特征维度一致
            self.track_update = TrackUpdateNet(feat_dim=feat_dim, hidden_dim=256)

            ckpt_path = Path(track_update_ckpt)
            if ckpt_path.exists():
                print(f"[Tracker] Loading TrackUpdateNet from {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu")
                if "model" in state:
                    self.track_update.load_state_dict(state["model"])
                else:
                    self.track_update.load_state_dict(state)
                self.track_update.eval()
            else:
                print(f"[Tracker] WARNING: TrackUpdateNet ckpt not found at {ckpt_path}, disable it.")
                self.track_update = None
                self.use_track_update = False

        # === SimpleMOTRLikeModel 检测器 ===
        self.use_detector = use_detector
        self.detector = None

        if self.use_detector:
            self.detector = SimpleMOTRLikeModel(num_queries=detector_num_queries).to(self.device)
            det_ckpt_path = Path(detector_ckpt)
            if det_ckpt_path.exists():
                print(f"[Tracker] Loading detector from {det_ckpt_path}")
                state = torch.load(det_ckpt_path, map_location=self.device)
                if "model" in state:
                    self.detector.load_state_dict(state["model"])
                else:
                    self.detector.load_state_dict(state)
                self.detector.eval()
            else:
                print(f"[Tracker] WARNING: detector ckpt not found at {det_ckpt_path}, disable detector.")
                self.detector = None
                self.use_detector = False



    def reset(self, seq_name: str):
        """开始一个新序列，清空已有轨迹。"""
        self.active_tracks = {}
        self.next_track_id = 1
        self.current_seq_name = seq_name

    def step(self, img: torch.Tensor, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        img: [3,H,W]，当前帧图像（现在只用红外）
        meta:
          - "frame_id": int
          - "gt_boxes": Tensor[M,4]，当前帧 GT（只用于可视化 / debug）
        """
        frame_id: int = meta["frame_id"]
        gt_boxes: torch.Tensor = meta.get("gt_boxes", None)

        C, H, W = img.shape

        if self.use_detector and self.detector is not None:
            # 用 SimpleMOTRLikeModel 做检测
            img_in = img.unsqueeze(0).to(self.device)  # [1,3,H,W]
            with torch.no_grad():
                pred_boxes_norm, pred_logits = self.detector(img_in)  # [1,Q,4], [1,Q]

            pred_boxes_norm = pred_boxes_norm[0].to("cpu")  # [Q,4]
            pred_logits = pred_logits[0].to("cpu")          # [Q]

            # 分类 score
            scores = pred_logits.sigmoid()                  # [Q]

            # ---------- 关键：先取 top-K，保证一定有框 ----------
            Q = scores.numel()
            if Q == 0:
                det_boxes = torch.zeros((0, 4))
            else:
                top_k = min(20, Q)  # 每帧最多先保留 20 个 query
                topk_scores, topk_idx = scores.topk(top_k)
                pred_boxes_norm = pred_boxes_norm[topk_idx]   # [K,4]
                scores = topk_scores                          # [K]

                # 归一化 cxcywh -> 像素 xyxy
                det_boxes_xyxy = boxes_cxcywh_norm_to_xyxy(pred_boxes_norm, H, W)  # [K,4]

                # 过滤非法框
                x1, y1, x2, y2 = det_boxes_xyxy.unbind(dim=1)
                keep = (x2 > x1) & (y2 > y1)
                det_boxes_xyxy = det_boxes_xyxy[keep]
                scores = scores[keep]

                if det_boxes_xyxy.numel() == 0:
                    # 如果全是非法框，就直接认为没有检测
                    det_boxes = torch.zeros((0, 4))
                else:
                    # NMS（但如果 NMS 把全抹掉了，就退回到全部）
                    iou_thresh = 0.5
                    keep_idx = nms(det_boxes_xyxy, scores, iou_thresh)
                    if keep_idx.numel() == 0:
                        # 防止一个都不剩，退回到原始的所有框
                        det_boxes = det_boxes_xyxy
                    else:
                        det_boxes = det_boxes_xyxy[keep_idx]

        else:
            # 不用检测器，就退回到“GT 当检测”的老逻辑
            if gt_boxes is None:
                det_boxes = torch.zeros((0, 4))
            else:
                det_boxes = gt_boxes

        # 当前没有任何检测，直接返回空
        if det_boxes.numel() == 0:
            return []


        # 当前没有任何检测，直接返回
        if det_boxes.numel() == 0:
            return []

        # 把当前 active_tracks 的 box 收集起来
        if len(self.active_tracks) > 0:
            track_ids = list(self.active_tracks.keys())
            # 上一帧 box
            track_boxes = torch.stack(
                [self.active_tracks[track_id]["box"] for track_id in track_ids],
                dim=0,
            )  # [T,4]

            if self.use_track_update and self.track_update is not None:
                track_feats = torch.stack(
                    [self.active_tracks[track_id]["feat"] for track_id in track_ids],
                    dim=0,
                )  # [T,D]

                box_t_cxcywh = boxes_xyxy_to_cxcywh_norm(track_boxes, H, W)  # [T,4]

                with torch.no_grad():
                    delta = self.track_update(track_feats, box_t_cxcywh)  # [T,4]
                    box_pred_cxcywh = box_t_cxcywh + delta                 # [T,4]

                # 当前帧的“预测框”（还没被 GT 校正）
                track_pred_boxes = boxes_cxcywh_norm_to_xyxy(box_pred_cxcywh, H, W)  # [T,4]
            else:
                # 不用 TrackUpdateNet，就用上一帧 box 当“预测框”
                track_pred_boxes = track_boxes  # [T,4]

            # 用预测框和当前帧 GT 计算 IoU
            ious = box_iou(track_pred_boxes, det_boxes)  # [T,N]
        else:
            track_ids = []
            track_pred_boxes = torch.zeros((0, 4))
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

                feat = extract_roi_feature(img, box)  # [D]
                pred_box = track_pred_boxes[t_idx]     # [4]，TrackUpdateNet 预测的当前帧位置

                self.active_tracks[track_id]["box"] = box
                self.active_tracks[track_id]["feat"] = feat
                self.active_tracks[track_id]["pred_box"] = pred_box
                self.active_tracks[track_id]["last_frame"] = frame_id
                det_assigned[det_idx] = True

                frame_tracks.append(
                    {
                        "track_id": track_id,
                        "box": box,          # 最终用 GT 更新后的 box
                        "pred_box": pred_box,
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
