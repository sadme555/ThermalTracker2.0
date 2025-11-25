import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackUpdateNet(nn.Module):
    """
    一个极简版的轨迹更新网络：

    输入向量 x 包含：
      - ROI 特征 feat_t: [D]
      - 当前帧 box 的 (cx, cy, w, h) （归一化到 0~1）

    我们把它们拼起来： [D + 4]

    输出：
      - 对 box 的偏移量 delta_box: [4]，表示 (Δcx, Δcy, Δw, Δh)
        用来预测下一帧的 box:
          box_{t+1}_pred = box_t + delta_box

    训练目标：
      - 让 box_{t+1}_pred 逼近下一帧的 GT box（同一目标）。
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = feat_dim + 4

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, feat: torch.Tensor, box_cxcywh: torch.Tensor) -> torch.Tensor:
        """
        :param feat: [N, D]
        :param box_cxcywh: [N, 4]
        :return: delta_box: [N, 4]
        """
        x = torch.cat([feat, box_cxcywh], dim=-1)  # [N, D+4]
        delta = self.mlp(x)  # [N, 4]
        return delta
