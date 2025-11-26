import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    简单的多层感知机，用于从 query 特征回归 box。
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: [..., input_dim]
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


class SimpleQueryHead(nn.Module):
    """
    把每个 query 的特征 -> 回归 box（cx,cy,w,h）+ 分类 logit。
    """

    def __init__(self, hidden_dim: int, num_queries: int):
        super().__init__()
        self.num_queries = num_queries
        # 回归 head：输出 cx, cy, w, h（归一化到 0~1）
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        # 分类 head：输出 1 维 logit（是否有目标）
        self.class_head = nn.Linear(hidden_dim, 1)

    def forward(self, query_feats: torch.Tensor):
        """
        query_feats: [B, Q, C]
        返回:
          - boxes:  [B, Q, 4]，归一化到 0~1
          - logits: [B, Q]，未过 sigmoid
        """
        boxes = self.bbox_head(query_feats)      # [B,Q,4]
        boxes = boxes.sigmoid()                  # 保证 0~1
        logits = self.class_head(query_feats)    # [B,Q,1]
        logits = logits.squeeze(-1)              # [B,Q]
        return boxes, logits


class SimpleBackbone(nn.Module):
    """
    一个非常简单的小 CNN，用来提取全局特征。
    不追求 SOTA，只要能学到一点 image-level 语义即可。
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            # [3,H,W] -> [64,H/2,W/2]
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # [64,H/4,W/4]

            # [64,H/4,W/4] -> [128,H/8,W/8]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # [128,H/8,W/8] -> [out_dim,H/8,W/8]
            nn.Conv2d(128, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W]
        return: [B,C,H',W']
        """
        return self.conv(x)


class SimpleMOTRLikeModel(nn.Module):
    """
    升级版 SimpleMOTRLikeModel：
      - backbone 提取 [B,C,H',W'] 特征；
      - 展平为 [HW,B,C] 当作 key/value；
      - num_queries 个可学习 query embedding 当作 query；
      - 用一层 MultiHeadAttention 让 query 去“看”整张 feature map；
      - head 输出 per-query 的 box 和 objectness logit。
    """

    def __init__(self, num_queries: int = 16, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # 简单 CNN backbone
        self.backbone = SimpleBackbone(out_dim=hidden_dim)

        # 2D 位置编码，用来给特征图加上 (y,x)
        self.pos_encoding = SimplePositionalEncoding2D(hidden_dim)

        # 可学习的 query embedding: [Q, C]
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 一个 MultiHeadAttention，让 query 和特征图交互
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # 把注意力后的 query 特征 -> box + logit
        self.head = SimpleQueryHead(hidden_dim=hidden_dim, num_queries=num_queries)


    def forward(self, images: torch.Tensor):
        """
        images: [B,3,H,W]
        返回:
          - boxes:  [B, num_queries, 4] (cx,cy,w,h, 0~1)
          - logits: [B, num_queries]    (是否有目标的 logit)
        """
        # 1) 提取特征并加上 2D 位置编码
        feats = self.backbone(images)           # [B,C,H,W]
        pos = self.pos_encoding(feats)          # [B,C,H,W]
        feats = feats + pos                     # [B,C,H,W]
        B, C, H, W = feats.shape

        # 2) 展平特征图: [B,C,H,W] -> [HW,B,C]
        feat_seq = feats.flatten(2).permute(2, 0, 1)  # [HW, B, C]

        # 3) 准备 query embedding: [Q,C] -> [Q,B,C]
        query_embed = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)  # [Q,B,C]

        # 4) MultiHeadAttention: query 看到整张特征图
        attn_out, _ = self.self_attn(query_embed, feat_seq, feat_seq)  # [Q,B,C]

        # 5) 转成 [B,Q,C] 喂给 head
        query_feats = attn_out.permute(1, 0, 2)  # [B,Q,C]

        boxes, logits = self.head(query_feats)   # [B,Q,4], [B,Q]
        return boxes, logits



class SimplePositionalEncoding2D(nn.Module):
    """
    非常简易的 2D 位置编码：
    - 在特征图尺寸 HxW 上生成归一化的 (y, x) 坐标；
    - 通过 1x1 conv 投影到 hidden_dim 通道；
    - 加到 backbone feature 上。
    """

    def __init__(self, num_channels: int):
        super().__init__()
        # 从 2 通道 (y,x) -> num_channels 的 1x1 卷积
        self.proj = nn.Conv2d(2, num_channels, kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, C, H, W]
        返回: [B, C, H, W] 的位置编码张量
        """
        B, C, H, W = feats.shape
        device = feats.device

        # y: [H], x: [W]，归一化到 [0,1]
        ys = torch.linspace(0, 1, H, device=device)
        xs = torch.linspace(0, 1, W, device=device)

        # 生成网格： [H,W]
        yy, xx = torch.meshgrid(ys, xs)  # 默认 indexing='ij' 在低版本 torch 是 OK 的

        # 变成 [1,2,H,W] 然后扩展到 [B,2,H,W]
        # 通道 0: y, 通道 1: x
        coords = torch.stack([yy, xx], dim=0).unsqueeze(0)  # [1,2,H,W]
        coords = coords.expand(B, -1, -1, -1)               # [B,2,H,W]

        # 投影到 num_channels
        pos = self.proj(coords)                             # [B,C,H,W]
        return pos
