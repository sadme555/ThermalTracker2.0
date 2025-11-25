import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBackbone(nn.Module):
    """
    非常简单的 backbone：几层卷积 + 下采样，把 [3,H,W] 映射到 [C, H/8, W/8] 特征图。
    后面你可以改成 ResNet、Swin 等。
    """
    def __init__(self, in_channels=3, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        # x: [B,3,H,W]
        x = F.relu(self.bn1(self.conv1(x)))  # [B,64,H/2,W/2]
        x = F.relu(self.bn2(self.conv2(x)))  # [B,128,H/4,W/4]
        x = F.relu(self.bn3(self.conv3(x)))  # [B,C,H/8,W/8]
        return x  # [B,C,H',W']


class SimpleQueryHead(nn.Module):
    """
    一个极简的 "DETR-like head"：
    - 有 N 个 learnable object queries；
    - 用全局平均特征 + MLP 把 global feature 映射到 N 个 query；
    - 用一个 box_head 输出每个 query 的 [cx,cy,w,h]（相对 0~1 的归一化坐标）。
    """
    def __init__(self, hidden_dim=128, num_queries=16):
        super().__init__()
        self.num_queries = num_queries

        # learnable query embedding
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 一个极简的 transformer encoder (单层 self-attention) 或者就直接做 MLP
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)


        # box 预测头：把每个 query hidden -> 4维 box (cx,cy,w,h) in [0,1]
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),  # 保证输出在 [0,1]
        )

    def forward(self, feat_map):
        """
        :param feat_map: [B,C,H',W']
        :return: boxes: [B, num_queries, 4] (cx,cy,w,h in 0~1)
        """
        B, C, H, W = feat_map.shape
        # 简单起见，先把整个特征图做 global avg pooling -> [B,C]
        global_feat = feat_map.mean(dim=[2, 3])  # [B,C]

        # 用 query embedding + global_feat 简单组合，比如 query + global_feat
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B,N,C]
        # broadcast global_feat 到每个 query
        global_feat_expanded = global_feat[:, None, :].expand(-1, self.num_queries, -1)  # [B,N,C]
        q_input = queries + global_feat_expanded  # [B,N,C]

        # 旧版 TransformerEncoder 期望 [S,B,E]，所以先转置
        q_input_t = q_input.transpose(0, 1)  # [N,B,C]
        q_encoded_t = self.encoder(q_input_t)  # [N,B,C]
        q_encoded = q_encoded_t.transpose(0, 1)  # [B,N,C]

        # 预测 boxes
        boxes = self.box_head(q_encoded)  # [B,N,4], 每个在 0~1
        return boxes



class SimpleMOTRLikeModel(nn.Module):
    """
    一个极简版本的 "MOTR-like" 检测模型：
    - backbone 提特征
    - query head 输出 num_queries 个候选框
    现在先不处理 track query / 时间维度，只做单帧检测。
    后面我们再把 track 信息加进来。
    """
    def __init__(self, num_queries=16, hidden_dim=128):
        super().__init__()
        self.backbone = SimpleBackbone(in_channels=3, hidden_dim=hidden_dim)
        self.head = SimpleQueryHead(hidden_dim=hidden_dim, num_queries=num_queries)

    def forward(self, images):
        """
        :param images: [B,3,H,W]
        :return: boxes: [B, num_queries, 4]  (cx,cy,w,h in 0~1)
        """
        feat = self.backbone(images)
        boxes = self.head(feat)
        return boxes
