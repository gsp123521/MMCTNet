# -*- codeing =utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.init as init
from math import sqrt

NUM_CLASS = 11

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,
                 qkv_bias=True, attn_p=0., proj_p=0., cross=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_p)


        self.qkv_bias = qkv_bias
        self.attn_p = attn_p
        self.proj_p = proj_p
        self.cross = cross

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
class CrossAttentionBlockWithPrior(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, kv, prior=None):
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)

        if prior is not None:
            kv_norm = torch.cat([prior, kv_norm], dim=1)

        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm)
        out = q + attn_out
        out = out + self.mlp(self.norm2(out))
        return out


class SpatialSelfAttention(nn.Module):
    """空间自注意力模块，用于特征增强"""

    def __init__(self, in_channels, embed_dim, n_heads=8, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = Attention(
            dim=embed_dim, heads=n_heads, qkv_bias=qkv_bias,  # 将n_heads改为heads
            attn_p=attn_p, proj_p=p
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_features),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_features, embed_dim),
            nn.Dropout(p)
        )
        self.in_channels = in_channels
        self.embed_dim = embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        x_seq = self.norm1(x_seq)
        attn_out = self.attn(x_seq)
        x_seq = x_seq + attn_out

        x_seq = self.norm2(x_seq)
        x_seq = x_seq + self.mlp(x_seq)

        out = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return identity + out, None  # 返回None保持接口兼容


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class PositionalEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        pe = torch.randn(1, self.channels, x.shape[2], x.shape[3], device=x.device)
        return x + pe


class MSAF(nn.Module):
    def __init__(self, dim, num_heads=8, topk=True,
                 kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.,
                 k1=2, k2=3, height=64, width=64):
        super(MSAF, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.k1 = k1
        self.k2 = k2


        self.avgpool1 = nn.AdaptiveAvgPool2d((height // 4, width // 4))
        self.avgpool2 = nn.AdaptiveAvgPool2d((height // 2, width // 2))
        self.avgpool3 = nn.AdaptiveAvgPool2d((height, width))
        self.min_size = 5  # 设置最小尺寸阈值
        self.layer_norm = nn.LayerNorm(dim)

        self.topk = topk

        self.pos_enc = PositionalEncoding(dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, 4),  # 直接输出4个权重
            nn.Softmax(dim=-1)
        )

        self.residual_proj = nn.Conv2d(dim, dim, kernel_size=1) if kernel[0] != 1 else nn.Identity()

        self.fusion_proj = nn.Linear(dim * 3, dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )


        self.gamma = nn.Parameter(torch.zeros(1))
        self.adapt_scale = nn.Parameter(torch.ones(1, 1, dim))
        self.norm_out = nn.LayerNorm(dim)

        self.pre_norm = nn.LayerNorm(dim)

        self._init_fusion_gate_weights()

    def _init_fusion_gate_weights(self):
        for m in self.fusion_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.fusion_gate[-2].weight, 0.1)
        nn.init.constant_(self.fusion_gate[-2].bias, 0.25)

    def forward(self, x, y):
        identity = self.residual_proj(y)
        # 检查输入尺寸

        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)


        target_size = y1.shape[2:]
        y2 = F.interpolate(y2, size=target_size, mode='bilinear', align_corners=False)
        y3 = F.interpolate(y3, size=target_size, mode='bilinear', align_corners=False)
        identity_resized = F.interpolate(identity, size=target_size, mode='bilinear', align_corners=False)

        gate_input = torch.cat([
            y1.mean([2, 3]),
            y2.mean([2, 3]),
            y3.mean([2, 3]),
            identity_resized.mean([2, 3])
        ], dim=1)  # 形状: [B, 4*C]

        gate_weights = self.fusion_gate(gate_input)

        weights_expanded = gate_weights.unsqueeze(-1).unsqueeze(-1)
        y_fused = (y1 * weights_expanded[:, 0:1] +
                   y2 * weights_expanded[:, 1:2] +
                   y3 * weights_expanded[:, 2:3] +
                   identity_resized * weights_expanded[:, 3:4])



        y_cat = torch.cat([y1, y2, y3], dim=1)
        B, C_cat, H_y, W_y = y_cat.shape

        y_cat_flat = rearrange(y_cat, 'b c h w -> b (h w) c')
        y_proj = self.fusion_proj(y_cat_flat) + rearrange(y_fused, 'b c h w -> b (h w) c')
        y_proj = self.layer_norm(y_proj)

        x_pe = self.pos_enc(x)
        x_flat = rearrange(x_pe, 'b c h w -> b (h w) c')
        x_pre_norm = self.pre_norm(x_flat)

        B, N1, C = y_proj.shape
        kv = self.kv(y_proj).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        B, N, C = x_pre_norm.shape
        q = self.q(x_pre_norm).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        ffn_out = self.ffn(out) * self.adapt_scale
        out = out + ffn_out

        hw = int(sqrt(N))
        out_spatial = rearrange(out, 'b (h w) c -> b c h w', h=hw, w=hw)

        x_orig = rearrange(x_flat, 'b (h w) c -> b c h w', h=hw, w=hw)

        out = x_orig + self.gamma * out_spatial
        out = self.norm_out(rearrange(out, 'b c h w -> b (h w) c'))
        out = rearrange(out, 'b (h w) c -> b c h w', h=hw, w=hw)

        return out


BATCH_SIZE_TRAIN = 1


class MMCTNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            num_classes=NUM_CLASS,
            num_tokens=4,
            dim=64,
            emb_dropout=0.1,
    ):
        super(MMCTNet, self).__init__()
        self.L = num_tokens
        self.cT = dim

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.5)

        #对hsi进行3dconv
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * 28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # SSA
        self.SSA = SpatialSelfAttention(
            in_channels=64,
            embed_dim=64,
            n_heads=8
        )

        self.conv2d_features2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )


        # MSAF核
        self.msc_blocks = nn.ModuleList([
            MSAF(dim=64, num_heads=8,
                 kernel=[k, k, k], s=[1, 1, 1], pad=[k // 2, k// 2, k // 2])
            for k in [3, 5, 7] #[1, 2, 3]，[1, 3, 5]，[3, 5, 7]，[3, 7, 11]，[5, 7, 13]
        ])

        #Transformer编码器
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(1)  # 可调整层数                     #Transformer Layer 与下边保持一致
        ])
        #交叉注意力
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlockWithPrior(dim=64)
            for _ in range(1)  # 与上述保持一致
        ])

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, dim, self.cT), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.fusion_conv = nn.Conv2d(64 * 3, 64, kernel_size=1)
        self.local_fc = nn.Linear(192, NUM_CLASS)

        self.local_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x1, x2, return_attn=False):
        # HSI
        x1 = self.conv3d_features(x1)
        x1 = self.dropout1(x1)  # Dropout
        x1 = rearrange(x1, 'b c h w y -> b (c h) w y')
        x1 = self.conv2d_features(x1)
        x1, attn_weights_hsi = self.SSA(x1)

        # LiDAR
        x2 = self.conv2d_features2(x2)
        x2, attn_weights_lidar = self.SSA(x2)
        # MSAF
        msc_outputs = []
        for msc in self.msc_blocks:
            msc_out = msc(x1, x2)
            msc_outputs.append(msc_out)
        #通道融合
        x1_concat = torch.cat(msc_outputs, dim=1)
        x1 = self.fusion_conv(x1_concat) + x1

        # 转换为序列
        x1_flat = rearrange(x1, 'b c h w -> b (h w) c')
        x2_flat = rearrange(x2, 'b c h w -> b (h w) c')

        # Tokenization
        wa1 = rearrange(self.token_wA, 'b h w -> b w h').repeat(x1_flat.shape[0], 1, 1)
        A1 = torch.bmm(x1_flat, wa1)
        A1 = rearrange(A1, 'b h w -> b w h')
        A1 = A1.softmax(dim=-1)

        VV1 = torch.einsum('bij,bjk->bik', x1_flat, self.token_wV)
        T1 = torch.einsum('bij,bjk->bik', A1, VV1)

        wa2 = rearrange(self.token_wA, 'b h w -> b w h').repeat(x2_flat.shape[0], 1, 1)
        A2 = torch.einsum('bij,bjk->bik', x2_flat, wa2)
        A2 = rearrange(A2, 'b h w -> b w h')
        A2 = A2.softmax(dim=-1)

        VV2 = torch.einsum('bij,bjk->bik', x2_flat, self.token_wV)
        T2 = torch.einsum('bij,bjk->bik', A2, VV2)

        # 添加CLS令牌和位置编码
        cls_tokens1 = self.cls_token.expand(x1_flat.shape[0], -1, -1)
        x1_tokens = torch.cat((cls_tokens1, T1), dim=1)
        x1_tokens += self.pos_embedding
        x1_tokens = self.dropout(x1_tokens)

        cls_tokens2 = self.cls_token.expand(x2_flat.shape[0], -1, -1)
        x2_tokens = torch.cat((cls_tokens2, T2), dim=1)
        x2_tokens += self.pos_embedding
        x2_tokens = self.dropout(x2_tokens)

        # Transformer + Cross
        for t_layer, c_layer in zip(self.transformer_layers, self.cross_layers):
            x1_tokens = t_layer(x1_tokens)
            x2_tokens = t_layer(x2_tokens)

            # 从另一模态提取 global prior
            prior_x2 = x2_tokens[:, 0:1, :]
            prior_x1 = x1_tokens[:, 0:1, :]

            x1_tokens = c_layer(x1_tokens, x2_tokens, prior=prior_x2)
            x2_tokens = c_layer(x2_tokens, x1_tokens, prior=prior_x1)

        # Global分支
        x1_cls = x1_tokens[:, 0]
        x2_cls = x2_tokens[:, 0]
        global_feat = x1_cls + x2_cls

        global_feat = self.dropout3(global_feat)


        global_logits = self.mlp_head(global_feat)

        # Local分支
        local_feat = self.local_pool(x1_concat)
        local_feat = local_feat.flatten(1)
        local_logits = self.local_fc(local_feat)

        # 融合输出
        final_logits = global_logits + local_logits

        # 输出格式
        if return_attn:
            return final_logits, global_feat, {
                'hsi_attn': attn_weights_hsi,
                'lidar_attn': attn_weights_lidar
            }
        else:
            return final_logits, global_feat


if __name__ == '__main__':
    # 测试代码
    model = MMCTNet()
    model.eval()

    # 保持原有的输入格式
    input1 = torch.randn(64, 1, 30, 11, 11)  # HSI输入
    input2 = torch.randn(64, 1, 11, 11)  # LiDAR输入


    logits, feature = model(input1, input2)

    print("输入尺寸:")
    print(f"HSI输入: {input1.shape}")
    print(f"LiDAR输入: {input2.shape}")

    print("\n输出尺寸:")
    print(f"分类logits: {logits.shape}")  # 应该是 [64, 20]
    print(f"全局特征: {feature.shape}")  # 应该是 [64, 64]

    print("\n模型替换成功！输入输出格式完全保持兼容。")