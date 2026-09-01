import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math
import numpy as np


class LinearDCT(nn.Module):
    def __init__(self, seq_len, type='ortho'):
        super(LinearDCT, self).__init__()
        self.seq_len = seq_len
        self.register_buffer('dct_matrix', self._get_dct_matrix(seq_len))

    def _get_dct_matrix(self, N):
        dct_m = np.zeros((N, N))
        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct_m[k, n] = np.sqrt(1 / N)
                else:
                    dct_m[k, n] = np.sqrt(2 / N) * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        return torch.from_numpy(dct_m).float()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_dct = torch.matmul(x, self.dct_matrix.t())
        return x_dct.permute(0, 2, 1)


class ECALayer(nn.Module):
    def __init__(self, kernel_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.avg_pool(x)  # (B, C, 1)

        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        return x * self.sigmoid(y)


class UltraLiteBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UltraLiteBlock, self).__init__()


        inter_channels = out_channels // 2
        self.reduce = nn.Conv1d(in_channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(inter_channels)
        self.relu = nn.ReLU()


        assert inter_channels % 4 == 0
        group_width = inter_channels // 4



        self.conv_g2 = nn.Conv1d(group_width, group_width, 3, padding=1, groups=group_width, bias=False)

        self.conv_g3 = nn.Conv1d(group_width, group_width, 3, padding=2, dilation=2, groups=group_width, bias=False)

        self.conv_g4 = nn.Conv1d(group_width, group_width, 3, padding=3, dilation=3, groups=group_width, bias=False)


        self.fusion = nn.Conv1d(inter_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)


        self.eca = ECALayer(kernel_size=3)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)

        x_mid = self.relu(self.bn1(self.reduce(x)))

        chunks = torch.chunk(x_mid, 4, dim=1)

        c1 = chunks[0]  # Identity
        c2 = self.conv_g2(chunks[1])
        c3 = self.conv_g3(chunks[2])
        c4 = self.conv_g4(chunks[3])


        x_out = torch.cat([c1, c2, c3, c4], dim=1)


        x_out = self.fusion(x_out)
        x_out = self.bn2(x_out)
        x_out = self.eca(x_out)

        return self.relu(x_out + residual)


class SpectralGating(nn.Module):
    def __init__(self, seq_len):
        super(SpectralGating, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, seq_len, 1))
        self.bias = nn.Parameter(torch.zeros(1, seq_len, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.weight * x + self.bias)


class CrossDomainGating(nn.Module):
    def __init__(self, dim):
        super(CrossDomainGating, self).__init__()

        self.time_gate = nn.Sequential(nn.Linear(dim, dim // 4), nn.ReLU(), nn.Linear(dim // 4, dim), nn.Sigmoid())
        self.freq_gate = nn.Sequential(nn.Linear(dim, dim // 4), nn.ReLU(), nn.Linear(dim // 4, dim), nn.Sigmoid())

    def forward(self, x_time, x_freq):
        return x_time * self.time_gate(x_freq), x_freq * self.freq_gate(x_time)


class EDTSNet (nn.Module):
    def __init__(self, num_classes=6, num_features=9, seq_len=40,
                 num_transformer_layers=3, num_heads=4,
                 conv1_nf=64, conv2_nf=128, conv3_nf=128, fc_drop_p=0.1):  # 通道数微调：48起步
        super(EDTSNet , self).__init__()


        self.embed_dim = num_features
        if self.embed_dim % num_heads != 0:
            self.embed_dim = ((num_features + num_heads - 1) // num_heads) * num_heads


        self.dct = LinearDCT(seq_len)
        self.spec_gating = SpectralGating(seq_len)
        self.freq_proj = nn.Linear(num_features, self.embed_dim)

        encoder_layer = TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads,
                                                dim_feedforward=self.embed_dim,
                                                dropout=0.1, batch_first=True)
        self.freq_transformer = TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.time_stem = nn.Sequential(
            nn.Conv1d(num_features, conv1_nf, 1, bias=False),
            nn.BatchNorm1d(conv1_nf),
            nn.ReLU()
        )

        self.layer1 = UltraLiteBlock(conv1_nf, conv2_nf)
        self.layer2 = UltraLiteBlock(conv2_nf, conv3_nf)

        self.time_pool = nn.AdaptiveAvgPool1d(1)

        self.fusion_dim = conv3_nf
        self.freq_map = nn.Linear(self.embed_dim, self.fusion_dim)
        self.cross_gating = CrossDomainGating(self.fusion_dim)

        self.fc = nn.Sequential(
            nn.Dropout(fc_drop_p),
            nn.Linear(self.fusion_dim * 2, num_classes)
        )

    def forward(self, x):
        x_f = self.dct(x)
        x_f = self.spec_gating(x_f)
        x_f = self.freq_proj(x_f)
        x_f = self.freq_transformer(x_f)
        x_f = x_f.mean(dim=1)
        x_f = self.freq_map(x_f)

        x_t = x.transpose(1, 2)
        x_t = self.time_stem(x_t)
        x_t = self.layer1(x_t)
        x_t = self.layer2(x_t)
        x_t = self.time_pool(x_t).flatten(1)

        x_t_gated, x_f_gated = self.cross_gating(x_t, x_f)
        out = torch.cat([x_t_gated, x_f_gated], dim=1)


        return self.fc(out)

