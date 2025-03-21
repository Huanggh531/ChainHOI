import torch
import torch.nn as nn


class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1,
                 norm='BN', dropout=0, group_size=8):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.GroupNorm(group_size, out_channels) if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):

        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        pass

class mstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1,
                 group_size=8):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.GroupNorm(group_size, branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.GroupNorm(group_size, branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.GroupNorm(group_size, tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.GroupNorm(group_size, out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape#torch.Size([32, 64, 196, 25])

        branch_outs = []
        # tempconv:Sequential(
        # (0): Conv2d(64, 14, kernel_size=(1, 1), stride=(1, 1))
        # (1): GroupNorm(1, 14, eps=1e-05, affine=True)
        # (2): ReLU()
        # (3): unit_tcn(
        #     (conv): Conv2d(14, 14, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        #     (bn): Identity()
        #     (drop): Dropout(p=0, inplace=True)
        # )
        # )
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)

    def init_weights(self):
        pass
