import torch
from torch.nn.utils import weight_norm

from model.base import BaseModule
from model.utils import sequence_mask


def WNConv1d(*args, **kwargs):
    return weight_norm(torch.nn.Conv1d(*args, **kwargs))


class ResnetBlock(BaseModule):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.LeakyReLU(0.2), 
                                         WNConv1d(dim, dim, kernel_size=3, padding=1),
                                         torch.nn.LeakyReLU(0.2), 
                                         WNConv1d(dim, dim, kernel_size=1, padding=0))
        self.shortcut = WNConv1d(dim, dim, kernel_size=1, padding=0)

    def forward(self, x, mask):
        y = self.shortcut(x) + self.block(x)
        return y * mask


class PitchExtractor(BaseModule):
    def __init__(self, pitch_min, pitch_max, pitch_mel, base_dim=48, depth=3, num_res=1):
        super(PitchExtractor, self).__init__()
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.pitch_center = 0.5*(pitch_min + pitch_max)
        self.pitch_mel = pitch_mel
        self.base_dim = base_dim
        self.depth = depth
        self.num_res = num_res

        self.pre_conv = WNConv1d(self.pitch_mel, base_dim*(2 ** depth), 
                                 kernel_size=7, padding=3)
        self.convs = []
        self.res_blocks = []
        for i in range(depth):
            dim_in = base_dim * (2 ** (depth - i))
            dim_out = base_dim * (2 ** (depth - i - 1))
            self.convs.append(torch.nn.Sequential(torch.nn.LeakyReLU(0.2), 
                              WNConv1d(dim_in, dim_out, kernel_size=5, padding=2)))
            for _ in range(num_res):
                self.res_blocks.append(ResnetBlock(dim_out))
        self.convs = torch.nn.ModuleList(self.convs)
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)

        self.post_conv = torch.nn.Sequential(torch.nn.LeakyReLU(0.2),
                         WNConv1d(base_dim, 1, kernel_size=7, padding=3), 
                         torch.nn.Tanh())

    def denormalize(self, y):
        return 0.5*(self.pitch_max - self.pitch_min)*y + self.pitch_center

    def forward(self, x, lengths):
        mask = sequence_mask(lengths).unsqueeze(1).to(x)
        x = self.pre_conv(x) * mask
        for i in range(self.depth):
            x = self.convs[i](x) * mask
            for j in range(self.num_res):
                x = self.res_blocks[self.num_res*i + j](x, mask)
        y = self.post_conv(x) * mask
        return self.denormalize(y)

    def compute_loss(self, y_predicted, y_target, lengths):
        mask = sequence_mask(lengths).unsqueeze(1).to(y_target)
        error = ((y_predicted - y_target) ** 2) * mask
        loss = torch.sum(error[:, :, 3:-3])
        loss = loss / torch.sum(mask[:, :, 3:-3])
        return loss
