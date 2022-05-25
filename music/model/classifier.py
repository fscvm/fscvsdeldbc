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


class Classifier(BaseModule):
    def __init__(self, n_classes, n_feats, base_dim=16, depth=3, num_res=1):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.n_feats = n_feats
        self.base_dim = base_dim
        self.depth = depth
        self.num_res = num_res

        self.pre_conv = WNConv1d(n_feats, base_dim*(2 ** depth), 
                                 kernel_size=7, padding=3)
        self.pre_drop = torch.nn.Dropout()
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

        self.post_drop = torch.nn.Dropout()
        self.post_conv = torch.nn.Sequential(torch.nn.LeakyReLU(0.2),
                         WNConv1d(base_dim, n_classes, kernel_size=7, padding=3))
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, lengths):
        mask = sequence_mask(lengths).unsqueeze(1).to(x)
        x = self.pre_drop(self.pre_conv(x) * mask)
        for i in range(self.depth):
            x = self.convs[i](x) * mask
            for j in range(self.num_res):
                x = self.res_blocks[self.num_res*i + j](x, mask)
        logits = self.post_conv(self.post_drop(x)) * mask
        return logits

    def compute_loss(self, logits, targets, lengths):
        mask = sequence_mask(lengths).to(logits)
        targets = torch.stack([targets]*logits.shape[-1], -1)
        loss = self.ce_loss(logits, targets)
        loss = torch.sum(loss*mask) / torch.sum(mask)
        acc = torch.sum((logits.argmax(1) == targets)*mask).item() / torch.sum(mask).item()
        return loss, acc
