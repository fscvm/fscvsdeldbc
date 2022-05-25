import torch

from model.base import BaseModule
from model.diffusion import Diffusion
from model.utils import sequence_mask, fix_len_compatibility


class MusicGenerator(BaseModule):
    def __init__(self, n_feats, n_classes, base_dim, class_dim, 
                 beta_min, beta_max, feat_mean, feat_std):
        super(MusicGenerator, self).__init__()
        self.n_feats = n_feats
        self.n_classes = n_classes
        self.base_dim = base_dim
        self.class_dim = class_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.diffusion = Diffusion(n_feats, n_classes, base_dim, class_dim,
                                   beta_min, beta_max)

    def normalize(self, x):
        return (x - self.feat_mean) / self.feat_std

    def denormalize(self, x):
        return x * self.feat_std + self.feat_mean

    @torch.no_grad()
    def forward(self, z, lengths, c, n_timesteps=100):
        z, lengths, c = self.relocate_input([z, lengths, c])

        true_length = z.shape[-1]
        valid_length = fix_len_compatibility(true_length)

        z_ = torch.zeros((z.shape[0], self.n_feats, valid_length), 
                          dtype=z.dtype, device=z.device)
        z_[:, :, :true_length] = z
        mask = sequence_mask(lengths, valid_length).unsqueeze(1).to(z)

        output_feats_ = self.diffusion(z_, mask, c, n_timesteps=n_timesteps)
        output_feats = output_feats_[:, :, :true_length]
        return self.denormalize(output_feats)

    def compute_loss(self, x, lengths, c):
        x, lengths, c = self.relocate_input([x, lengths, c])

        true_length = x.shape[-1]
        valid_length = fix_len_compatibility(true_length)

        x0 = torch.zeros((x.shape[0], self.n_feats, valid_length), 
                          dtype=x.dtype, device=x.device)
        x0[:, :, :true_length] = x
        mask = sequence_mask(lengths, valid_length).unsqueeze(1).to(x0)

        diffusion_loss = self.diffusion.compute_loss(x0, mask, c)
        return diffusion_loss

    def convert(self, x_source, lengths_source, c_source, c_target, 
                pitch_fn, chroma_fn, loudness_fn, clf_fn, 
                pitch_value, chroma_value, loudness_value,
                pitch_weight, chroma_weight, loudness_weight, clf_weight,
                use_ot=True, n_timesteps=100):
        x_source, lengths_source = self.relocate_input([x_source, lengths_source])
        c_source, c_target = self.relocate_input([c_source, c_target])

        true_length = x_source.shape[-1]
        valid_length = fix_len_compatibility(true_length)

        x0_source = torch.zeros((x_source.shape[0], self.n_feats, valid_length), 
                                 dtype=x_source.dtype, device=x_source.device)
        x0_source[:, :, :true_length] = x_source
        x0_source = self.normalize(x0_source)
        mask = sequence_mask(lengths_source, valid_length).unsqueeze(1).to(x0_source)

        with torch.no_grad():
            if use_ot:
                z = self.diffusion.forward_diffusion(x0_source, mask, c_source, n_timesteps)
            else:
                z = torch.randn_like(x0_source)

        x0_converted = self.diffusion.reverse_diffusion(z, mask, c_target, 
                                                        pitch_fn, chroma_fn, loudness_fn, clf_fn, 
                                                        pitch_value, chroma_value, loudness_value, 
                                                        pitch_weight, chroma_weight, loudness_weight, 
                                                        clf_weight, n_timesteps=n_timesteps)
        output_feats = x0_converted[:, :, :true_length]
        return self.denormalize(output_feats)
