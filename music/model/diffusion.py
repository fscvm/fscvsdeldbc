import math
import numpy as np
import torch
from torch.autograd.functional import vjp
from einops import rearrange

from model.base import BaseModule


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim, pe_scale):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.pe_scale = pe_scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = self.pe_scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ScoreMatchingNetwork(BaseModule):
    def __init__(self, base_dim, class_dim, n_classes, dim_mults=(1, 2, 4), 
                 groups=8, pe_scale=1000):
        super(ScoreMatchingNetwork, self).__init__()
        
        dims = [1 + class_dim, *map(lambda m: base_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(base_dim, pe_scale)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(base_dim, base_dim * 4), 
                               Mish(), torch.nn.Linear(base_dim * 4, base_dim))

        cond_total = class_dim + base_dim
        self.c_embedding = torch.nn.Embedding(n_classes, class_dim)
        self.class_block = torch.nn.Sequential(torch.nn.Linear(cond_total, 4 * class_dim),
                                        Mish(), torch.nn.Linear(4 * class_dim, class_dim))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=base_dim),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=base_dim),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=base_dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=base_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=base_dim),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=base_dim),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(base_dim, base_dim)
        self.final_conv = torch.nn.Conv2d(base_dim, 1, 1)

    def forward(self, x, mask, c, t):
        t_ = self.time_pos_emb(t)
        t = self.mlp(t_)

        x = x.unsqueeze(1)
        mask = mask.unsqueeze(1)
        
        condition = torch.cat([t_, self.c_embedding(c)], 1)
        condition = self.class_block(condition).unsqueeze(-1).unsqueeze(-1)
        condition = torch.cat(x.shape[2]*[condition], 2)
        condition = torch.cat(x.shape[3]*[condition], 3)
        x = torch.cat([x, condition], 1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


class Diffusion(BaseModule):
    def __init__(self, n_feats, n_classes, base_dim, class_dim, 
                 beta_min, beta_max):
        super(Diffusion, self).__init__()
        self.estimator = ScoreMatchingNetwork(base_dim, class_dim, n_classes)
        self.n_feats = n_feats
        self.n_classes = n_classes
        self.base_dim = base_dim
        self.class_dim = class_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.batch = 16
        self.tau = 0.95
        self.omega = 0.05
        self.silence_threshold = -20.0

    def get_beta(self, t):
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        return beta

    def get_gamma(self, s, t, p=1.0, use_torch=False):
        beta_integral = self.beta_min + 0.5*(self.beta_max - self.beta_min)*(t + s)
        beta_integral *= (t - s)
        if not use_torch:
            gamma = math.exp(-0.5*p*beta_integral)
        else:
            gamma = torch.exp(-0.5*p*beta_integral).unsqueeze(-1).unsqueeze(-1)
        return gamma

    def estimate_x0(self, xt, mask, c, t):
        t_torch = torch.ones(xt.shape[0], dtype=xt.dtype, device=xt.device, 
                             requires_grad=False) * t
        ex0 = self.estimator(xt, mask, c, t_torch)
        alpha_t = self.get_gamma(0, t, p=1.0)
        sigma2_t = 1.0 - self.get_gamma(0, t, p=2.0)
        ex0 = (xt + ex0 * sigma2_t) / alpha_t
        return ex0

    def diffuse(self, x0, mask, t):
        mean = x0 * self.get_gamma(0, t, p=1.0, use_torch=True)
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def forward_diffusion(self, x0, mask, c, n_timesteps=100):
        h = self.tau / n_timesteps
        xs = x0 * mask
        for i in range(n_timesteps):
            s = i*h
            t = s + h
            ex0 = self.estimate_x0(xs, mask, c, s)
            alpha_s = self.get_gamma(0, s, p=1.0)
            alpha_t = self.get_gamma(0, t, p=1.0)
            sigma_s = math.sqrt(1.0 - self.get_gamma(0, s, p=2.0))
            sigma_t = math.sqrt(1.0 - self.get_gamma(0, t, p=2.0))
            xs = 2*xs - (alpha_s * ex0 + sigma_s * (xs - alpha_t * ex0) / sigma_t)
            xs *= mask
        return xs

    def reverse_diffusion(self, z, mask, c, 
                          pitch_fn, chroma_fn, loudness_fn, clf_fn, 
                          pitch_value, chroma_value, loudness_value, 
                          pitch_weight, chroma_weight, loudness_weight, 
                          clf_weight, n_timesteps=100):
        h = self.tau / n_timesteps
        base_norm = torch.linalg.norm(z).item()
        pitch_weight *= base_norm
        chroma_weight *= base_norm
        loudness_weight *= base_norm
        clf_weight *= base_norm
        alpha = torch.from_numpy(np.linspace(0, 1, self.batch)).float().to(z)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).detach()
        xt = torch.cat([z * mask]*self.batch, 0)
        mask = torch.cat([mask]*self.batch, 0)
        c = torch.cat([c]*self.batch, 0)

        def opt_pitch_fn(x):
            error = (pitch_fn(x) - pitch_value) ** 2
            error = error * (loudness_value > self.silence_threshold)
            return 0.5 * torch.mean(error, dim=(1, 2))

        def opt_chroma_fn(x):
            error = torch.mean((chroma_fn(x) - chroma_value) ** 2, dim=1)
            error = error * (loudness_value > self.silence_threshold).squeeze(1)
            return 0.5 * torch.mean(error, dim=1)

        def opt_loudness_fn(x):
            threshold = self.silence_threshold * torch.ones_like(loudness_value)
            loudness_source = torch.maximum(loudness_value, threshold)
            loudness_converted = torch.maximum(loudness_fn(x), threshold)
            error = (loudness_source - loudness_converted) ** 2
            return 0.5 * torch.mean(error, dim=(1, 2))

        def opt_clf_fn(x):
            msk = (loudness_value > self.silence_threshold).squeeze(1)
            cls_result = clf_fn(x)[:, c[0].item(), :]
            cls_result = ~msk + cls_result * msk
            return cls_result.min(-1)[0]

        with torch.no_grad():
            pitch_error = opt_pitch_fn(xt.detach())
            chroma_error = opt_chroma_fn(xt.detach())
            loudness_error = opt_loudness_fn(xt.detach())
            clf_prob = opt_clf_fn(xt.detach())
        print('\nInitial MSE in pitch = %.4f' % pitch_error.mean().item())
        print('Initial MSE in chroma = %.4f' % chroma_error.mean().item())
        print('Initial MSE in loudness = %.4f' % loudness_error.mean().item())
        print('Initial min probability = %.2f%%' % (100.0 * clf_prob.mean().item()))
        ones = torch.ones_like(clf_prob).detach()

        with torch.no_grad():
            ex0 = self.estimate_x0(xt, mask, c, self.tau)

        chroma_list = []
        for i in range(n_timesteps):
            t = self.tau - i*h
            s = t - h
            if t < self.omega:
                chroma_list.append(chroma_error[k])

            alpha_s = self.get_gamma(0, s, p=1.0)
            alpha_t = self.get_gamma(0, t, p=1.0)
            sigma_s = math.sqrt(1.0 - self.get_gamma(0, s, p=2.0))
            sigma_t = math.sqrt(1.0 - self.get_gamma(0, t, p=2.0))

            pitch_error, v = vjp(opt_pitch_fn, ex0, ones, create_graph=False, strict=True)
            v_norm = torch.linalg.norm(v, dim=(1, 2), keepdim=True)
            v_pitch = v * (pitch_weight / v_norm)

            chroma_error, v = vjp(opt_chroma_fn, ex0, ones, create_graph=False, strict=True)
            v_norm = torch.linalg.norm(v, dim=(1, 2), keepdim=True)
            v_chroma = v * (chroma_weight / v_norm)

            loudness_error, v = vjp(opt_loudness_fn, ex0, ones, create_graph=False, strict=True)
            v_norm = torch.linalg.norm(v, dim=(1, 2), keepdim=True)
            v_loudness = v * (loudness_weight / v_norm)

            clf_prob, v = vjp(opt_clf_fn, ex0, ones, create_graph=False, strict=True)
            v_norm = torch.linalg.norm(v, dim=(1, 2), keepdim=True)
            v_clf = v * (clf_weight / v_norm)

            v = (v_clf - v_chroma - v_loudness - v_pitch) / alpha_t
            with torch.no_grad():
                ex0_mod = ex0 + alpha*v
                xt = alpha_s * ex0_mod + sigma_s * (xt - alpha_t * ex0_mod) / sigma_t
                xt *= mask
                ex0_fwd = self.estimate_x0(xt, mask, c, s)
                pitch_error = opt_pitch_fn(ex0_fwd).cpu().numpy()
                chroma_error = opt_chroma_fn(ex0_fwd).cpu().numpy()
                loudness_error = opt_loudness_fn(ex0_fwd).cpu().numpy()
                clf_prob = opt_clf_fn(ex0_fwd).cpu().numpy()
                if np.isnan(pitch_error).all() or np.isnan(chroma_error).all():
                    xt = torch.cat([ex0[0, :, :].unsqueeze(0)]*self.batch, 0)
                    print('Forced stop at timestep %.3f [score function failure]' % t)
                    break
                if i % 4 == 0 or t < self.omega:
                    k = np.nanargmin(chroma_error)
                elif i % 4 == 1:
                    k = np.nanargmin(loudness_error)
                elif i % 4 == 2:
                    k = np.nanargmin(pitch_error)
                else:
                    k = np.nanargmax(clf_prob)
                if t < self.omega and chroma_error[k] > min(chroma_list):
                    xt = torch.cat([ex0[0, :, :].unsqueeze(0)]*self.batch, 0)
                    print('Early stop!')
                    break
                print('\nt = %.3f' % t)
                print('MSE in pitch = %.4f' % pitch_error[k].item())
                print('MSE in chroma = %.4f' % chroma_error[k].item())
                print('MSE in loudness = %.4f' % loudness_error[k].item())
                print('Min probability = %.2f%%' % (100.0 * clf_prob[k].item()))
                ex0 = torch.cat([ex0_fwd[k, :, :].unsqueeze(0)]*self.batch, 0)                
                xt = torch.cat([xt[k, :, :].unsqueeze(0)]*self.batch, 0)

        with torch.no_grad():
            pitch_error = opt_pitch_fn(xt.detach()).cpu().numpy()
            chroma_error = opt_chroma_fn(xt.detach()).cpu().numpy()
            loudness_error = opt_loudness_fn(xt.detach()).cpu().numpy()
            clf_prob = opt_clf_fn(xt.detach()).cpu().numpy()
        k = 0
        print('\nFinal MSE in pitch = %.4f' % pitch_error[k])
        print('Final MSE in chroma = %.4f' % chroma_error[k])
        print('Final MSE in loudness = %.4f' % loudness_error[k])
        print('Final min probability = %.2f%%' % (100.0 * clf_prob[k]))
        return xt[k, :, :].unsqueeze(0)

    @torch.no_grad()
    def forward(self, z, mask, c, n_timesteps=100):
        h = self.tau / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = self.tau - i*h
            s = t - h
            ex0 = self.estimate_x0(xt, mask, c, t)
            alpha_s = self.get_gamma(0, s, p=1.0)
            alpha_t = self.get_gamma(0, t, p=1.0)
            sigma_s = math.sqrt(1.0 - self.get_gamma(0, s, p=2.0))
            sigma_t = math.sqrt(1.0 - self.get_gamma(0, t, p=2.0))
            xt = alpha_s * ex0 + sigma_s * (xt - alpha_t * ex0) / sigma_t
            xt *= mask
        return xt

    def loss_t(self, x0, mask, c, t):
        xt, z = self.diffuse(x0, mask, t)
        noise_estimation = self.estimator(xt, mask, c, t)
        noise_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss

    def compute_loss(self, x0, mask, c, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device, 
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, c, t)
