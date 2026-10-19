import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.netvlad import NetVLADLoupe
from torchvision.models.resnet import resnet18
from einops import rearrange
from einops import repeat as ei_repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # return F.avg_pool1d(x.clamp(min=eps).pow(p), x.size(-1)).pow(1./p).squeeze(-1)
        return F.avg_pool1d(x, x.size(-1)).squeeze(-1)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)), dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = ei_repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class MambaImportanceModule(nn.Module):
    def __init__(self, in_channels=1, dim=32):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        self.mamba_azimuth = MambaVisionMixer(d_model=dim)
        self.mamba_vertical = MambaVisionMixer(d_model=dim)
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        self.norm = nn.LayerNorm(dim)
        self.mask_pred = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.patch_embed(x)
        feat_flat_row = feat.flatten(2).transpose(1, 2)
        out_row = self.mamba_azimuth(feat_flat_row)
        out_row = out_row.transpose(1, 2).view(B, -1, H, W)  # (B, dim, H, W)
        feat_col_permuted = feat.permute(0, 1, 3, 2)  # (B, dim, W, H)
        feat_flat_col = feat_col_permuted.flatten(2).transpose(1, 2)  # (B, W*H, dim)
        out_col = self.mamba_vertical(feat_flat_col)  # (B, W*H, dim)
        out_col = out_col.transpose(1, 2).view(B, -1, W, H).permute(0, 1, 3, 2)
        feat_cat = torch.cat([out_row, out_col], dim=1)  # (B, 2*dim, H, W)
        feat_fused = self.fusion_proj(feat_cat)  # (B, dim, H, W)
        feat_fused = feat_fused.permute(0, 2, 3, 1)  # (B, H, W, dim)
        feat_fused = self.norm(feat_fused)
        feat_fused = feat_fused.permute(0, 3, 1, 2)  # (B, dim, H, W)
        importance_map = self.mask_pred(feat_fused)  # (B, 1, H, W)
        importance_map = importance_map.repeat(1, 3, 1, 1)
        return importance_map


class LRCross(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.importance_l = MambaImportanceModule(in_channels=1, dim=32)
        self.importance_r = MambaImportanceModule(in_channels=1, dim=32)

        self.conv1_l = pretrained.conv1
        self.bn1_l = pretrained.bn1
        self.maxpool1_l = pretrained.maxpool
        self.layer1_l = pretrained.layer1
        self.layer2_l = pretrained.layer2

        self.conv1_r = copy.deepcopy(pretrained.conv1)
        self.bn1_r = copy.deepcopy(pretrained.bn1)
        self.maxpool1_r = copy.deepcopy(pretrained.maxpool)
        self.layer1_r = copy.deepcopy(pretrained.layer1)
        self.layer2_r = copy.deepcopy(pretrained.layer2)

        self.encoder_layer_l = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0., batch_first=True)
        self.encoder_l = nn.TransformerEncoder(self.encoder_layer_l, num_layers=1)
        self.encoder_layer_r = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0., batch_first=True)
        self.encoder_r = nn.TransformerEncoder(self.encoder_layer_r, num_layers=1)

        self.pos_emb_l = LearnedPositionalEncoding(64, 7, 29)
        self.pos_emb_r = LearnedPositionalEncoding(64, 7, 29)

        self.net_vlad_l_g = NetVLADLoupe(
            feature_size=128,
            cluster_size=32,
            output_dim=256,
            gating=True,
            add_batch_norm=False
        )
        self.gem_l = GeM()
        self.linear_l = nn.Linear(203, 256)
        self.net_vlad_r_g = NetVLADLoupe(
            feature_size=128,
            cluster_size=32,
            output_dim=256,
            gating=True,
            add_batch_norm=False
        )
        self.gem_r = GeM()
        self.linear_r = nn.Linear(203, 256)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, l_bev, r_bev):
        l_bev = l_bev.view(l_bev.shape[0] * l_bev.shape[1], 3, 50, 225)
        r_bev = r_bev.view(r_bev.shape[0] * r_bev.shape[1], 3, 50, 225)

        importance_map_l = self.importance_l(l_bev[:, 0:1, :, :])
        l_bev_weighted = l_bev * importance_map_l

        feat_l_bev = self.conv1_l(l_bev_weighted)
        feat_l_bev = self.bn1_l(feat_l_bev)
        feat_l_bev = self.relu(feat_l_bev)
        feat_l_bev = self.maxpool1_l(feat_l_bev)
        feat_l_bev = self.layer1_l(feat_l_bev)
        feat_l_bev = self.layer2_l(feat_l_bev)
        avg_channel_value_l = feat_l_bev

        importance_map_r = self.importance_r(r_bev[:, 0:1, :, :])
        r_bev_weighted = r_bev * importance_map_r

        feat_r_bev = self.conv1_r(r_bev_weighted)
        feat_r_bev = self.bn1_r(feat_r_bev)
        feat_r_bev = self.relu(feat_r_bev)
        feat_r_bev = self.maxpool1_r(feat_r_bev)
        feat_r_bev = self.layer1_r(feat_r_bev)
        feat_r_bev = self.layer2_r(feat_r_bev)
        avg_channel_value_r = feat_r_bev

        # positional embedding
        feat_l_bev_shape = feat_l_bev.shape
        bev_mask_l = torch.zeros((feat_l_bev_shape[0], feat_l_bev_shape[2], feat_l_bev_shape[3]), device=feat_l_bev.device, dtype=feat_l_bev.dtype)
        pos_l = self.pos_emb_l(bev_mask_l)
        feat_l_bev = feat_l_bev + pos_l
        pos_r = self.pos_emb_r(bev_mask_l)
        feat_r_bev = feat_r_bev + pos_r

        feat_l_bev = feat_l_bev.flatten(2).permute(0, 2, 1)
        feat_r_bev = feat_r_bev.flatten(2).permute(0, 2, 1)
        feat_l_bev_g = self.encoder_l(feat_l_bev)
        feat_r_bev_g = self.encoder_r(feat_r_bev)

        # aggregation
        descriptor_l_g = self.net_vlad_l_g(feat_l_bev_g)
        descriptor_l_g = F.normalize(descriptor_l_g, dim=1)
        descriptor_l_f = self.gem_l(feat_l_bev)
        descriptor_l_f = self.linear_l(descriptor_l_f)
        descriptor_l_f = F.normalize(descriptor_l_f, dim=1)
        descriptor_l_c = torch.cat([descriptor_l_g, descriptor_l_f], dim=-1)
        descriptor_r_g = self.net_vlad_r_g(feat_r_bev_g)
        descriptor_r_g = F.normalize(descriptor_r_g, dim=1)
        descriptor_r_f = self.gem_r(feat_r_bev)
        descriptor_r_f = self.linear_r(descriptor_r_f)
        descriptor_r_f = F.normalize(descriptor_r_f, dim=1)
        descriptor_r_c = torch.cat([descriptor_r_g, descriptor_r_f], dim=-1)

        return descriptor_l_c, descriptor_r_c, avg_channel_value_l, avg_channel_value_r

    @classmethod
    def create(cls, weights=None):
        if weights is not None:
            pretrained = resnet18(weights=weights)
        else:
            pretrained = resnet18()
        model = cls(pretrained)
        return model
