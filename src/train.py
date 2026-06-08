import argparse
import comet_ml
import copy
import geopandas as gpd
import torch
import torch.nn.functional as F
import json
import yaml

from neuralop.layers.fno_block import FNOBlocks
from rasterio.windows import from_bounds
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchinfo import summary
from transformers import ViTConfig, ViTModel

from constants import CF_PATH, GRID_PLOTS_PATH, IMAGERY_DIR, PLOTS_DIR, PRISM_FILES, SOLUS_FILES, UNITS_DIR, UNITS_PATH, STAT_PATH
from constants import TRAIN_OUT_DIR, PT_PATH, TOP_CKPT
from constants import DIRS_YEARS, PRISM_ELEMENTS, PRISM_YEARS
from constants import CDTYPE, CHIP_SIZE, DEVICE, DTYPE, L_SIZE, NO_DATA, PATCH_SIZE, PIXEL_SIZE

from prithvi_mae import PrithviMAE
from utils import crop_and_pad, random_box_crop, read_window, cache_anc


class MSELoss(torch.nn.Module):
    def __init__(self, ignore_index=NO_DATA):
        super().__init__()

        self.ignore_index = ignore_index

    def forward(self, preds, target):
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            print(preds[valid_mask], target[valid_mask])
            print(preds[valid_mask][torch.isnan(preds[valid_mask])], target[valid_mask][torch.isnan(target[valid_mask])])
            
            diff = preds[valid_mask] - target[valid_mask]
            print(diff[torch.isnan(diff)])
            mse = torch.mean(diff ** 2)
            return mse, diff.shape[0]
        else:
            diff = preds - target
            mse = torch.mean(diff ** 2)
            return mse, torch.sum(diff.shape[-2:])


class CachedUnitDataset(Dataset):
    def __init__(self, source, dem, cache, imagery_dir):
        self.source = source
        self.dem = dem
        self._cache = bool(cache)
        
        self.imagery_dir = imagery_dir
        self.dem_file = self.imagery_dir.glob(f"{self.dem}.tif").__next__()
        self.prism_dat = cache["prism_dat"]
        self.solus_dat = cache["solus_dat"]
        self.patch_transform = cache["patch_transform"]
    
    def __get_cache__(self, chip_bounds, year):
        win = from_bounds(*chip_bounds, self.patch_transform)
        col_off, row_off = int(win.col_off), int(win.row_off)

        out = {}
        out["SOLUS100"] = crop_and_pad(self.solus_dat, col_off, row_off, L_SIZE, L_SIZE).clone()
        
        ancp = []
        for e in PRISM_ELEMENTS:
            E = e.upper()
            out[f"PRISM_{E}"] = crop_and_pad(self.prism_dat[E], col_off, row_off, L_SIZE, L_SIZE).clone()
            Y = int((year-PRISM_YEARS[0])*12)
            YM = slice(Y+6,Y+8)
            ancp.append(torch.mean(out[f"PRISM_{E}"][YM], dim=0))
        out["PRISM__"] = torch.stack(ancp)
    
        return out
    
    def __getitem__(self, idx):
        unit_path, bounds, year = self.get_path_bounds_year(idx)
        
        out = {"YEAR": year}
        out["agb"] = read_window(unit_path, bounds, 1)
        out[self.source] = read_window(self.imagery_dir / f"{self.source}{year}.tif", bounds)
        out[self.dem] = read_window(self.dem_file, bounds, 1)

        out.update(self.__get_cache__(bounds, year)) 

        return out


class LidarUnitTrainDataset(CachedUnitDataset):
    def __init__(self, source='LS', dem='NASADEM', cache={}, imagery_dir=IMAGERY_DIR,
                 units_dir=UNITS_DIR,
                 units_path=UNITS_PATH):
        super().__init__(source=source, dem=dem, cache=cache, imagery_dir=imagery_dir)
        self.years = list(DIRS_YEARS[self.source])
        self.units = gpd.read_file(units_path)
        self.units = self.units[self.units['LidarYear'].isin(self.years)]  
        self.unit_files = {u.with_suffix("").name.replace("-", "_"): u for u in units_dir.glob("*.tif") if "StdDev" not in u.name}

    def __len__(self):
        return len(self.units)

    def get_path_bounds_year(self, idx):
        unit_path = self.unit_files[self.units.iloc[idx]['LidarUnit']]
        bounds = random_box_crop(self.units.iloc[idx].geometry, CHIP_SIZE*PIXEL_SIZE)
        year = int(self.units.iloc[idx]['LidarYear'])
        
        return unit_path, bounds, year
    

class PlotValDataset(CachedUnitDataset):
    def __init__(self, source='LS', dem='NASADEM', cache={}, imagery_dir=IMAGERY_DIR,
                 grid_plots_path=GRID_PLOTS_PATH,
                 plots_dir=PLOTS_DIR):
        super().__init__(source=source, dem=dem, cache=cache, imagery_dir=imagery_dir)
        self.plots_dir = plots_dir
        self.grid_plots = gpd.read_file(grid_plots_path)

    def __len__(self):
        return len(self.grid_plots)

    def get_path_bounds_year(self, idx):
        bounds = self.grid_plots.iloc[idx].geometry.bounds
        year = int(self.grid_plots.iloc[idx]['INV_YEAR'])
        unit_path = self.plots_dir / f"{self.grid_plots.iloc[idx]['fid']:05d}_{year}.tif"
        return unit_path, bounds, year


class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => GELU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, padding=1, mid=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.GELU(),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            *[torch.nn.BatchNorm2d(out_channels), torch.nn.GELU()] if mid else [torch.nn.Identity()],
        )

    def forward(self, x):
        return self.double_conv(x)


class FFTBlock(torch.nn.Module):
    def __init__(self, n_modes=64, channels=360, bias=False):
        super().__init__()
        self.n_modes = n_modes
        self.mlp = torch.nn.Conv2d(in_channels=n_modes, out_channels=channels, kernel_size=1, bias=bias).to(DEVICE, CDTYPE)
        self.norm = torch.nn.Sequential(
            torch.nn.BatchNorm2d(channels),
            torch.nn.GELU()
        )

    def forward(self, x):  
        _, T, _, _ = x.shape
        TM = T // 2
        
        x = torch.fft.fftn(x, dim=-3)
        x = torch.fft.fftshift(x, dim=-3)

        x = x[:, TM-self.n_modes//2: TM+self.n_modes//2]
        x = self.mlp(x)
        x = torch.fft.ifftshift(x, dim=-3)
        x = torch.fft.ifftn(x, dim=-3)
        x = self.norm(x.real)

        return x

class PRISMEncoderBlock(torch.nn.Module):
    def __init__(self, num_ftts=3, n_tmodes=64, tfno_channels=360, out_channels=64):
        super().__init__()
        self.tfno = torch.nn.ModuleList([FFTBlock(n_modes=n_tmodes, channels=tfno_channels) for _ in range(num_ftts)])
        self.mlpnorm = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=tfno_channels, out_channels=out_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.GELU()
            )
        self.to(DEVICE) # necessary call due to complex floats and current version of cuda

    def forward(self, x):
        for tfno in self.tfno:
            x = x + tfno(x)
        x = self.mlpnorm(x)
        return x


class SOLUSEncoderBlock(torch.nn.Module):
    def __init__(self, channels=128, num_fnos=3, n_modes=16):
        super().__init__()
        self.sfno = FNOBlocks(n_modes=(n_modes, n_modes),
                        in_channels=channels,
                    out_channels=channels,
                    n_layers=num_fnos)

    def forward(self, x):
        return self.sfno(x)
            

class Upscaler(torch.nn.Module):
    def __init__(self, embed_dim: int, depth: int, dropout: bool = True):
        super().__init__()

        def build_block(in_ch, out_ch): return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2) if dropout else torch.nn.Identity(),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.GELU())
        self.embed_dim = embed_dim

        self.upscale_blocks = torch.nn.ModuleList()
        for i in range(depth):
            if i == 0:
                dim0 = self.embed_dim
            else:
                dim0 = self.embed_dim // (2 ** i)
            dim1 = self.embed_dim // (2 ** (i + 1))
            self.upscale_blocks.append(build_block(dim0, dim1))

    def forward(self, x):
        for blk in self.upscale_blocks:
            x = blk(x)
        return x


class CrossAttentionFusion(torch.nn.Module):
    def __init__(self, C_latent, C_anc, num_heads=8, ff_multiplier=4, dropout=0.1):
        super().__init__()

        self.q_proj = torch.nn.Linear(C_latent, C_latent)
        self.k_proj = torch.nn.Linear(C_anc, C_latent)
        self.v_proj = torch.nn.Linear(C_anc, C_latent)

        self.attn = torch.nn.MultiheadAttention(C_latent, num_heads, batch_first=True)

        self.out_proj = torch.nn.Linear(C_latent, C_latent)

        self.ln1 = torch.nn.LayerNorm(C_latent)
        self.ln2 = torch.nn.LayerNorm(C_latent)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(C_latent, C_latent * ff_multiplier),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(C_latent * ff_multiplier, C_latent),
            torch.nn.Dropout(dropout),
        )

    def forward(self, latent, anc):
        B, _, D  = latent.shape

        anc_flat = anc.flatten(2).transpose(1, 2)      # B, HW, C2

        q = self.q_proj(latent)
        k = self.k_proj(anc_flat)
        v = self.v_proj(anc_flat)

        attn_out, _ = self.attn(q, k, v)
        attn_out = self.out_proj(attn_out)

        x = self.ln1(latent + attn_out)
        x = self.ln2(x + self.ff(x))

        out = x.transpose(1, 2).reshape(B, D, L_SIZE, L_SIZE)
        return out


class AttnSETRPUP(torch.nn.Module):
    def __init__(self, img, dem, prithvi='frozen_pe', ablate=[], num_prism_fnos=3, **kwargs):
        super().__init__()
        self.img = img
        self.dem = dem
        self.ablate = ablate
        
        with open(CF_PATH, "r") as f:
            conf = json.load(f)

        if prithvi == 'conv':
            self.prithvi_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(6, 1024, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=False),
                DoubleConv(1024, out_channels=1024, mid_channels=1024, kernel_size=3, stride=1, padding=1)
            )
            self.embed_dim = 1024
        elif prithvi == 'vit':
            self.vit_config = ViTConfig(hidden_size=1024, num_channels=6, num_hidden_layers=24, num_attention_heads=16)
            self.prithvi_encoder = ViTModel(self.vit_config)
            self.embed_dim = self.vit_config.hidden_size
        else:
            prithvi_model = PrithviMAE(**conf["pretrained_cfg"])
            if prithvi != 'random':
                state_dict = torch.load(PT_PATH)
                prithvi_model.load_state_dict(state_dict)
            
            self.embed_dim = prithvi_model.encoder.embed_dim
            self.prithvi_encoder = copy.deepcopy(prithvi_model.encoder)
            
            if prithvi == 'frozen':
                for param in self.prithvi_encoder.parameters():
                    param.requires_grad = False
            elif prithvi == "frozen_pe":
                for blk in self.prithvi_encoder.blocks:
                    for param in blk.parameters():
                        param.requires_grad = False
        self.prithvi_mode = prithvi
        
        self.upscaler = Upscaler(embed_dim=self.embed_dim, depth=4)
        if self.dem in self.ablate:
            self.dem_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=False),
                torch.nn.BatchNorm2d(16),
                torch.nn.GELU(),
                DoubleConv(in_channels=16, out_channels=16, mid_channels=64)
            )
        else:
            self.dem_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=False),
                torch.nn.BatchNorm2d(16),
                torch.nn.GELU(),
                FNOBlocks(n_modes=(16, 16),
                        in_channels=16,
                        out_channels=16,
                        n_layers=3),
            )
        
        if "SOLUS100" in self.ablate:
            self.solus_encoder = DoubleConv(in_channels=128, out_channels=128, mid_channels=256)
        else:
            self.solus_encoder = SOLUSEncoderBlock()
        
        if "PRISM__" in self.ablate:
            self.prism_encoder = DoubleConv(in_channels=len(PRISM_ELEMENTS), out_channels=len(PRISM_ELEMENTS), mid_channels=len(PRISM_ELEMENTS)*2)
        else:
            self.prism_encoder = FNOBlocks(n_modes=(16, 16),
                        in_channels=len(PRISM_ELEMENTS),
                        out_channels=len(PRISM_ELEMENTS),
                        n_layers=num_prism_fnos)
        
        self.prism_encoders = {}
        for p in PRISM_ELEMENTS:
            name = f"PRISM_{p.upper()}"
            if name in self.ablate:
                self.prism_encoders[name] = DoubleConv(in_channels=360, out_channels=64, mid_channels=512)
            else:
                self.prism_encoders[name] = PRISMEncoderBlock()
        self.prism_encoders = torch.nn.ModuleDict(self.prism_encoders)
        
        C_anc = 16+128+((64+1)*len(PRISM_ELEMENTS)) 
        self.fusion = CrossAttentionFusion(C_latent=self.embed_dim, C_anc=C_anc)
        self.head = DoubleConv(self.embed_dim // 2**4, 1, mid_channels=self.embed_dim // 2**4, mid=False)
    
    def forward(self, x):
        anc = []
        anc.append(self.dem_encoder(x[self.dem]))
        anc.append(self.solus_encoder(x["SOLUS100"]))
        
        for p in PRISM_ELEMENTS:
            P = f"PRISM_{p.upper()}"
            anc.append(self.prism_encoders[P](x[P]))
        anc.append(self.prism_encoder(x["PRISM__"]))
        
        anc = torch.cat(anc, dim=1)
        if self.prithvi_mode == "vit":
            latent = self.prithvi_encoder(x[self.img])
            latent = latent['last_hidden_state'][:,1:,:]
        elif self.prithvi_mode == "conv":
            latent = self.prithvi_encoder(x[self.img])
            latent = latent.flatten(2).transpose(1, 2)
        else:
            latent = self.prithvi_encoder.forward_features(x[self.img].unsqueeze(2))
            latent = latent[-1][:,1:,:]

        latent = self.fusion(latent, anc)
        latent = self.upscaler(latent)
        out = self.head(latent)

        return out


class PPM(torch.nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(pool_scale),
                    torch.nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        padding=0,
                    ),
                    torch.nn.BatchNorm2d(self.channels),
                    torch.nn.GELU(),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class Feature2Pyramid(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        rescales=(4, 2, 1, 0.5),
    ):
        super().__init__()
        self.rescales = rescales
        self.ops = torch.nn.ModuleList()

        for i, k in enumerate(self.rescales):
            if k == 4:
                self.ops.append(
                    torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                        torch.nn.BatchNorm2d(embed_dim[i]),
                        torch.nn.GELU(),
                        torch.nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                    )
                )
            elif k == 2:
                self.ops.append(
                    torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        )
                    )
                )
            elif k == 1:
                self.ops.append(torch.nn.Identity())
            elif k == 0.5:
                self.ops.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            elif k == 0.25:
                self.ops.append(torch.nn.MaxPool2d(kernel_size=4, stride=4))
            else:
                raise KeyError(f"invalid {k} for feature2pyramid")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []

        for i in range(len(inputs)):
            outputs.append(self.ops[i](inputs[i]))
        return tuple(outputs)


class PrithviUPerNet(torch.nn.Module):
    def __init__(
        self,
        img: str,
        dem: str,
        input_layers: list[int] = [5, 11, 17, 23],
        pool_scales=(1, 2, 3, 6),
        feature_multiplier: int = 1,
        **kwargs
    ):
        super().__init__()
        with open(CF_PATH, "r") as f:
            conf = json.load(f)
        
        self.img = img
        self.dem = dem
        
        prithvi_model = PrithviMAE(**conf["pretrained_cfg"])
        state_dict = torch.load(PT_PATH)
        prithvi_model.load_state_dict(state_dict)
        
        self.embed_dim = prithvi_model.encoder.embed_dim
        self.prithvi_encoder = copy.deepcopy(prithvi_model.encoder)

        self.input_layers = input_layers
        self.input_layers_num = len(self.input_layers)

        self.dem_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=False),
                torch.nn.BatchNorm2d(16),
                torch.nn.GELU(),
                DoubleConv(in_channels=16, out_channels=16, mid_channels=64)
            )
        self.prism_encoder = DoubleConv(in_channels=len(PRISM_ELEMENTS), out_channels=len(PRISM_ELEMENTS), mid_channels=len(PRISM_ELEMENTS)*2)
        self.prism_encoders = {}
        for p in PRISM_ELEMENTS:
            name = f"PRISM_{p.upper()}"
            self.prism_encoders[name] = DoubleConv(in_channels=360, out_channels=64, mid_channels=512)
        self.prism_encoders = torch.nn.ModuleDict(self.prism_encoders)
        self.solus_encoder = DoubleConv(in_channels=128, out_channels=128, mid_channels=256)
        
        C_anc = 16+128+((64+1)*len(PRISM_ELEMENTS))
        self.fusion = torch.nn.ModuleList([CrossAttentionFusion(C_latent=self.embed_dim, C_anc=C_anc) for _ in range(self.input_layers_num)])

        self.in_channels = [self.embed_dim * feature_multiplier for _ in self.input_layers]
        rescales = [4, 2, 1, 0.5]
        self.neck = Feature2Pyramid(embed_dim=self.in_channels, rescales=rescales)
    
        self.num_classes = 1
        self.align_corners = False
        
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.embed_dim,
            align_corners=self.align_corners,
        )

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.in_channels[-1] + len(pool_scales) * self.embed_dim,
                out_channels=self.embed_dim,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm2d(self.embed_dim),
            torch.nn.GELU(),
        )

        self.lateral_convs = torch.nn.ModuleList()
        self.fpn_convs = torch.nn.ModuleList()
        for in_channels in self.in_channels[:-1]:
            l_conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.embed_dim,
                    kernel_size=1,
                    padding=0,
                ),
                torch.nn.BatchNorm2d(self.embed_dim),
                torch.nn.GELU(),
            )
            fpn_conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim,
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.BatchNorm2d(self.embed_dim),
                torch.nn.GELU(),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=len(self.in_channels) * self.embed_dim,
                out_channels=self.embed_dim,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm2d(self.embed_dim),
            torch.nn.GELU(),
        )
        self.dropout = torch.nn.Dropout(.1)
        self.upscaler = Upscaler(self.embed_dim, depth=2)
        self.conv_reg = DoubleConv(self.embed_dim // 2**2, 1, mid_channels=self.embed_dim // 2**2, mid=False)

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, feats):
        laterals = [
            lateral_conv(feats[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(feats))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, x):
        feats = self.prithvi_encoder.forward_features(x[self.img].unsqueeze(2))

        anc = []
        anc.append(self.dem_encoder(x[self.dem]))
        anc.append(self.solus_encoder(x["SOLUS100"]))
        
        for p in PRISM_ELEMENTS:
            P = f"PRISM_{p.upper()}"
            anc.append(self.prism_encoders[P](x[P]))
        anc.append(self.prism_encoder(x["PRISM__"]))
        anc = torch.cat(anc, dim=1)
        
        anc_feats = []
        for i, layer in enumerate(self.input_layers):
            anc_feats.append(self.fusion[i](feats[layer][:,1:,:], anc))

        anc_feats = self.neck(anc_feats)
        anc_feats = self._forward_feature(anc_feats)
        anc_feats = self.dropout(anc_feats)
        anc_feats = self.upscaler(anc_feats)
        out = self.conv_reg(anc_feats)

        return out


def save_model(model, path, kwargs):
    torch.save({
        "model_kwargs": kwargs,
        "state_dict": model.state_dict()
    }, path)
    

def proc_sample(samples, stats, config):
    procs = {}
    for k, sample in samples.items():
        if k != "YEAR":
            sample = sample.to(DEVICE, DTYPE)
            if k != 'agb':
                if 'LS' in k and config['norm']:
                    sample = (sample - stats[f"MEANS_PRITHVI"]) / stats[f"STDDV_PRITHVI"]
                if k == 'PRISM__':
                    for i, p in enumerate(PRISM_ELEMENTS):
                        P = p.upper()
                        sample[:,i:i+1] = (sample[:,i:i+1] - stats[f"MEANS_PRISM_{P}"]) / stats[f"STDDV_PRISM_{P}"]
                else:
                    sample = (sample - stats[f"MEANS_{k}"]) / stats[f"STDDV_{k}"]
            sample[torch.isnan(sample)] = NO_DATA
        procs[k] = sample
    
    return procs


def main(config):
    with open(STAT_PATH) as f:
        stat = yaml.safe_load(f)
    
    stats = {}
    for k, s in stat.items():
        stats[k] = torch.tensor(s, device=DEVICE, dtype=DTYPE).view(-1, 1, 1)
    
    experiment_config = comet_ml.ExperimentConfig(name=config["name"], log_code=True, auto_metric_logging=False, log_git_metadata=False, log_git_patch=False)
    EXP = comet_ml.start(workspace='emapr', project_name="hudak_agb", experiment_config=experiment_config, mode="create")
    EXP.log_parameters(config)

    _OUT_DIR= TRAIN_OUT_DIR / config["name"]
    _CKPT_DIR = _OUT_DIR / 'ckpt'
    _PRED_DIR = _OUT_DIR / 'pred'
    _PROB_DIR = _OUT_DIR / 'prob'

    _OUT_DIR.mkdir(exist_ok=True, parents=True)
    _CKPT_DIR.mkdir(exist_ok=True)
    _PRED_DIR.mkdir(exist_ok=True)
    _PROB_DIR.mkdir(exist_ok=True)

    # instantiating model and optimizers
    # kwargs = {"img": config['img'], "dem": config['dem'], "prithvi": config['prithvi'], "ablate": config['ablate'], "norm": config["norm"]}
    # model = AttnSETRPUP(**kwargs)
    kwargs = {"img": config['img'], "dem": config['dem'], "norm": config["norm"]}
    model = PrithviUPerNet(**kwargs)
    model.to(DEVICE, DTYPE)
    
    ran_input = ({
        "YEAR": torch.tensor([2000]),
        config['img']: torch.randn(2, 6, 224, 224, device=DEVICE, dtype=DTYPE),
        config['dem']: torch.randn(2, 1, 224, 224, device=DEVICE, dtype=DTYPE),
        "SOLUS100": torch.randn(2, 128, 14, 14, device=DEVICE, dtype=DTYPE),
        **{f"PRISM_{p.upper()}": torch.randn(2, 360, 14, 14, device=DEVICE, dtype=DTYPE) for p in PRISM_ELEMENTS},
        "PRISM__": torch.randn(2,len(PRISM_ELEMENTS), 14, 14, device=DEVICE, dtype=DTYPE)
    },)
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        print(kwargs)
        summary(model, input_data=ran_input)

    cache = cache_anc(IMAGERY_DIR, config['img'], DIRS_YEARS[config['img']], PATCH_SIZE, PRISM_ELEMENTS, PRISM_FILES, SOLUS_FILES, DTYPE)
    
    # instantiating train and val dataloaders
    train_dataset = LidarUnitTrainDataset(config['img'], config['dem'], cache=cache)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=4)
    
    val_dataset = PlotValDataset(config['img'], config['dem'], cache=cache)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=4)
    
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0)
    criterion = MSELoss()

    # instantiating metric aggregation
    step = 0
    metric_agg = {}
    metric_mins = {}
    metric_mins['val_rmse'] = [(None, float('inf'))]
    metric_mins['train_rmse'] = [(None, float('inf'))]

    for epoch in range(config['num_epochs']):
        model.train()
        train_rmse = 0
        train_weight = 0
        for samples in train_loader:
            procs = proc_sample(samples, stats, config)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                pred = model(procs)    
            mse, N = criterion(pred, procs['agb'])
            loss = torch.sqrt(mse)
            loss.backward()
            optimizer.step()
            EXP.log_metrics({'train_rmse': loss.item()}, step=step, epoch=epoch)

            train_rmse += mse.item()*N
            train_weight += N
            step += 1
        train_rmse /= train_weight
        train_rmse **= .5
        metric_agg['train_rmse'] = train_rmse
        
        model.eval()
        val_rmse = 0
        val_weight = 0
        with torch.no_grad():
            for samples in val_loader:
                procs = proc_sample(samples, stats, config)
                with torch.autocast(device_type="cuda", dtype=DTYPE):
                    pred = model(procs)
                mse, N = criterion(pred, procs['agb'])
                val_rmse += mse*N
                val_weight += N

        val_rmse /= val_weight
        val_rmse **= .5
        metric_agg['val_rmse'] = val_rmse
        EXP.log_metrics({'val_rmse': val_rmse}, step=step, epoch=epoch)

        if epoch % config["chck_int"] == 0:
            for m, v in metric_agg.items():
                pm = metric_mins[m][-1]
                model_path = _CKPT_DIR/ f"{m}_{v:.4f}_s{step:05d}_e{epoch:04d}.pt"
                if len(metric_mins[m]) < TOP_CKPT:
                    metric_mins[m].append((model_path, v))
                    save_model(model, model_path, kwargs)
                elif v <= pm[1]:
                    if pm[0] is not None and pm[0].exists():
                        pm[0].unlink()
                    metric_mins[m].pop()
                    metric_mins[m].append((model_path, v))
                    save_model(model, model_path, kwargs)
                metric_mins[m] = sorted(metric_mins[m], key=lambda item: item[1])
        EXP.log_epoch_end(epoch)
    save_model(model, _CKPT_DIR / f"final_s{step:05d}_e{config['num_epochs']:04d}.pt", kwargs)
    EXP.end()
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("idx", type=int)
    # args = parser.parse_args()
    
    # hardcoding experiments
    # i could do a yaml file but alas
    config = {  "name": 'upernet',
                "num_epochs": 2048,
                "batch_size": 32,
                "lr": 1e-4,
                "chck_int": 1,
                "img": "LS",
                "dem": "NASADEM",
                "prithvi": "frozen",
                "norm": False}
    # ablates = [
    #     (['NASADEM', 'SOLUS100', 'PRISM__'], "initial_attempt"),
    #     (['NASADEM', 'SOLUS100', 'PRISM__', *[f"PRISM_{p.upper()}" for p in PRISM_ELEMENTS]], "initial_attempt_notfno")
    # ]
    # prithvis = ['frozen', 'frozen_pe', 'unfrozen', 'random', 'conv', 'vit']
    # configurations = [
    #     *[{"name": n, "ablate": a} for a, n in ablates],
    #     *[{"name": f"prithvi_{p}","ablate": ablates[0][0], "prithvi": p} for p in prithvis],
    #     {"name": "prithvi_norm", "ablate": ablates[0][0], "norm": True, "prithvi": "frozen"},
    #     {"name": "prithvi_norm_pe", "ablate": ablates[0][0], "norm": True, "prithvi": "frozen_pe"}
    # ]
    # assert args.idx > -1 and args.idx < len(configurations)

    # config.update(configurations[args.idx])
    main(config)
