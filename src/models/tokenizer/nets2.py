"""Model architecture.

Defintions:
    - layer: refers to a raster layer in the composite. Each layer is assembled
      from multiple patches. A rendering can have multiple layers ordered from
      back to front.
"""
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from .partialconv import PartialConv2d
from dataclasses import dataclass
from typing import List

@dataclass
class EncoderDecoderConfig:
    resolution: int
    in_channels: int
    z_channels: int
    ch: int
    ch_mult: List[int]
    num_res_blocks: int
    attn_resolutions: List[int]
    out_ch: int
    dropout: float


class Dictionary(nn.Module):
    def __init__(self, num_classes, patch_size, num_chans, bottleneck_size=128,
                 no_layernorm=False):
        super().__init__()

        self.patch_size = patch_size
        self.num_chans = num_chans
        self.no_layernorm = no_layernorm

        self.latent = nn.Parameter(th.randn(num_classes, bottleneck_size))
        self.decode = nn.Sequential(
            nn.Linear(bottleneck_size, 8*bottleneck_size),
            nn.GroupNorm(8, 8*bottleneck_size),
            nn.ReLU(inplace=True),
            nn.Linear(8*bottleneck_size,
                      num_chans*patch_size[0]*patch_size[1]),
            nn.Sigmoid()
        )

    def forward(self, x=None):
        if x is None and not self.no_layernorm:
            x = F.layer_norm(self.latent, (self.latent.shape[-1],))
        out = self.decode(x).view(-1, self.num_chans, *self.patch_size)
        return out, x

class DualDictionary(nn.Module): #Might not need the below class
    def __init__(self, num_classes, patch_size, num_chans, bottleneck_size=128,
                 no_layernorm=False):
        super().__init__()

        self.patch_size = patch_size
        self.num_chans = num_chans
        self.no_layernorm = no_layernorm

        self.latent = nn.Parameter(th.randn(num_classes, bottleneck_size))
        self.decode_A = nn.Sequential(
            nn.Linear(bottleneck_size, 8*bottleneck_size),
            nn.GroupNorm(8, 8*bottleneck_size),
            nn.ReLU(inplace=True),
            nn.Linear(8*bottleneck_size,
                      num_chans*patch_size[0]*patch_size[1]),
            nn.Sigmoid()
        )
        self.decode_B = nn.Sequential(
            nn.Linear(bottleneck_size, 8*bottleneck_size),
            nn.GroupNorm(8, 8*bottleneck_size),
            nn.ReLU(inplace=True),
            nn.Linear(8*bottleneck_size,
                      num_chans*patch_size[0]*patch_size[1]),
            nn.Sigmoid()
        )

    def forward(self, domain, x=None):
        decode = self.decode_A if domain == "A" else self.decode_B
        if x is None and not self.no_layernorm:
            x = F.layer_norm(self.latent, (self.latent.shape[-1],))
        out = decode(x).view(-1, self.num_chans, *self.patch_size)
        return out, x

class _DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, return_mask=True):
        super().__init__()
        self.return_mask = return_mask
        self.conv1 = PartialConv2d(in_ch, out_ch, 3, padding=1,
                                   return_mask=True)
        self.conv2 = PartialConv2d(out_ch, out_ch, 3, padding=1, stride=2,
                                   return_mask=True)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.nonlinearity = nn.LeakyReLU(inplace=True)

    def forward(self, x, mask=None):
        y, mask = self.conv1(x, mask)
        y = self.nonlinearity(self.norm1(y))
        y, mask = self.conv2(y, mask)
        y = self.nonlinearity(self.norm2(y))

        if self.return_mask:
            return y, mask
        else:
            return y

class Encoder(nn.Module):
    """Encodes image data into a grid of patch latent codes used for affinity
    matching.
    The encoder is a chain of blocks that each dowsamples by 2x. It outputs
    a list of latent codes for the patches in each layer. Each layer can have a
    variable (power of two) number of patches. The latent codes are predicted
    from the corresponding block.
    im
     |
     V
    block1 -> (optional) latent codes for all layers with 2x2 patches
     |
     V
    block2 -> (optional) latent codes for all layers with 4x4 patches
     |
     V
    ...    -> ...
    Args:
        num_channels(int): number of image channels (e.g. 4 for RGBA images).
        canvas_size(int): size of the (square) input image.
        layer_sizes(list of int): list of patch count along the x (resp. y)
            dimension for each layer. The number of layers is the length of
            the list.
    """

    def __init__(self, num_channels, canvas_size, layer_sizes, dim_z=1024,
                 no_layernorm=False):
    # def __init__(self):    
        super().__init__()

        # self.num_channels = 64
        # num_channels = 64

        # dim_z = 512
        # no_layernorm = False
        # self.canvas_size = 64
        # canvas_size = 64
        
        # self.layer_sizes = 2
        # layer_sizes = 2

        self.num_channels = num_channels

        self.canvas_size = canvas_size
        self.layer_sizes = layer_sizes
        self.no_layernorm = no_layernorm

        num_ds = int(np.log2(canvas_size / min(layer_sizes)))

        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        for i in range(num_ds):
            in_ch = num_channels if i == 0 else dim_z
            self.blocks.append(_DownBlock(in_ch, dim_z, return_mask=True))

            for lsize in layer_sizes:
                if canvas_size // (2**(i+1)) == lsize:
                    self.heads.append(PartialConv2d(
                        dim_z, dim_z, 3, padding=1))

    def forward(self, x):
        out = [None] * len(self.layer_sizes)

        y = x
        mask = None
        for _block in self.blocks:
            y, mask = _block(y, mask)  # encoding + downsampling step

            # Look for layers whose spatial dimension match the current block
            for i, l in enumerate(self.layer_sizes):
                if y.shape[-1] == l:
                    # size match, output the latent codes for this layer
                    out[i] = self.heads[i](y, mask).permute(
                        0, 2, 3, 1).contiguous()
                    if not self.no_layernorm:
                        out[i] = F.layer_norm(out[i], (out[i].shape[-1],))

        # Check all outputs were set
        for o in out:
            if o is None:
                raise RuntimeError("Unexpected output count for Encoder.")

        return out

class Decoder(nn.Module): #Decoder
    # def __init__(self, learned_dict, layer_size, num_layers, patch_size=1,
    #              canvas_size=128, dim_z=128, shuffle_all=False, bg_color=None,
    #              no_layernorm=False, no_spatial_transformer=False,
    #              spatial_transformer_bg=False, straight_through_probs=False):
    def __init__(self, learned_dict, patch_size=1,
                 canvas_size=128, dim_z=128, shuffle_all=False, bg_color=None,
                 no_layernorm=False, no_spatial_transformer=False,
                 spatial_transformer_bg=False, straight_through_probs=False):
        super().__init__()

        num_layers = 2
        layer_size = 16 
        self.layer_size = 16
        self.num_layers = num_layers
        self.canvas_size = 64
        self.patch_size = canvas_size // layer_size
        self.dim_z = 128
        self.shuffle_all = shuffle_all
        self.no_spatial_transformer = no_spatial_transformer
        self.spatial_transformer_bg = spatial_transformer_bg
        self.straight_through_probs = straight_through_probs

        self.im_encoder = Encoder(3, canvas_size, [layer_size]*num_layers,
                                  dim_z, no_layernorm=no_layernorm)

        # self.im_encoder = Encoder()

        self.project = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.LayerNorm(dim_z, elementwise_affine=False) if not no_layernorm
            else nn.Identity()
        )

        self.encoder_xform = Encoder(7, self.patch_size*2, [1], dim_z,
                                     no_layernorm=no_layernorm)
        self.probs = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.GroupNorm(8, dim_z),
            nn.LeakyReLU(),
            nn.Linear(dim_z, 1),
            nn.Sigmoid()
        )

        if self.no_spatial_transformer:
            self.xforms_x = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, self.patch_size+1),
                nn.Softmax(dim=-1)
            )
            self.xforms_y = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, self.patch_size+1),
                nn.Softmax(dim=-1)
            )
        else:
            self.shifts = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, 2),
                nn.Tanh()
            )

        self.learned_dict = learned_dict

        if bg_color is None:
            self.bg_encoder = Encoder(3, canvas_size, [1], dim_z,
                                      no_layernorm=no_layernorm)
            if self.spatial_transformer_bg:
                self.bg_shift = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, 1),
                    nn.Tanh()
                )
            else:
                self.bg_x = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, 3*canvas_size + 1),
                    nn.Softmax(dim=-1)
                )
        else:
            self.bg_color = nn.Parameter(
                th.tensor(bg_color), requires_grad=False)

    # should also take in tokens (z_q)
    def forward(self, model_type = None, im = None, z_q = None, bg = None, hard=False, custom_dict=None, rng=None, custom_bg=None):
        bg = '#737373'
        bs = im.shape[0]

        learned_dict, dict_codes = self.learned_dict()
        if rng is not None:
            learned_dict = learned_dict[rng]
            dict_codes = dict_codes[rng]

        if im == None:
            im_codes = z_q
        else:
            im_codes = th.stack(self.im_encoder(im), dim=1) # replace im_codes with z_q

        if model_type == 'encoder':
            return im_codes.flatten(0, 1)            

        probs = self.probs(im_codes.flatten(0, 3))
        if self.straight_through_probs:
            probs = probs.round() - probs.detach() + probs

        print('incodes:', self.project(im_codes).shape)
        print('dict_code:',dict_codes.transpose(0,1).shape)


        logits = (self.project(im_codes) @ dict_codes.transpose(0, 1)) \
            / np.sqrt(im_codes.shape[-1])
        weights = F.softmax(logits, dim=-1)

        patches = (weights[..., None, None, None] * learned_dict).sum(4)
        patches = patches.flatten(0, 3)

        im_patches = F.pad(im, (self.patch_size // 2,)*4) # dont change
        im_patches = im_patches.unfold(
            2, self.patch_size * 2, self.patch_size) \
            .unfold(3, self.patch_size * 2, self.patch_size)
        im_patches = im_patches.reshape(bs, 3, self.layer_size,
                                        self.layer_size, 2*self.patch_size,
                                        2*self.patch_size) \
            .permute(0, 2, 3, 1, 4, 5).contiguous()
        im_patches = im_patches[:, None].repeat(1,
                                                self.num_layers,
                                                1, 1, 1, 1, 1).flatten(0, 3)

        # codes_xform = self.encoder_xform(
        #     th.cat([im_patches, patches], dim=1))[0].squeeze(-2).squeeze(-2)
        codes_xform = self.encoder_xform(
            th.cat([0 * im_patches, patches], dim=1))[0].squeeze(-2).squeeze(-2)

        if hard:
            weights = th.eye(
                weights.shape[-1]).to(weights)[weights.argmax(-1)]
            probs = probs.round()
            patches = (weights[..., None, None, None] * learned_dict).sum(4)
            patches = patches.flatten(0, 3)

        if custom_dict is not None:
            learned_dict = custom_dict
            patches = (weights[..., None, None, None] * learned_dict).sum(4)
            patches = patches.flatten(0, 3)

        patches = patches * probs[:, :, None, None]

        if self.no_spatial_transformer:
            xforms_x = self.xforms_x(codes_xform)
            xforms_y = self.xforms_y(codes_xform)

            if hard:
                xforms_x = th.eye(
                    xforms_x.shape[-1]).to(xforms_x)[xforms_x.argmax(-1)]
                xforms_y = th.eye(
                    xforms_y.shape[-1]).to(xforms_y)[xforms_y.argmax(-1)]

            patches = F.pad(patches, (self.patch_size//2,)*4)
            patches = patches.unfold(2, self.patch_size * 2, 1)
            patches = (patches * xforms_y[:, None, :, None, None]).sum(2)
            patches = patches.unfold(2, self.patch_size * 2, 1)
            patches = (patches * xforms_x[:, None, :, None, None]).sum(2)
        else:
            shifts = self.shifts(codes_xform) / 2
            theta = th.eye(2)[None].repeat(shifts.shape[0], 1, 1).to(shifts)
            theta = th.cat([theta, -shifts[:, :, None]], dim=-1)
            grid = F.affine_grid(theta, [patches.shape[0], 1,
                                         self.patch_size*2, self.patch_size*2],
                                 align_corners=False)

            patches_rgb, patches_a = th.split(patches, [3, 1], dim=1)
            patches_rgb = F.grid_sample(patches_rgb, grid, align_corners=False,
                                        padding_mode='border', mode='bilinear')
            patches_a = F.grid_sample(patches_a, grid, align_corners=False,
                                      padding_mode='zeros', mode='bilinear')
            patches = th.cat([patches_rgb, patches_a], dim=1)

        patches = patches.view(
            bs, self.num_layers, self.layer_size, self.layer_size, -1,
            2*self.patch_size, 2*self.patch_size
        ).permute(0, 1, 4, 2, 5, 3, 6)

        group1 = patches[..., ::2, :, ::2, :].contiguous()
        group1 = group1.view(bs, self.num_layers, -1,
                             self.canvas_size, self.canvas_size)
        group1 = group1[..., self.patch_size//2:, self.patch_size//2:]
        group1 = F.pad(group1,
                       (0, self.patch_size//2, 0, self.patch_size//2))

        group2 = patches[..., 1::2, :, 1::2, :].contiguous()
        group2 = group2.view(bs, self.num_layers, -1,
                             self.canvas_size, self.canvas_size)
        group2 = group2[..., :-self.patch_size//2, :-self.patch_size//2]
        group2 = F.pad(group2,
                       (self.patch_size//2, 0, self.patch_size//2, 0))

        group3 = patches[..., 1::2, :, ::2, :].contiguous()
        group3 = group3.view(bs, self.num_layers, -1,
                             self.canvas_size, self.canvas_size)
        group3 = group3[..., :-self.patch_size//2, self.patch_size//2:]
        group3 = F.pad(group3,
                       (0, self.patch_size//2, self.patch_size//2, 0))

        group4 = patches[..., ::2, :, 1::2, :].contiguous()
        group4 = group4.view(bs, self.num_layers, -1,
                             self.canvas_size, self.canvas_size)
        group4 = group4[..., self.patch_size//2:, :-self.patch_size//2]
        group4 = F.pad(group4,
                       (self.patch_size//2, 0, 0, self.patch_size//2))

        layers = th.stack([group1, group2, group3, group4], dim=2)
        layers_out = layers.clone()

        if self.shuffle_all:
            layers = layers.flatten(1, 2)[:, th.randperm(4 * self.num_layers)]
        else:
            layers = layers[:, :, th.randperm(4)].flatten(1, 2)

        if bg is not None:
            bg_codes = self.bg_encoder(im)[0].squeeze(-2).squeeze(-2)
            if not self.spatial_transformer_bg:
                bg_x = self.bg_x(bg_codes)
                bgs = bg.squeeze(0).unfold(2, self.canvas_size, 1)
                out = (bgs[None] * bg_x[:, None, None, :, None]).sum(3)
            else:
                shift = self.bg_shift(bg_codes) * 3/4
                shift = th.cat([shift, th.zeros_like(shift)], dim=-1)
                theta = th.eye(2)[None].repeat(shift.shape[0], 1, 1).to(shift)
                theta[:, 0, 0] = 1/4
                theta = th.cat([theta, -shift[:, :, None]], dim=-1)
                grid = F.affine_grid(theta, [bs, 1, self.canvas_size,
                                             self.canvas_size],
                                     align_corners=False)

                out = F.grid_sample(bg.repeat(bs, 1, 1, 1), grid,
                                    align_corners=False, padding_mode='border',
                                    mode='bilinear')

        else:
            if custom_bg is not None:
                out = custom_bg[None, :, None, None].clamp(0, 1).repeat(
                    bs, 1, self.canvas_size, self.canvas_size)
            else:
                out = self.bg_color[None, :, None, None].clamp(0, 1).repeat(
                    bs, 1, self.canvas_size, self.canvas_size)
            bg = self.bg_color[None, :, None, None].clamp(0, 1).repeat(
                1, 1, self.canvas_size, self.canvas_size)

        rgb, a = th.split(layers, [3, 1], dim=2)

        for i in range(4 * self.num_layers):
            out = (1-a[:, i])*out + a[:, i]*rgb[:, i]

        ret = {
            "weights": weights,
            "probs": probs.view(bs, self.num_layers, -1),
            "layers": layers_out,
            "patches": patches,
            "dict_codes": dict_codes,
            "im_codes": im_codes.flatten(0, 1),
            "reconstruction": out,
            "dict": learned_dict,
            "background": bg
        }

        if not self.no_spatial_transformer:
            ret['shifts'] = shifts

        return ret
#Above


class _DualModel(nn.Module):
    def __init__(self, learned_dict, layer_size, num_layers, patch_size=1,
                 canvas_size=128, dim_z=128, shuffle_all=False, bg_color_A=None,
                 bg_color_B=None, no_layernorm=False, no_spatial_transformer=False,
                 spatial_transformer_bg=False, straight_through_probs=False):
        super().__init__()

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.canvas_size = canvas_size
        self.patch_size = canvas_size // layer_size
        self.dim_z = dim_z
        self.shuffle_all = shuffle_all
        self.no_spatial_transformer = no_spatial_transformer
        self.spatial_transformer_bg = spatial_transformer_bg
        self.straight_through_probs = straight_through_probs

        self.learned_dict = learned_dict

        self.encoder = {'A': self._Encoder(canvas_size=canvas_size, layer_size=layer_size, num_layers=num_layers,
                                dim_z=dim_z, no_layernorm=no_layernorm, bg_color=bg_color_A),
                        'B': self._Encoder(canvas_size=canvas_size, layer_size=layer_size, num_layers=num_layers,
                                dim_z=dim_z, no_layernorm=no_layernorm, bg_color=bg_color_B)}

        self.decoder = {'A': self._Decoder(dim_z=dim_z, bg_color=bg_color_A, canvas_size=canvas_size),
                        'B': self._Decoder(dim_z=dim_z, bg_color=bg_color_B, canvas_size=canvas_size)}

        self.disc =     {'A': self._Discriminator(),
                         'B': self._Discriminator()}

    def forward(self, im, bg, hard=False, custom_dict=None, rng=None, custom_bg=None, source_domain='A', target_domain='A'):
        encoder = self.encoder[source_domain]
        decoder = self.decoder[target_domain]

        learned_dict, dict_codes = self.learned_dict(source_domain)
        if rng is not None:
            learned_dict = learned_dict[rng]
            dict_codes = dict_codes[rng]

        encoding = encoder(im, learned_dict, dict_codes)
        ret = decoder(im, bg, custom_bg, encoding, hard, custom_dict)
        ret["dict_codes"] = dict_codes
        ret["im_codes"] = encoding["im_codes"].flatten(0, 1)

    class _Encoder(nn.Module):
        def __init__(self, canvas_size, layer_size, num_layers, dim_z, no_layernorm, bg_color):
            super().__init__()

            self.canvas_size = canvas_size
            self.layer_size = layer_size
            self.num_layers = num_layers
            self.dim_z = dim_z
            self.no_layernorm = no_layernorm

            self.im_encoder = Encoder(3, canvas_size, [layer_size]*num_layers,
                                  dim_z, no_layernorm=no_layernorm)

            self.probs = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, 1),
                nn.Sigmoid()
            )

            self.project = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.LayerNorm(dim_z, elementwise_affine=False) if not no_layernorm
                else nn.Identity()
            )

            self.encoder_xform = Encoder(7, self.patch_size*2, [1], dim_z,
                                        no_layernorm=no_layernorm)

            if bg_color is None:
                self.bg_encoder = Encoder(3, canvas_size, [1], dim_z,
                                        no_layernorm=no_layernorm)

        def forward(self, im, learned_dict, dict_codes):
            bs = im.shape[0]

            im_codes = th.stack(self.im_encoder(im), dim=1)

            probs = self.probs(im_codes.flatten(0, 3))
            if self.straight_through_probs:
                probs = probs.round() - probs.detach() + probs

            logits = (self.project(im_codes) @ dict_codes.transpose(0, 1)) \
                / np.sqrt(im_codes.shape[-1])
            weights = F.softmax(logits, dim=-1)

            patches = (weights[..., None, None, None] * learned_dict).sum(4)
            patches = patches.flatten(0, 3)

            im_patches = F.pad(im, (self.patch_size // 2,)*4)
            im_patches = im_patches.unfold(
                2, self.patch_size * 2, self.patch_size) \
                .unfold(3, self.patch_size * 2, self.patch_size)
            im_patches = im_patches.reshape(bs, 3, self.layer_size,
                                            self.layer_size, 2*self.patch_size,
                                            2*self.patch_size) \
                .permute(0, 2, 3, 1, 4, 5).contiguous()
            im_patches = im_patches[:, None].repeat(1,
                                                    self.num_layers,
                                                    1, 1, 1, 1, 1).flatten(0, 3)

            codes_xform = self.encoder_xform(
                th.cat([im_patches, patches], dim=1))[0].squeeze(-2).squeeze(-2)

            ret = { "weights": weights,
                    "patches": patches,
                    "codes_xform": codes_xform,
                    "im_codes": im_codes}

            return ret

    class _Decoder(nn.Module):
        def __init__(self, dim_z, bg_color, canvas_size):
            super().__init__()

            self.dim_z = dim_z
            self.bg_color = bg_color
            self.canvas_size = canvas_size

            if self.no_spatial_transformer:
                self.xforms_x = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, self.patch_size+1),
                    nn.Softmax(dim=-1)
                )
                self.xforms_y = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, self.patch_size+1),
                    nn.Softmax(dim=-1)
                )
            else:
                self.shifts = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, 2),
                    nn.Tanh()
                )

            if bg_color is None:
                if self.spatial_transformer_bg:
                    self.bg_shift = nn.Sequential(
                        nn.Linear(dim_z, dim_z),
                        nn.GroupNorm(8, dim_z),
                        nn.LeakyReLU(),
                        nn.Linear(dim_z, 1),
                        nn.Tanh()
                    )
                else:
                    self.bg_x = nn.Sequential(
                        nn.Linear(dim_z, dim_z),
                        nn.GroupNorm(8, dim_z),
                        nn.LeakyReLU(),
                        nn.Linear(dim_z, 3*canvas_size + 1),
                        nn.Softmax(dim=-1)
                    )
            else:
                self.bg_color = nn.Parameter(
                    th.tensor(bg_color), requires_grad=False)

        def forward(self, im, bg, custom_bg, encoding, hard, custom_dict):
            bs = im.shape[0]
            weights = encoding["weights"]
            patches = encoding["patches"]
            codes_xform = encoding["codes_xform"]

            if hard:
                weights = th.eye(
                    weights.shape[-1]).to(weights)[weights.argmax(-1)]
                probs = probs.round()
                patches = (weights[..., None, None, None] * learned_dict).sum(4)
                patches = patches.flatten(0, 3)

            if custom_dict is not None:
                learned_dict = custom_dict
                patches = (weights[..., None, None, None] * learned_dict).sum(4)
                patches = patches.flatten(0, 3)

            patches = patches * probs[:, :, None, None]

            if self.no_spatial_transformer:
                xforms_x = self.xforms_x(codes_xform)
                xforms_y = self.xforms_y(codes_xform)

                if hard:
                    xforms_x = th.eye(
                        xforms_x.shape[-1]).to(xforms_x)[xforms_x.argmax(-1)]
                    xforms_y = th.eye(
                        xforms_y.shape[-1]).to(xforms_y)[xforms_y.argmax(-1)]

                patches = F.pad(patches, (self.patch_size//2,)*4)
                patches = patches.unfold(2, self.patch_size * 2, 1)
                patches = (patches * xforms_y[:, None, :, None, None]).sum(2)
                patches = patches.unfold(2, self.patch_size * 2, 1)
                patches = (patches * xforms_x[:, None, :, None, None]).sum(2)
            else:
                shifts = self.shifts(codes_xform) / 2
                theta = th.eye(2)[None].repeat(shifts.shape[0], 1, 1).to(shifts)
                theta = th.cat([theta, -shifts[:, :, None]], dim=-1)
                grid = F.affine_grid(theta, [patches.shape[0], 1,
                                            self.patch_size*2, self.patch_size*2],
                                    align_corners=False)

                patches_rgb, patches_a = th.split(patches, [3, 1], dim=1)
                patches_rgb = F.grid_sample(patches_rgb, grid, align_corners=False,
                                            padding_mode='border', mode='bilinear')
                patches_a = F.grid_sample(patches_a, grid, align_corners=False,
                                        padding_mode='zeros', mode='bilinear')
                patches = th.cat([patches_rgb, patches_a], dim=1)

            patches = patches.view(
                bs, self.num_layers, self.layer_size, self.layer_size, -1,
                2*self.patch_size, 2*self.patch_size
            ).permute(0, 1, 4, 2, 5, 3, 6)

            group1 = patches[..., ::2, :, ::2, :].contiguous()
            group1 = group1.view(bs, self.num_layers, -1,
                                self.canvas_size, self.canvas_size)
            group1 = group1[..., self.patch_size//2:, self.patch_size//2:]
            group1 = F.pad(group1,
                        (0, self.patch_size//2, 0, self.patch_size//2))

            group2 = patches[..., 1::2, :, 1::2, :].contiguous()
            group2 = group2.view(bs, self.num_layers, -1,
                                self.canvas_size, self.canvas_size)
            group2 = group2[..., :-self.patch_size//2, :-self.patch_size//2]
            group2 = F.pad(group2,
                        (self.patch_size//2, 0, self.patch_size//2, 0))

            group3 = patches[..., 1::2, :, ::2, :].contiguous()
            group3 = group3.view(bs, self.num_layers, -1,
                                self.canvas_size, self.canvas_size)
            group3 = group3[..., :-self.patch_size//2, self.patch_size//2:]
            group3 = F.pad(group3,
                        (0, self.patch_size//2, self.patch_size//2, 0))

            group4 = patches[..., ::2, :, 1::2, :].contiguous()
            group4 = group4.view(bs, self.num_layers, -1,
                                self.canvas_size, self.canvas_size)
            group4 = group4[..., self.patch_size//2:, :-self.patch_size//2]
            group4 = F.pad(group4,
                        (self.patch_size//2, 0, 0, self.patch_size//2))

            layers = th.stack([group1, group2, group3, group4], dim=2)
            layers_out = layers.clone()

            if self.shuffle_all:
                layers = layers.flatten(1, 2)[:, th.randperm(4 * self.num_layers)]
            else:
                layers = layers[:, :, th.randperm(4)].flatten(1, 2)

            if bg is not None:
                bg_codes = self.bg_encoder(im)[0].squeeze(-2).squeeze(-2)
                if not self.spatial_transformer_bg:
                    bg_x = self.bg_x(bg_codes)
                    bgs = bg.squeeze(0).unfold(2, self.canvas_size, 1)
                    out = (bgs[None] * bg_x[:, None, None, :, None]).sum(3)
                else:
                    shift = self.bg_shift(bg_codes) * 3/4
                    shift = th.cat([shift, th.zeros_like(shift)], dim=-1)
                    theta = th.eye(2)[None].repeat(shift.shape[0], 1, 1).to(shift)
                    theta[:, 0, 0] = 1/4
                    theta = th.cat([theta, -shift[:, :, None]], dim=-1)
                    grid = F.affine_grid(theta, [bs, 1, self.canvas_size,
                                                self.canvas_size],
                                        align_corners=False)

                    out = F.grid_sample(bg.repeat(bs, 1, 1, 1), grid,
                                        align_corners=False, padding_mode='border',
                                        mode='bilinear')

            else:
                if custom_bg is not None:
                    out = custom_bg[None, :, None, None].clamp(0, 1).repeat(
                        bs, 1, self.canvas_size, self.canvas_size)
                else:
                    out = self.bg_color[None, :, None, None].clamp(0, 1).repeat(
                        bs, 1, self.canvas_size, self.canvas_size)
                bg = self.bg_color[None, :, None, None].clamp(0, 1).repeat(
                    1, 1, self.canvas_size, self.canvas_size)

            rgb, a = th.split(layers, [3, 1], dim=2)

            for i in range(4 * self.num_layers):
                out = (1-a[:, i])*out + a[:, i]*rgb[:, i]

            ret = {
                "weights": weights,
                "probs": probs.view(bs, self.num_layers, -1),
                "layers": layers_out,
                "patches": patches,
                "reconstruction": out,
                "dict": learned_dict,
                "background": bg
            }

            return ret

    class _Discriminator(nn.Module):
        def __init__(self):
            super(DualModel._Discriminator, self).__init__()
            nc = 3
            ndf = 64
            self.main = nn.Sequential(
                # input is ``(nc) x 64 x 64``
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf) x 32 x 32``
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*2) x 16 x 16``
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*4) x 8 x 8``
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*8) x 4 x 4``
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

class DualModel(nn.Module):
    def __init__(self, learned_dict, layer_size, num_layers, patch_size=1,
                 canvas_size=128, dim_z=128, shuffle_all=False, bg_color_A=None,
                 bg_color_B=None, no_layernorm=False, no_spatial_transformer=False,
                 spatial_transformer_bg=False, straight_through_probs=False):
        super().__init__()

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.canvas_size = canvas_size
        self.patch_size = canvas_size // layer_size
        self.dim_z = dim_z
        self.shuffle_all = shuffle_all
        self.no_spatial_transformer = no_spatial_transformer
        self.spatial_transformer_bg = spatial_transformer_bg
        self.straight_through_probs = straight_through_probs

        self.im_encoder_A = Encoder(3, canvas_size, [layer_size]*num_layers,
                                  dim_z, no_layernorm=no_layernorm)

        self.project_A = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.LayerNorm(dim_z, elementwise_affine=False) if not no_layernorm
            else nn.Identity()
        )

        self.encoder_xform_A = Encoder(7, self.patch_size*2, [1], dim_z,
                                     no_layernorm=no_layernorm)
        self.probs_A = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.GroupNorm(8, dim_z),
            nn.LeakyReLU(),
            nn.Linear(dim_z, 1),
            nn.Sigmoid()
        )

        self.im_encoder_B = Encoder(3, canvas_size, [layer_size]*num_layers,
                                  dim_z, no_layernorm=no_layernorm)

        self.project_B = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.LayerNorm(dim_z, elementwise_affine=False) if not no_layernorm
            else nn.Identity()
        )

        self.encoder_xform_B = Encoder(7, self.patch_size*2, [1], dim_z,
                                     no_layernorm=no_layernorm)
        self.probs_B = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.GroupNorm(8, dim_z),
            nn.LeakyReLU(),
            nn.Linear(dim_z, 1),
            nn.Sigmoid()
        )

        if self.no_spatial_transformer:
            self.xforms_x_A = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, self.patch_size+1),
                nn.Softmax(dim=-1)
            )
            self.xforms_y_A = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, self.patch_size+1),
                nn.Softmax(dim=-1)
            )

            self.xforms_x_B = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, self.patch_size+1),
                nn.Softmax(dim=-1)
            )
            self.xforms_y_B = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, self.patch_size+1),
                nn.Softmax(dim=-1)
            )
        else:
            self.shifts_A = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, 2),
                nn.Tanh()
            )

            self.shifts_B = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, 2),
                nn.Tanh()
            )

        self.learned_dict = learned_dict

        if bg_color_A is None:
            self.bg_encoder_A = Encoder(3, canvas_size, [1], dim_z,
                                      no_layernorm=no_layernorm)
            if self.spatial_transformer_bg:
                self.bg_shift_A = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, 1),
                    nn.Tanh()
                )
            else:
                self.bg_x_A = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, 3*canvas_size + 1),
                    nn.Softmax(dim=-1)
                )
        else:
            self.bg_color_A = nn.Parameter(
                th.tensor(bg_color_A), requires_grad=False)

        if bg_color_B is None:
            self.bg_encoder_B = Encoder(3, canvas_size, [1], dim_z,
                                      no_layernorm=no_layernorm)
            if self.spatial_transformer_bg:
                self.bg_shift_B = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, 1),
                    nn.Tanh()
                )
            else:
                self.bg_x_B = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, 3*canvas_size + 1),
                    nn.Softmax(dim=-1)
                )
        else:
            self.bg_color_B = nn.Parameter(
                th.tensor(bg_color_B), requires_grad=False)

        self.disc_A = Discriminator()
        self.disc_B = Discriminator()

    def forward(self, im, bgs, hard=False, custom_dict=None, rng=None, custom_bg=None, cross=False, cycle=False):
        im_a = im[:,0,:,:,:]
        im_b = im[:,1,:,:,:]

        if not (cross or cycle):
            source_domain = ['A', 'B']
            target_domain = ['A', 'B']
        elif cross:
            source_domain = ['A', 'B']
            target_domain = ['B', 'A']
        elif cycle:
            source_domain = ['B', 'A']
            target_domain = ['A', 'B']

        ret_a = self._forward(im_a, bgs[0], hard=hard, custom_dict=custom_dict, rng=rng, custom_bg=custom_bg,
                            source_domain=source_domain[0], target_domain=target_domain[0])
        ret_b = self._forward(im_b, bgs[1], hard=hard, custom_dict=custom_dict, rng=rng, custom_bg=custom_bg,
                            source_domain=source_domain[1], target_domain=target_domain[1])

        out_A = ret_a["reconstruction"]
        out_B = ret_b["reconstruction"]
        sh_A = list(out_A.shape)
        sh_B = list(out_B.shape)
        sh_A.insert(1,1)
        sh_B.insert(1,1)
        out_A = th.reshape(out_A, sh_A)
        out_B = th.reshape(out_B, sh_B)
        out = th.cat([out_A, out_B], axis=1)

        learned_dict_A, dict_codes = self.learned_dict("A")
        learned_dict_B, dict_codes = self.learned_dict("B")

        layers_A = ret_a["layers"]
        layers_B = ret_b["layers"]
        sh_A = list(layers_A.shape)
        sh_B = list(layers_B.shape)
        sh_A.insert(1,1)
        sh_B.insert(1,1)
        layers_A = th.reshape(layers_A, sh_A)
        layers_B = th.reshape(layers_B, sh_B)
        layers = th.cat([layers_A,layers_B], axis=1)

        ret = { "reconstruction": out,
                "layers": layers,
                "dict": [learned_dict_A, learned_dict_B],
                "dict_codes": dict_codes,
                "weights_A": ret_a["weights"],
                "weights_B": ret_b["weights"],
                "probs_A": ret_a["probs"],
                "probs_B": ret_b["probs"],
                "im_codes": [ret_a["im_codes"], ret_b["im_codes"]],
                "background_A": ret_a["background"],
                "background_B": ret_b["background"]
                }

        return ret

    def _forward(self, im, bg, hard=False, custom_dict=None, rng=None, custom_bg=None, source_domain='A', target_domain='A'):
        if source_domain == 'A':
            _im_encoder = self.im_encoder_A
            _encoder_xform = self.encoder_xform_A
            _project = self.project_A
            if bg is not None:
                _bg_encoder = self.bg_encoder_A

        elif source_domain == 'B':
            _im_encoder = self.im_encoder_B
            _encoder_xform = self.encoder_xform_B
            _project = self.project_B
            if bg is not None:
                _bg_encoder = self.bg_encoder_B

        if target_domain == 'A':
            _probs = self.probs_A
            if self.no_spatial_transformer:
                _xforms_x = self.xforms_x_A
                _xforms_y = self.xforms_y_A
            else:
                _shifts = self.shifts_A
            if bg is not None:
                if self.spatial_transformer_bg:
                    _bg_shift = self.bg_shift_A
                else:
                    _bg_x = self.bg_x_A
            else:
                _bg_color = self.bg_color_A

        elif target_domain == 'B':
            _probs = self.probs_B
            if self.no_spatial_transformer:
                _xforms_x = self.xforms_x_B
                _xforms_y = self.xforms_y_B
            else:
                _shifts = self.shifts_B
            if bg is not None:
                if self.spatial_transformer_bg:
                    _bg_shift = self.bg_shift_B
                else:
                    _bg_x = self.bg_x_B
            else:
                _bg_color = self.bg_color_B

        bs = im.shape[0]

        learned_dict, dict_codes = self.learned_dict(source_domain)
        if rng is not None:
            learned_dict = learned_dict[rng]
            dict_codes = dict_codes[rng]

        im_codes = th.stack(_im_encoder(im), dim=1)
        probs = _probs(im_codes.flatten(0, 3))
        if self.straight_through_probs:
            probs = probs.round() - probs.detach() + probs

        logits = (_project(im_codes) @ dict_codes.transpose(0, 1)) \
            / np.sqrt(im_codes.shape[-1])
        weights = F.softmax(logits, dim=-1)

        patches = (weights[..., None, None, None] * learned_dict).sum(4)
        patches = patches.flatten(0, 3)

        im_patches = F.pad(im, (self.patch_size // 2,)*4)
        im_patches = im_patches.unfold(
            2, self.patch_size * 2, self.patch_size) \
            .unfold(3, self.patch_size * 2, self.patch_size)
        im_patches = im_patches.reshape(bs, 3, self.layer_size,
                                        self.layer_size, 2*self.patch_size,
                                        2*self.patch_size) \
            .permute(0, 2, 3, 1, 4, 5).contiguous()
        im_patches = im_patches[:, None].repeat(1,
                                                self.num_layers,
                                                1, 1, 1, 1, 1).flatten(0, 3)

        codes_xform = _encoder_xform(
            th.cat([im_patches, patches], dim=1))[0].squeeze(-2).squeeze(-2)

        if hard:
            weights = th.eye(
                weights.shape[-1]).to(weights)[weights.argmax(-1)]
            probs = probs.round()
            patches = (weights[..., None, None, None] * learned_dict).sum(4)
            patches = patches.flatten(0, 3)

        if custom_dict is not None:
            learned_dict = custom_dict
            patches = (weights[..., None, None, None] * learned_dict).sum(4)
            patches = patches.flatten(0, 3)

        patches = patches * probs[:, :, None, None]

        if self.no_spatial_transformer:
            xforms_x = _xforms_x(codes_xform)
            xforms_y = _xforms_y(codes_xform)

            if hard:
                xforms_x = th.eye(
                    xforms_x.shape[-1]).to(xforms_x)[xforms_x.argmax(-1)]
                xforms_y = th.eye(
                    xforms_y.shape[-1]).to(xforms_y)[xforms_y.argmax(-1)]

            patches = F.pad(patches, (self.patch_size//2,)*4)
            patches = patches.unfold(2, self.patch_size * 2, 1)
            patches = (patches * xforms_y[:, None, :, None, None]).sum(2)
            patches = patches.unfold(2, self.patch_size * 2, 1)
            patches = (patches * xforms_x[:, None, :, None, None]).sum(2)
        else:
            shifts = _shifts(codes_xform) / 2
            theta = th.eye(2)[None].repeat(shifts.shape[0], 1, 1).to(shifts)
            theta = th.cat([theta, -shifts[:, :, None]], dim=-1)
            grid = F.affine_grid(theta, [patches.shape[0], 1,
                                         self.patch_size*2, self.patch_size*2],
                                 align_corners=False)

            patches_rgb, patches_a = th.split(patches, [3, 1], dim=1)
            patches_rgb = F.grid_sample(patches_rgb, grid, align_corners=False,
                                        padding_mode='border', mode='bilinear')
            patches_a = F.grid_sample(patches_a, grid, align_corners=False,
                                      padding_mode='zeros', mode='bilinear')
            patches = th.cat([patches_rgb, patches_a], dim=1)

        patches = patches.view(
            bs, self.num_layers, self.layer_size, self.layer_size, -1,
            2*self.patch_size, 2*self.patch_size
        ).permute(0, 1, 4, 2, 5, 3, 6)

        group1 = patches[..., ::2, :, ::2, :].contiguous()
        group1 = group1.view(bs, self.num_layers, -1,
                             self.canvas_size, self.canvas_size)
        group1 = group1[..., self.patch_size//2:, self.patch_size//2:]
        group1 = F.pad(group1,
                       (0, self.patch_size//2, 0, self.patch_size//2))

        group2 = patches[..., 1::2, :, 1::2, :].contiguous()
        group2 = group2.view(bs, self.num_layers, -1,
                             self.canvas_size, self.canvas_size)
        group2 = group2[..., :-self.patch_size//2, :-self.patch_size//2]
        group2 = F.pad(group2,
                       (self.patch_size//2, 0, self.patch_size//2, 0))

        group3 = patches[..., 1::2, :, ::2, :].contiguous()
        group3 = group3.view(bs, self.num_layers, -1,
                             self.canvas_size, self.canvas_size)
        group3 = group3[..., :-self.patch_size//2, self.patch_size//2:]
        group3 = F.pad(group3,
                       (0, self.patch_size//2, self.patch_size//2, 0))

        group4 = patches[..., ::2, :, 1::2, :].contiguous()
        group4 = group4.view(bs, self.num_layers, -1,
                             self.canvas_size, self.canvas_size)
        group4 = group4[..., self.patch_size//2:, :-self.patch_size//2]
        group4 = F.pad(group4,
                       (self.patch_size//2, 0, 0, self.patch_size//2))

        layers = th.stack([group1, group2, group3, group4], dim=2)
        layers_out = layers.clone()

        if self.shuffle_all:
            layers = layers.flatten(1, 2)[:, th.randperm(4 * self.num_layers)]
        else:
            layers = layers[:, :, th.randperm(4)].flatten(1, 2)

        if bg is not None:
            bg_codes = _bg_encoder(im)[0].squeeze(-2).squeeze(-2)
            if not self.spatial_transformer_bg:
                bg_x = _bg_x(bg_codes)
                bgs = bg.squeeze(0).unfold(2, self.canvas_size, 1)
                out = (bgs[None] * bg_x[:, None, None, :, None]).sum(3)
            else:
                shift = _bg_shift(bg_codes) * 3/4
                shift = th.cat([shift, th.zeros_like(shift)], dim=-1)
                theta = th.eye(2)[None].repeat(shift.shape[0], 1, 1).to(shift)
                theta[:, 0, 0] = 1/4
                theta = th.cat([theta, -shift[:, :, None]], dim=-1)
                grid = F.affine_grid(theta, [bs, 1, self.canvas_size,
                                             self.canvas_size],
                                     align_corners=False)

                out = F.grid_sample(bg.repeat(bs, 1, 1, 1), grid,
                                    align_corners=False, padding_mode='border',
                                    mode='bilinear')

        else:
            if custom_bg is not None:
                out = custom_bg[None, :, None, None].clamp(0, 1).repeat(
                    bs, 1, self.canvas_size, self.canvas_size)
            else:
                out = _bg_color[None, :, None, None].clamp(0, 1).repeat(
                    bs, 1, self.canvas_size, self.canvas_size)
            bg = _bg_color[None, :, None, None].clamp(0, 1).repeat(
                1, 1, self.canvas_size, self.canvas_size)

        rgb, a = th.split(layers, [3, 1], dim=2)

        for i in range(4 * self.num_layers):
            out = (1-a[:, i])*out + a[:, i]*rgb[:, i]

        ret = {
            "weights": weights,
            "probs": probs.view(bs, self.num_layers, -1),
            "layers": layers_out,
            "patches": patches,
            "dict_codes": dict_codes,
            "im_codes": im_codes.flatten(0, 1),
            "reconstruction": out,
            "dict": learned_dict,
            "background": bg
        }

        if not self.no_spatial_transformer:
            ret['shifts_' + source_domain + source_domain] = shifts

        return ret

class _Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        nc = 3
        ndf = 64
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim=3):
        super(Discriminator, self).__init__()

        params = {"n_layer": 4, "gan_type": "lsgan", "dim": 64, "norm": "none", "activ": "lrelu", "num_scales": 3, "pad_type": "reflect"}

        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += th.mean((out0 - 0)**2) + th.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(th.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(th.ones_like(out1.data).cuda(), requires_grad=False)
                loss += th.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += th.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(th.ones_like(out0.data).cuda(), requires_grad=False)
                loss += th.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', th.zeros(num_features))
        self.register_buffer('running_var', th.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(th.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(th.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(th.mv(th.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(th.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
