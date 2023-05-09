"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn

from dataset import Batch
from .lpips import LPIPS
from .nets2 import Encoder, Decoder, Dictionary
from utils import LossWithIntermediateLosses


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, with_lpips: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.learned_dict = Dictionary(512,
                                  (64 // 2*2,
                                  64 // 2*2),
                                  4, bottleneck_size=128)        
        self.encoder_decoder_model = Decoder(learned_dict=self.learned_dict)
        # self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(16, embed_dim, 1) # Changed 512 to 16
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, 16, 1) # Changed 512 to 16
        # self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None

        # learned_dict = Dictionary(args.num_classes,
        #                           (args.canvas_size // args.layer_size*2,
        #                           args.canvas_size // args.layer_size*2),
        #                           4, bottleneck_size=args.dim_z)
        # learned_dict.to(device)

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder_decoder_model(model_type = "encoder", im = x)
        # z = self.encoder(x)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()
        # print(z.shape)

        # Reshape to original
        # z = z.reshape(*shape[:-3], *z.shape[1:])
        # z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])

        z = z.reshape(2, *z.shape[1:])
        z_q = z_q.reshape(2, *z_q.shape[1:])

        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    # decode should take in og image (x) and the tokens (z_q)
    def decode(self, x, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.encoder_decoder_model(im = x, z_q = z_q)
        # rec = self.decoder(x, z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(x, z_q, should_postprocess)

    # @torch.no_grad()
    # def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
    #     z_q = self.encode(x, should_preprocess).z_quantized
    #     return self.forward(x)['image']


    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)
