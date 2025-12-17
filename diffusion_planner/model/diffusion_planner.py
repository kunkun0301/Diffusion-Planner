import torch
import torch.nn as nn

from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.decoder import Decoder
from diffusion_planner.model.module.long_short_decoder import LongShortDecoder


class Diffusion_Planner(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Diffusion_Planner_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)

    @property
    def sde(self):
        # keep compatibility with existing training code: ddp.get_model(model).sde.marginal_prob
        return self.decoder.decoder.sde

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)
        return encoder_outputs, decoder_outputs


class Diffusion_Planner_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Initialize embedding tables (Encoder-specific)
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):
        return self.encoder(inputs)


class Diffusion_Planner_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Select decoder implementation
        if getattr(config, "use_longshort_decoder", False):
            self.decoder = LongShortDecoder(config)
        else:
            self.decoder = Decoder(config)

        self.initialize_weights()

    def initialize_weights(self):
        # Basic init for all modules
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Apply DiT-specific init to all DiT instances used by the decoder
        dits = []
        if hasattr(self.decoder, "dit"):  # original Decoder
            dits = [self.decoder.dit]
        else:  # LongShortDecoder
            if hasattr(self.decoder, "dit_short"):
                dits.append(self.decoder.dit_short)
            if hasattr(self.decoder, "dit_full"):
                dits.append(self.decoder.dit_full)

        for dit in dits:
            # Timestep embedding MLP
            nn.init.normal_(dit.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(dit.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks
            for block in dit.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers
            nn.init.constant_(dit.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(dit.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(dit.final_layer.proj[-1].weight, 0)
            nn.init.constant_(dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs, inputs):
        return self.decoder(encoder_outputs, inputs)
