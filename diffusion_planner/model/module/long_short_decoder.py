import torch
import torch.nn as nn

from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.model.module.decoder import DiT, RouteEncoder


class LongShortDecoder(nn.Module):
    """
      - Training requires:
          inputs["sampled_trajectories_full"]  : [B,P,future_len+1,4]
          inputs["sampled_trajectories_short"] : [B,P,short_future_len+1,4]
          inputs["diffusion_time"]
      - If any key is missing, raise KeyError immediately.
    """

    def __init__(self, config):
        super().__init__()

        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._short_future_len = config.short_future_len
        self._sde = VPSDE_linear()

        route_encoder = RouteEncoder(
            config.route_num,
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim
        )

        self.dit_short = DiT(
            sde=self._sde,
            route_encoder=route_encoder,
            depth=config.decoder_depth,
            output_dim=(self._short_future_len + 1) * 4,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=config.decoder_drop_path_rate,
            model_type=config.diffusion_model_type
        )

        self.dit_full = DiT(
            sde=self._sde,
            route_encoder=route_encoder,
            depth=config.decoder_depth,
            output_dim=(self._future_len + 1) * 4,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=config.decoder_drop_path_rate,
            model_type=config.diffusion_model_type
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        self._guidance_fn = config.guidance_fn

    @property
    def sde(self):
        return self._sde

    def _prepare_common_inputs(self, encoder_outputs, inputs):
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1)

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        return current_states, neighbor_current_mask, ego_neighbor_encoding, route_lanes, B, P

    def _forward_train(self, encoder_outputs, inputs):
        current_states, neighbor_current_mask, ego_neighbor_encoding, route_lanes, B, P = \
            self._prepare_common_inputs(encoder_outputs, inputs)
            
        if "sampled_trajectories_full" not in inputs:
            raise KeyError(
                "Missing key inputs['sampled_trajectories_full'] (scheme-1 required). "
                "It should be [B,P,future_len+1,4] with short clean + tail noised."
            )
        if "sampled_trajectories_short" not in inputs:
            raise KeyError(
                "Missing key inputs['sampled_trajectories_short'] (scheme-1 required). "
                "It should be [B,P,short_future_len+1,4] with short frames noised."
            )
        if "diffusion_time" not in inputs:
            raise KeyError("Missing key inputs['diffusion_time'].")

        sampled_full_traj = inputs["sampled_trajectories_full"]
        sampled_short_traj = inputs["sampled_trajectories_short"]
        diffusion_time = inputs["diffusion_time"]  # scalar or [B]

        # Optional sanity checks (shape only; will raise AssertionError if mismatch)
        assert sampled_full_traj.dim() == 4 and sampled_full_traj.shape[:2] == (B, P), \
            f"sampled_trajectories_full shape mismatch, got {tuple(sampled_full_traj.shape)}, expected [B,P,*,4]"
        assert sampled_full_traj.shape[-1] == 4, \
            f"sampled_trajectories_full last dim must be 4, got {sampled_full_traj.shape[-1]}"
        assert sampled_short_traj.dim() == 4 and sampled_short_traj.shape[:2] == (B, P), \
            f"sampled_trajectories_short shape mismatch, got {tuple(sampled_short_traj.shape)}, expected [B,P,*,4]"
        assert sampled_short_traj.shape[-1] == 4, \
            f"sampled_trajectories_short last dim must be 4, got {sampled_short_traj.shape[-1]}"
        assert sampled_full_traj.shape[2] == self._future_len + 1, \
            f"sampled_trajectories_full time dim must be future_len+1={self._future_len+1}, got {sampled_full_traj.shape[2]}"
        assert sampled_short_traj.shape[2] == self._short_future_len + 1, \
            f"sampled_trajectories_short time dim must be short_future_len+1={self._short_future_len+1}, got {sampled_short_traj.shape[2]}"

        # ---- Full horizon ----
        sampled_full = sampled_full_traj.reshape(B, P, -1)
        score_full = self.dit_full(
            sampled_full,
            diffusion_time,
            ego_neighbor_encoding,
            route_lanes,
            neighbor_current_mask,
        ).reshape(B, P, self._future_len + 1, 4)

        # ---- Short horizon ----
        short_len = self._short_future_len
        sampled_short = sampled_short_traj.reshape(B, P, -1)
        score_short = self.dit_short(
            sampled_short,
            diffusion_time,
            ego_neighbor_encoding,
            route_lanes,
            neighbor_current_mask,
        ).reshape(B, P, short_len + 1, 4)

        return {"score": score_full, "score_short": score_short}

    # -------------------------
    # Sampling: short (pivot)
    # -------------------------
    def _sample_short(self, encoder_outputs, inputs):
        current_states, neighbor_current_mask, ego_neighbor_encoding, route_lanes, B, P = \
            self._prepare_common_inputs(encoder_outputs, inputs)

        short_len = self._short_future_len
        xT = torch.cat(
            [
                current_states[:, :, None],
                torch.randn(B, P, short_len, 4, device=current_states.device) * 0.5,
            ],
            dim=2,
        ).reshape(B, P, -1)

        def initial_state_constraint_short(xt, t, step):
            xt = xt.reshape(B, P, short_len + 1, 4)
            xt[:, :, 0, :] = current_states
            return xt.reshape(B, P, -1)

        x0_short = dpm_sampler(
            self.dit_short,
            xT,
            other_model_params={
                "cross_c": ego_neighbor_encoding,
                "route_lanes": route_lanes,
                "neighbor_current_mask": neighbor_current_mask,
            },
            dpm_solver_params={"correcting_xt_fn": initial_state_constraint_short},
            model_wrapper_params={
                "classifier_fn": self._guidance_fn,
                "classifier_kwargs": {
                    "model": self.dit_short,
                    "model_condition": {
                        "cross_c": ego_neighbor_encoding,
                        "route_lanes": route_lanes,
                        "neighbor_current_mask": neighbor_current_mask,
                    },
                    "inputs": inputs,
                    "observation_normalizer": self._observation_normalizer,
                    "state_normalizer": self._state_normalizer,
                },
                "guidance_scale": 0.5,
                "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
            },
        )

        return x0_short.reshape(B, P, short_len + 1, 4)

    # -------------------------
    # Sampling: full with pivot (re-noise + overwrite short segment)
    # -------------------------
    def _sample_full_with_pivot(self, encoder_outputs, inputs, pivot, num_branches):
        current_states, neighbor_current_mask, ego_neighbor_encoding, route_lanes, B, P = \
            self._prepare_common_inputs(encoder_outputs, inputs)

        short_len = self._short_future_len
        T = self._future_len

        assert pivot.shape[0] == B and pivot.shape[1] == P and pivot.shape[2] == short_len + 1
        pivot_clean = pivot

        def _to_time_vector(t, device, dtype):
            if not torch.is_tensor(t):
                t = torch.tensor(t, device=device, dtype=dtype)
            if t.dim() == 0:
                t = t.expand(B)
            elif t.dim() == 1 and t.shape[0] != B:
                if t.shape[0] == 1:
                    t = t.expand(B)
                else:
                    raise ValueError(f"Unexpected t shape: {t.shape}, expected scalar or [B].")
            return t

        def _renoise_x0_to_xt(x0, t_vec):
            device, dtype = x0.device, x0.dtype
            t_vec = _to_time_vector(t_vec, device=device, dtype=dtype)

            if hasattr(self._sde, "perturb_data"):
                noise = torch.randn_like(x0)
                return self._sde.perturb_data(x0, t_vec, noise)

            if hasattr(self._sde, "marginal_prob"):
                mean, std = self._sde.marginal_prob(x0, t_vec)
                if torch.is_tensor(std):
                    while std.dim() < mean.dim():
                        std = std.unsqueeze(-1)
                else:
                    std = torch.tensor(std, device=device, dtype=dtype).view(
                        B, *([1] * (mean.dim() - 1))
                    )
                eps = torch.randn_like(mean)
                return mean + std * eps

            raise AttributeError(
                "VPSDE_linear must expose `perturb_data` or `marginal_prob` for re-noise."
            )

        preds = []
        for _ in range(num_branches):
            xT = torch.randn(B, P, T + 1, 4, device=current_states.device) * 0.5
            xT = xT.reshape(B, P, -1)

            def correcting_fn(xt, t, step, _pivot_clean=pivot_clean):
                xt = xt.reshape(B, P, T + 1, 4)
                pivot_x0 = _pivot_clean.reshape(B, P, -1)
                pivot_xt = _renoise_x0_to_xt(pivot_x0, t).reshape(B, P, short_len + 1, 4)
                xt[:, :, : short_len + 1, :] = pivot_xt
                return xt.reshape(B, P, -1)

            x0 = dpm_sampler(
                self.dit_full,
                xT,
                other_model_params={
                    "cross_c": ego_neighbor_encoding,
                    "route_lanes": route_lanes,
                    "neighbor_current_mask": neighbor_current_mask,
                },
                dpm_solver_params={"correcting_xt_fn": correcting_fn},
                model_wrapper_params={
                    "classifier_fn": self._guidance_fn,
                    "classifier_kwargs": {
                        "model": self.dit_full,
                        "model_condition": {
                            "cross_c": ego_neighbor_encoding,
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask,
                        },
                        "inputs": inputs,
                        "observation_normalizer": self._observation_normalizer,
                        "state_normalizer": self._state_normalizer,
                    },
                    "guidance_scale": 0.5,
                    "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
                },
            )

            x0 = self._state_normalizer.inverse(x0.reshape(B, P, T + 1, 4))[:, :, 1:]
            preds.append(x0)

        prediction = torch.stack(preds, dim=1)
        pivot_world = self._state_normalizer.inverse(pivot_clean)[:, :, 1:]
        return prediction, pivot_world

    def forward(self, encoder_outputs, inputs, num_branches=1):
        if self.training:
            return self._forward_train(encoder_outputs, inputs)
        else:
            pivot = self._sample_short(encoder_outputs, inputs)
            prediction, pivot_world = self._sample_full_with_pivot(
                encoder_outputs, inputs, pivot, num_branches=num_branches
            )
            return {"prediction": prediction, "pivot": pivot_world}
