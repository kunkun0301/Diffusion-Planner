from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    futures: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    norm: StateNormalizer,
    loss: Dict[str, Any],
    model_type: str,
    short_future_len: int,          # short horizon length (future steps, excluding current)
    w_full: float = 1.0,
    w_short: float = 1.0,
    eps: float = 1e-3,
):
    """
    Scheme-1 compatible loss:
      - Build TWO noised inputs:
          * sampled_trajectories_short: standard diffusion corruption on short horizon [0:short_future_len)
          * sampled_trajectories_full : long-term corruption mask (keep short segment clean anchor, noise tail)
      - Call model ONCE.
      - Compute:
          * short loss on short frames
          * full loss on tail/branch frames only (t >= short_future_len)
      - Return combined losses in `loss` for backward-compat logging.

    Expected shapes (same as your original):
      ego_future:            [B, T, 4]
      neighbors_future:      [B, Pn, T, 4]
      neighbor_future_mask:  [B, Pn, T]  (True = masked/invalid)
    """

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, T]

    B, Pn, T, _ = neighbors_future.shape
    if not (0 <= short_future_len <= T):
        raise ValueError(f"short_future_len must be in [0, T], got {short_future_len}, T={T}")

    # ---- Current states & masks ----
    ego_current = inputs["ego_current_state"][:, :4]                           # [B,4]
    neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]         # [B,Pn,4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0  # [B,Pn]
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)  # [B,Pn,1+T]

    # ---- GT future stack ----
    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future], dim=1)   # [B,P,T,4], P=1+Pn
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B,P,4]
    P = gt_future.shape[1]

    # ---- diffusion time & noise ----
    t = torch.rand(B, device=gt_future.device) * (1.0 - eps) + eps          # [B]
    z_full = torch.randn_like(gt_future, device=gt_future.device)           # [B,P,T,4]

    # ---- normalized x0 (current + future) ----
    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)   # [B,P,1+T,4]
    # keep your original neighbor invaliding behavior
    all_gt[:, 1:][neighbor_mask] = 0.0

    x0_fut = all_gt[:, :, 1:, :]  # [B,P,T,4]

    # ---- forward marginal q(x_t | x0) ----
    mean, std = marginal_prob(x0_fut, t)
    # std broadcast to [B,1,1,1]
    std = std.view(-1, *([1] * (len(x0_fut.shape) - 1)))

    # ============================================================
    # Build xT_short: standard diffusion corruption on SHORT frames
    # ============================================================
    # short future frames are indices [0, short_future_len) within x0_fut
    if short_future_len > 0:
        xT_short_fut = mean[:, :, :short_future_len, :] + std * z_full[:, :, :short_future_len, :]
        xT_short = torch.cat([all_gt[:, :, :1, :], xT_short_fut], dim=2)  # [B,P,1+short,4]
    else:
        # only current frame
        xT_short = all_gt[:, :, :1, :]  # [B,P,1,4]

    # ============================================================
    # Build xT_full: long-term corruption mask
    #   - keep short segment CLEAN anchor
    #   - noise only tail frames [short_future_len, T)
    # ============================================================
    xT_full_fut = x0_fut.clone()
    if short_future_len < T:
        xT_full_fut[:, :, short_future_len:, :] = mean[:, :, short_future_len:, :] + std * z_full[:, :, short_future_len:, :]
    xT_full = torch.cat([all_gt[:, :, :1, :], xT_full_fut], dim=2)  # [B,P,1+T,4]

    # ---- merge inputs (keep legacy key too, but provide full/short separately) ----
    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT_full,                # legacy/default: full
        "sampled_trajectories_full": xT_full,           # for dit_full path
        "sampled_trajectories_short": xT_short,         # for dit_short path
        "diffusion_time": t,
    }

    # IMPORTANT: scheme-1 assumes your model training forward uses the two keys above.
    # i.e., dit_full reads sampled_trajectories_full; dit_short reads sampled_trajectories_short.
    _, decoder_output = model(merged_inputs)

    # ---- collect predictions ----
    score_full = decoder_output["score"][:, :, 1:, :]  # [B,P,T,4]
    score_short = decoder_output["score_short"]        # expected [B,P,1+short,4]
    score_short_fut = score_short[:, :, 1:, :]         # [B,P,short,4]

    # ---- build targets for x_start (if used) ----
    x0_short_fut = x0_fut[:, :, :short_future_len, :]  # [B,P,short,4]

    # ---- per-frame diffusion losses ----
    if model_type == "score":
        # full horizon loss tensor [B,P,T]
        dpm_loss_full = torch.sum((score_full * std + z_full) ** 2, dim=-1)

        # short loss tensor [B,P,short]
        if short_future_len > 0:
            z_short = z_full[:, :, :short_future_len, :]
            dpm_loss_short = torch.sum((score_short_fut * std + z_short) ** 2, dim=-1)
        else:
            dpm_loss_short = score_full.new_zeros((B, P, 0))

    elif model_type == "x_start":
        dpm_loss_full = torch.sum((score_full - x0_fut) ** 2, dim=-1)

        if short_future_len > 0:
            dpm_loss_short = torch.sum((score_short_fut - x0_short_fut) ** 2, dim=-1)
        else:
            dpm_loss_short = score_full.new_zeros((B, P, 0))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # ---- time masks ----
    # full tail/branch mask: indices [short_future_len, T)
    tail_mask = dpm_loss_full.new_zeros((1, 1, T))
    if short_future_len < T:
        tail_mask[:, :, short_future_len:] = 1.0

    # short mask: indices [0, short_future_len)
    short_mask = dpm_loss_full.new_zeros((1, 1, T))
    if short_future_len > 0:
        short_mask[:, :, :short_future_len] = 1.0

    # ---- neighbor losses (respect validity masks) ----
    # Full (tail only)
    if short_future_len < T:
        neighbors_tail_valid = neighbors_future_valid & tail_mask.expand(B, Pn, T).bool()
        neighbor_full_vals = dpm_loss_full[:, 1:, :][neighbors_tail_valid]
        neighbor_pred_loss_full = neighbor_full_vals.mean() if neighbor_full_vals.numel() > 0 else torch.tensor(0.0, device=gt_future.device)
    else:
        neighbor_pred_loss_full = torch.tensor(0.0, device=gt_future.device)

    # Short (short frames only) — only meaningful if short_future_len>0
    if short_future_len > 0:
        neighbors_short_valid = neighbors_future_valid[:, :, :short_future_len]
        neighbor_short_vals = dpm_loss_short[:, 1:, :][neighbors_short_valid]
        neighbor_pred_loss_short = neighbor_short_vals.mean() if neighbor_short_vals.numel() > 0 else torch.tensor(0.0, device=gt_future.device)
    else:
        neighbor_pred_loss_short = torch.tensor(0.0, device=gt_future.device)

    # ---- ego losses ----
    # Full: tail only
    if short_future_len < T:
        ego_full_vals = dpm_loss_full[:, 0, :] * tail_mask.squeeze(0).squeeze(0)  # [B,T]
        denom = tail_mask.sum().clamp_min(1.0)
        ego_loss_full = ego_full_vals.sum() / (B * denom)
    else:
        ego_loss_full = torch.tensor(0.0, device=gt_future.device)

    # Short: all short frames (mean over available frames)
    if short_future_len > 0:
        ego_loss_short = dpm_loss_short[:, 0, :].mean()
    else:
        ego_loss_short = torch.tensor(0.0, device=gt_future.device)

    # ---- combined (backward-compatible keys) ----
    loss["neighbor_prediction_loss_full"] = neighbor_pred_loss_full
    loss["neighbor_prediction_loss_short"] = neighbor_pred_loss_short
    loss["ego_planning_loss_full"] = ego_loss_full
    loss["ego_planning_loss_short"] = ego_loss_short

    loss["neighbor_prediction_loss"] = w_full * neighbor_pred_loss_full + w_short * neighbor_pred_loss_short
    loss["ego_planning_loss"] = w_full * ego_loss_full + w_short * ego_loss_short
    loss["total_loss"] = loss["neighbor_prediction_loss"] + loss["ego_planning_loss"]

    # safety
    assert not torch.isnan(dpm_loss_full).any(), f"loss cannot be nan, z_full={z_full}"

    return loss, decoder_output
