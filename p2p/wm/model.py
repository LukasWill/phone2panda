"""A small action-conditioned world model of the agentview camera.

    image x_t (64x64) --encoder--> z_t (256-D) --decoder--> x_hat_t
    h_{t+1} = GRU(h_t, [z_t, a_t]),   z_hat_{t+1} = z_t + MLP(h_{t+1})          (latent dynamics)
    bowl head on any latent: where the bowl is (3-D, metres, world frame)

One world-model step = 2 control steps (10 Hz), so a_t is the two 7-D robot actions concatenated (14-D).

Imagination ("dreaming"): encode ONE real frame, then feed only actions: every next latent comes from
the model's own previous prediction, never from a real image again. Errors therefore compound with
the horizon, which is exactly what the evaluation measures.

Why a bowl head and no "success" head: success = "the bowl ends on the plate", so we read it off the
imagined bowl position at the last step (distance to the plate). A head trained on "does this episode
succeed" would be trained only on episode ends, and successful episodes END at the success step while
failures run to the 300-step limit, so such a head could learn the episode length instead of the
physics. The bowl head is trained on every frame, successes and failures alike.

Design choices, kept deliberately simple (a few hours of free GPU):
  * deterministic, not stochastic (RSSM/Dreamer). Given the actions our scenes are deterministic,
    and a deterministic model is easier to train and to evaluate. The price: where the model is
    unsure (did the grasp hold?) it predicts the average, which shows up as blur / a bowl "in between".
  * residual latent update (z + delta): most of the image (table, background) does not change.
  * training rolls the model out open-loop for the whole window, so it is trained in the same regime
    it is evaluated in (no teacher-forcing gap).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 64
ACT_DIM = 14          # two consecutive 7-D LIBERO actions
Z_DIM = 256
H_DIM = 512


def _conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), nn.GroupNorm(8, cout), nn.SiLU())


def _deconv(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.GroupNorm(8, cout), nn.SiLU())


class WorldModel(nn.Module):
    def __init__(self, z_dim: int = Z_DIM, h_dim: int = H_DIM, act_dim: int = ACT_DIM):
        super().__init__()
        self.enc = nn.Sequential(_conv(3, 32), _conv(32, 64), _conv(64, 128), _conv(128, 256),  # 64 -> 4
                                 nn.Flatten(), nn.Linear(256 * 4 * 4, z_dim),
                                 nn.LayerNorm(z_dim, elementwise_affine=False))   # fixes the latent scale
        self.dec_in = nn.Linear(z_dim, 256 * 4 * 4)
        self.dec = nn.Sequential(_deconv(256, 128), _deconv(128, 64), _deconv(64, 32),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1))
        self.act_in = nn.Sequential(nn.Linear(act_dim, 128), nn.SiLU())
        self.h0 = nn.Linear(z_dim, h_dim)                      # initial memory from the first latent
        self.gru = nn.GRUCell(z_dim + 128, h_dim)
        self.delta = nn.Sequential(nn.Linear(h_dim, 512), nn.SiLU(), nn.Linear(512, z_dim))
        self.bowl_head = nn.Sequential(nn.Linear(z_dim, 256), nn.SiLU(), nn.Linear(256, 3))
        # bowl positions are regressed in normalised units; the statistics are saved with the weights
        self.register_buffer("bowl_mean", torch.zeros(3))
        self.register_buffer("bowl_std", torch.ones(3))

    # ---- per-frame ---------------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:          # (B, 3, 64, 64) in [0, 1]
        return self.enc(x * 2 - 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:          # -> (B, 3, 64, 64) in [0, 1]
        return torch.sigmoid(self.dec(self.dec_in(z).view(-1, 256, 4, 4)))

    def bowl(self, z: torch.Tensor) -> torch.Tensor:            # -> metres, world frame
        return self.bowl_head(z) * self.bowl_std + self.bowl_mean

    # ---- dynamics ------------------------------------------------------------------------------
    def imagine(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z0 (B, Z), actions (B, T, A) -> predicted latents (B, T, Z) for steps 1..T, open loop."""
        h = torch.tanh(self.h0(z0))
        z, out = z0, []
        for t in range(actions.shape[1]):
            h = self.gru(torch.cat([z, self.act_in(actions[:, t])], -1), h)
            z = z + self.delta(h)
            out.append(z)
        return torch.stack(out, 1)


def losses(model: WorldModel, frames: torch.Tensor, actions: torch.Tensor, bowl: torch.Tensor,
           w_latent: float = 0.1, w_bowl: float = 0.1) -> dict:
    """frames (B, T+1, 3, 64, 64) in [0, 1], actions (B, T, A), bowl (B, T+1, 3) in metres.

    * reconstruction: every real frame through encoder -> decoder (keeps the latent space visual)
    * prediction: decode the IMAGINED latents and compare with the real future frames, plus pull the
      imagined latents towards the encoder's latents of those frames (target detached, so the encoder
      is shaped by reconstruction only and cannot "cheat" by collapsing)
    * bowl: the head must read the bowl position from real latents AND from imagined ones
    Weights: image MSE is per pixel in [0, 1] (~0.003 once trained); the latent (LayerNorm'd, ~unit
    scale) and normalised bowl terms are larger numbers, so they get 0.1 to keep the image terms in charge.
    """
    B, T1 = frames.shape[:2]
    z_real = model.encode(frames.flatten(0, 1)).view(B, T1, -1)
    rec = F.mse_loss(model.decode(z_real.flatten(0, 1)), frames.flatten(0, 1))
    z_pred = model.imagine(z_real[:, 0], actions)                                   # (B, T, Z)
    pred_img = F.mse_loss(model.decode(z_pred.flatten(0, 1)), frames[:, 1:].flatten(0, 1))
    lat = F.mse_loss(z_pred, z_real[:, 1:].detach())
    target = (torch.cat([bowl, bowl[:, 1:]], 1) - model.bowl_mean) / model.bowl_std
    bowl_l = F.mse_loss(model.bowl_head(torch.cat([z_real, z_pred], 1)), target)
    total = rec + pred_img + w_latent * lat + w_bowl * bowl_l
    return {"total": total, "rec": rec.detach(), "pred": pred_img.detach(), "latent": lat.detach(),
            "bowl": bowl_l.detach()}
