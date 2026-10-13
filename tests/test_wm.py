"""World model: shapes, open-loop imagination, and the evaluation metrics on known cases."""
import numpy as np
import torch

from p2p.wm.evaluate import auroc, psnr, ssim
from p2p.wm.model import ACT_DIM, IMG, Z_DIM, WorldModel, losses


def test_imagine_is_open_loop_and_losses_are_finite():
    torch.manual_seed(0)
    m = WorldModel()
    B, T = 2, 5
    frames, actions, bowl = torch.rand(B, T + 1, 3, IMG, IMG), torch.randn(B, T, ACT_DIM), torch.randn(B, T + 1, 3)
    z = m.imagine(m.encode(frames[:, 0]), actions)
    assert z.shape == (B, T, Z_DIM)
    # imagination only sees the first frame: changing later frames must not change it
    frames2 = frames.clone()
    frames2[:, 1:] = 0
    assert torch.allclose(z, m.imagine(m.encode(frames2[:, 0]), actions))
    L = losses(m, frames, actions, bowl)
    assert all(torch.isfinite(v) for v in L.values())
    L["total"].backward()


def test_metrics():
    x = torch.rand(3, 3, IMG, IMG)
    assert torch.allclose(ssim(x, x), torch.ones(3), atol=1e-4)
    assert (psnr(x, x) > 90).all()
    assert auroc(np.array([3.0, 2.0, 1.0, 0.0]), np.array([True, True, False, False])) == 1.0
    assert auroc(np.array([1.0, 1.0]), np.array([True, False])) == 0.5
