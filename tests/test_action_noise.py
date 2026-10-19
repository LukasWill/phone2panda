import numpy as np

from p2p.data.generate import ActionNoise, noise_rng


def test_noise_spares_gripper_and_is_reproducible():
    a = np.array([0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0])
    n1, n2 = ActionNoise(0.05, 5, noise_rng(3)), ActionNoise(0.05, 5, noise_rng(3))
    out1 = [n1(a) for _ in range(50)]
    out2 = [n2(a) for _ in range(50)]
    assert np.allclose(out1, out2)                       # same seed, same disturbance
    assert all(o[6] == 1.0 for o in out1)                # gripper command never disturbed
    assert not np.allclose(out1[0][:6], a[:6])


def test_noise_off_is_identity_and_correlated_when_on():
    a = np.zeros(7)
    assert np.array_equal(ActionNoise(0.0, 5, noise_rng(0))(a), a)
    nz = ActionNoise(0.05, 5, noise_rng(0))
    x = np.array([nz(a)[0] for _ in range(4000)])
    assert 0.03 < x.std() < 0.07                          # stationary std ~ sigma
    assert np.corrcoef(x[:-1], x[1:])[0, 1] > 0.7         # drifts (exp(-1/5) = 0.82), does not jitter
