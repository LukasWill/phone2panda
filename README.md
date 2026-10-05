# phone2panda

Work in progress (submission for the Humanoid robot-learning intern challenge, due 9 Oct 2026).

Phone recordings (iPhone LiDAR) of me putting a bowl on a plate are turned into Panda robot behaviour in LIBERO
(task: LIBERO-Goal "put the bowl on the plate").

## Layout

```
p2p/human/     phone recordings -> metric 3D in a table frame (Stray Scanner loader, ArUco tags, hands)
p2p/sim/       LIBERO wrapper (LeRobot-compatible observations), pose controller, scripted pick-and-place
p2p/retarget/  human trajectory -> robot waypoints (absolute / object-relative / intent-only)
p2p/data/      sim-verified data generation, LeRobot dataset I/O
p2p/policy/    ACT training and evaluation
p2p/rl/        offline RL (IQL advantages for weighted imitation)
p2p/wm/        action-conditioned world model
notebooks/     Colab notebooks, run in order (00, 01, ...)
tools/         tag sheet generator, recording checker (runs on a Mac)
tests/         fast tests without the simulator: python -m pytest -q tests
```

## Running

- Recording check on a Mac: `pip install -r requirements-mac.txt`, then `python tools/check_recording.py <recording_dir>`
- Simulation: open `notebooks/00_sim_sanity.ipynb` in Colab (GPU runtime). The first cell installs `lerobot[libero]`.
