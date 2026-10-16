# phone2panda

<!-- DRAFT skeleton: tables and numbers are final (from results/); prose is a draft for you to rewrite in your
own voice. Lines marked TODO(you) need your input. -->

**From iPhone videos of my own hand to a Panda robot in simulation.** I recorded myself putting a bowl on a
plate (iPhone with LiDAR, Stray Scanner app). The pipeline turns each recording into a metric 3D
demonstration, retargets it to a Franka Panda in LIBERO (task: LIBERO-Goal *"put the bowl on the plate"*),
and replays it in many fresh simulated scenes, so that the **simulator verifies every robot demonstration**.
On that data I train an ACT policy, add offline RL that learns from the failed rollouts, and train an
action-conditioned **world model** that predicts the camera view and whether a plan will succeed.

| Question | Result (95% Wilson intervals; LIBERO's 50 fixed evaluation scenes, never used for data) |
|---|---|
| Q1: How should a human demo be retargeted? | Intent-only 100%, object-relative 81%, absolute replay 0% (13 demos x 20 scenes each) |
| Q2: How good is a policy trained only on my phone demos? | ACT 57% [47-66] (438 sim-verified episodes from 10 demos) vs 97% [92-99] on 49 official teleop demos |
| Q2b: Does learning from failures help? | All 600 rollouts weighted by an IQL critic: 66% [52-78]; the same data unweighted: 54% [40-67]; successes only: 57% [47-66]. Weighting helps by ~12 points; one seed each, so suggestive, not significant |
| Q3: Can a world model judge a plan before execution? | Success AUROC 0.93 imagining whole held-out episodes from their first frame (within one strategy); ranks the 3 strategies correctly |

![world model: real (left) vs imagined from the first frame (right)](assets/wm_hero_success.mp4)
<!-- TODO(you): GitHub shows mp4 only via drag-and-drop upload into the README editor, or link the file instead. -->

## Pipeline

```
iPhone (RGB 1920x1440 + LiDAR + ARKit pose)             p2p/human/
  ArUco tags on table / bowl / plate -> metric table frame; ARKit + tag-anchored camera track
  MediaPipe hand -> pinch point, grasp point on the rim (camera ray ∩ rim plane), events: grasp/lift/place/release
        |
        v  13 demos that pass every check (data/splits.json): 10 for training, 3 held out
retargeting to the robot                                   p2p/retarget/
  absolute (hand path, one global offset) | object-relative (path anchored to bowl/plate) | intent (grasp side + place point)
        |
        v
sim-verified data generation (MimicGen-style)              p2p/data/
  60 fresh scenes per demo (objects shifted +-3 cm, grasp/place perturbed); LIBERO decides success
  600 rollouts -> 438 successes (ours_all), 50 balanced (ours_50); all 600 kept (ours_mix)
        |
        +--> ACT (LeRobot)                                 p2p/policy/   notebook 03
        +--> IQL critic -> advantage weights -> AW-ACT     p2p/rl/       notebooks 05, 03
        +--> world model (image + bowl, 10 Hz)             p2p/wm/       notebook 04
```

## How to run

Everything that touches the simulator runs on Linux with a GPU (Colab T4 used throughout). Data and
checkpoints move between notebooks through a private Hugging Face dataset repo (`HF_TOKEN` in Colab secrets).

| Step | Where | Command / notebook | Time (T4) |
|---|---|---|---|
| Check a recording | Mac | `pip install -r requirements-mac.txt`; `python tools/check_recording.py <folder>` | seconds |
| Extract demos | Mac | `pip install mediapipe==0.10.14`; `python -m p2p.human.extract <raw_dir> data/processed` | minutes |
| Simulator sanity check | Colab | `notebooks/00_sim_sanity.ipynb` | ~10 min |
| Table 1 (retargeting) | any CPU | `python -m p2p.retarget.replay data/processed --inits 20` then `python tools/table1_summary.py` | ~40 min (2 cores) |
| Generate data | Colab | `notebooks/02_generate_data.ipynb` | ~65 min + ~30 min datasets |
| Train / evaluate ACT | Colab | `notebooks/03_train_eval_act.ipynb` (`DATASET` = official, ours_50, ours_all, ours_mix) | ~1.5 h per run |
| World model | Colab | `notebooks/04_world_model.ipynb` | ~1.5 h |
| Offline-RL weights | Colab CPU | `notebooks/05_offline_rl.ipynb` | ~25 min |
| Tests (no simulator) | anywhere | `python -m pytest -q tests` | seconds |

Rendering must run on the GPU: Colab lacks the NVIDIA EGL vendor file, so MuJoCo silently renders on the
CPU (~2.6 steps/s instead of ~29). `p2p.sim.env.ensure_nvidia_egl()` fixes this and notebook 02 checks it.

## Data: my recordings

- 25 recordings. **13 pass every check** and are the main set (10 training, 3 held out, chosen before any
  training); 6 more with my *natural* grasp are used only in Table 1; 6 were excluded because the grasp
  point could not be measured (hand hidden at the grasp).
- Protocol (kept minimal): tags visible in the first second; grasp the bowl's left rim; hand and bowl visible
  from grasp to release; bowl on the plate visible at the end. The phone is hand-held: the camera track is
  ARKit, anchored to the table tag (camera spread 3-11 mm).
- TODO(you): one sentence on your setup (bowl 100 mm, plate 198 mm, tags printed at 96%) and a frame from a
  recording with the tags and hand overlay.

## Q1. Retargeting: what transfers from a human demo?

Each of the 13 main demos replayed in 20 of LIBERO's evaluation scenes (`results/table1.md`):

| Strategy | Keeps from my demo | Success | Demos >= 50% |
|---|---|---|---|
| absolute | the whole hand path, one global offset | 0% [0-2] | 0/13 |
| object-relative | grasp point, approach, lift, carry path, place point, all relative to bowl/plate | 81% [76-85] | 11/13 |
| intent | only the grasp side and the place point | 100% [99-100] | 13/13 |

- **Absolute is the negative control.** My scenes were laid out differently (bowl start varies by a median of
  11 cm; bowl-to-plate vectors point every way), so the gripper closes where the bowl is not.
- **Object-relative inherits my scene's assumptions** (`results/table1_notes.md`): one demo's S-shaped carry
  swings the bowl into a cabinet that only the robot's scene has; another grasps at the edge of the
  reachable side.
- **Natural grasps fail for the robot:** people grab the bowl on the side facing the robot's right (177-272°
  in the robot frame), where the Panda cannot lower; success falls to 0/20 at 272°. Hence the left-rim protocol.
- **Design choice:** data generation uses *object-relative*, not intent: intent keeps two numbers from the
  video, so its data would be scripted motion in disguise. The simulator's success filter absorbs the lower yield.

## Q2. Imitation from sim-verified phone data (ACT)

Generation: 600 rollouts from 10 demos, 73% success (my per-demo prediction: 75%). Failures are mostly
near-misses: 50% end on the plate but more than 3 cm from its centre, which LIBERO's `On` check rejects
(`check_ontop` requires the centres within 3 cm).

ACT (LeRobot, 20k steps, batch 32, same settings for every run), evaluated on the 50 fixed scenes:

| Training data | Episodes | Success | Bowl never lifted | Lifted, off the plate | On the plate, > 3 cm off |
|---|---|---|---|---|---|
| official teleop (`lerobot/libero`) | 49 | **97%** [92-99] (2 evals, n=100) | 0 | 0 | 0 |
| ours_50 (count-matched) | 50 | **46%** [33-60] | 13 | 7 | 7 |
| ours_all | 438 | **57%** [47-66] (2 evals, n=100) | 16 | 4 | 1 |
| ours_mix + IQL advantage weights | 600 | **66%** [52-78] | - | - | - |
| ours_mix, unweighted (control) | 600 | **54%** [40-67] | - | - | - |

Failure columns are from one evaluation of 50 episodes each (not recorded for the two ours_mix runs).

- **The gap to teleop is in grasping, not placing.** Successful episodes place the bowl as accurately as the
  teleop policy (median 1.9 vs 1.1 cm from the plate centre); most failures never lift the bowl.
  My reading: our grasp is a millimetre-precise rim pinch whose target varies across demos (10 human
  approach styles, deliberate +-1 cm perturbation), and an imitation policy averages it; teleop grasps are
  consistent. Our episodes are also slower (median 128 vs 74 steps), so errors have more steps to compound.
- **9x more generated data: 46% -> 57%**, not significant. It seems to sharpen placement (7 vs 1 off-centre
  failures) but not grasping.
- **Teacher ceiling:** the planner that generated our data scores ~82% in these scenes; the student reaches 57%.
- **No recovery behaviour:** after a missed grasp the policy hovers over the plate. The success-filtered
  data contains no recoveries.

### Learning from failures (offline RL, gated)

ACT on successes only throws away 162 failed rollouts. An IQL critic (expectile 0.7, gamma 0.99, reward 1 at
success) trained on the simulator's 18-D state scores every frame; ACT is retrained on all 600 episodes with
loss weights `exp(beta * A)` on standardised advantages (beta = 1, mean 1, capped at 20). The critic sees the
simulator state, the policy only images.

- Gate, fixed before the run: the critic must tell success from failure halfway through an episode.
  AUROC 0.95 (0.66 from the first frame) -> passed (`results/rl/iql_diagnostics.json`).
- Weights: success frames 1.15 on average, failure frames 0.82; 36% effective sample size.
- Result, three ways to use the same 600 rollouts:

  | Training data | Success |
  |---|---|
  | successes only (filtered imitation, ours_all) | 57% [47-66] (n=100) |
  | all rollouts, unweighted (control) | 54% [40-67] |
  | all rollouts, IQL advantage-weighted | **66%** [52-78] |

  Adding the failures without weights does not help (imitating failed grasps should hurt, and the
  successes still dominate); the critic's weights turn the same data into the best of the three runs.
  The ordering matches the hypothesis, but the intervals overlap and each configuration is one seed, so this
  is suggestive, not significant; a few more seeds would settle it.

## Q3. World model (Part B)

A deterministic action-conditioned model at 10 Hz: 64 px image -> 256-D latent; a GRU predicts the next
latent from the latent and two 7-D actions; a decoder reconstructs the image and a head reads the bowl
position. Trained open-loop (curriculum 0.8 -> 3.2 s) on 680 episodes including failures; tested on
**180 episodes of the 3 held-out demos**, each replayed with all three strategies in the same scenes.

| Seconds imagined ahead | 0.1 | 1 | 2 | 4 |
|---|---|---|---|---|
| PSNR (dB): model / copy start frame / autoencoder ceiling | 30.3 / 39.1 / 30.7 | 27.9 / 28.7 / 30.1 | 26.9 / 25.9 / 29.9 | 25.3 / 22.9 / 29.0 |
| Bowl error while it moves (cm): model / "stays put" | - | 5.4 / 5.6 | 6.3 / 7.7 | 6.0 / 11.3 |

**Success from imagination:** imagine the last H seconds from a real frame, score success by the imagined
bowl's distance to the plate (true bowl position gives AUROC 0.99):

| Imagined before the end | 0 | 0.5 s | 1 s | 2 s | 4 s | 8 s | whole episode |
|---|---|---|---|---|---|---|---|
| AUROC, all strategies (n=180) | 0.98 | 0.95 | 0.96 | 0.86 | 0.96 | 0.96 | 0.97 |
| AUROC, object-relative only (n=60) | 0.96 | 0.95 | 0.89 | 0.80 | 0.91 | 0.89 | **0.93** |

**Judging plans before execution** (from the first frame + actions):

| Strategy | True success | Predicted by the world model |
|---|---|---|
| intent | 97% | 80% |
| object-relative | 57% | 32% |
| absolute | 0% | 0% |

![](results/wm/wm_horizon.png) ![](results/wm/wm_auroc.png)

- It predicts motion: past ~1.3 s it beats copying the start frame, and the carried bowl's position
  (6 cm error vs 11 cm for "stays put").
- It ranks the strategies in the true order but is pessimistic in absolute terms; its bowl readout has a
  ~2 cm floor, close to the 3 cm success tolerance.
- Why no "success" head: successful episodes end at success, failures run to 300 steps, so such a head can
  learn the episode length. Success is read from the imagined bowl position instead.

## What worked, what didn't

Worked
- The measurement chain: ARKit + tag-anchored camera track, LiDAR-free grasp point (ray-plane), sim-verified
  generation that reproduces my per-demo yield predictions within noise.
- Object-anchored retargeting (0% -> 81%), and the simulator as a filter.
- A small world model that ranks plans correctly on held-out demos.

Didn't (or only partly)
- Policies from phone data fall well short of teleop (57% vs 97%); grasp acquisition is the bottleneck.
- My natural grasp side is unusable for this robot; the protocol had to constrain it.
- Faithful human paths can be wrong for the robot's scene (the cabinet collision).
- TODO(you): anything from the recording side (hand lost at the grasp, LiDAR vs PnP mismatch of 3-6% on the
  table tag, unresolved but bypassed: the key quantities are LiDAR-free).

## Reproducibility notes

- Evaluation is not bit-reproducible on a GPU: re-evaluating the same checkpoint flipped 17 of 50 episodes
  (28 -> 29 successes), so headline rates pool two evaluations where available.
- Soft resets keep MuJoCo's solver warm-start: a seed fixes the scene exactly, not the rollout.
- One training seed per configuration; differences below ~15 points are not resolvable with n = 50.

## Related work

MimicGen (Mandlekar et al., 2023) for generating many robot demos from few; ACT (Zhao et al., 2023);
LIBERO (Liu et al., 2023); IQL (Kostrikov et al., 2021) and advantage-weighted regression (Peng et al., 2019);
world models in the Dreamer line (Hafner et al.). TODO(you): add what you know best (e.g. human-video
retargeting work).

## Layout

```
p2p/human/     phone recordings -> metric 3D in a table frame (Stray Scanner loader, ArUco tags, hands)
p2p/sim/       LIBERO wrapper (LeRobot-compatible observations), pose controller, scripted pick-and-place
p2p/retarget/  human trajectory -> robot waypoints (absolute / object-relative / intent)
p2p/data/      sim-verified data generation, LeRobot dataset I/O
p2p/policy/    ACT training and evaluation (with per-sample weights)
p2p/rl/        IQL critic -> advantage weights
p2p/wm/        action-conditioned world model: model, data, training, evaluation
notebooks/     Colab notebooks 00-05
tools/         tag sheet, recording checker, Table 1 summary, failure analysis, yield estimate
results/       every number in this README
tests/         python -m pytest -q tests
```
