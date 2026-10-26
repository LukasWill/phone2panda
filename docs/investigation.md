# Why was the policy trained on my demos worse than one trained on teleoperation?

This is the full record of the investigation summarised in the README (Q2). Every test after notebook 06 had its
prediction, and the threshold that would confirm or reject the hypothesis, written into the notebook before it
ran; the verdicts below follow those thresholds. Success rates are on LIBERO's 50 fixed evaluation scenes,
pooled over two evaluations (n = 100 per training seed) unless stated.

## Starting point

Same policy (ACT), settings, simulator, cameras and evaluation scenes; only the demonstrations differ.
Teleoperated demos (49): 97% [92-99]. My demos, copied in full (438 generated episodes from 10 demos): 57% [47-66].
Most failures never lift the bowl; when the policy does grasp, it places as accurately as the teleop policy.

## Hypotheses, evidence and tests

| | Hypothesis | Evidence so far | Test |
|---|---|---|---|
| H1 | **Teacher ceiling.** A student trained by imitation rarely beats its teacher. | Our teacher succeeds in 85% of these scenes; the official demos come from a human teleoperator, presumably near 100%. This explains at most ~15 of the 40 points: our student keeps 57/85 = 67% of its teacher's success, the teleop student 97%. | - |
| H2 | **Inconsistent grasps across demos.** In similar-looking scenes my 10 demos grasp different points of the rim (86-149°, so the wrist angle differs by up to 60°) and approach along 10 different paths. The image does not tell which demo to follow, so the policy may average them, and an average of two good rim grasps can be a grasp that matches neither. | **Real, but not the main cause.** Wrist angle at the moment of grasping, across episodes: teleop spread 5.6° (10th-90th percentile -12° to 1°), ours 21.3° (-8° to 51°), four times wider (`results/act/demo_consistency.jsonl`). Averaging the policy's own successive predictions (temporal ensembling) drops ours to 12%, mostly missed grasps, but leaves teleop at 100%: its predictions agree with each other, ours switch between grasps. Missed grasps dominate the failures; 9x more data of the same kind barely helps.  Direct test (notebook 08, prediction fixed beforehand: >= 70% would confirm H2 as the main cause, <= 60% would reject it): ACT on the 5 demos with grasp sides within 10° reaches **59%** [49-68] (2 evals), the same as `ours_all`. Missed grasps halve (7.5 vs 16 per 50 episodes) and averaging hurts less (32% vs 12%), but placement errors rise (9 vs 1 on the plate > 3 cm off), so the total does not move. Notebook 13 settles it: with the approach planned, all 10 demos (inconsistent grasp sides) reach 98% [93-99]. | Done (`results/act/ours_consistent_s0/`, `results/act/ours_all_vertical_s0/`). |
| H3 | **The ±1 cm grasp perturbation** of data generation adds variety where the gripper closes. | Probably minor. Only perturbed rollouts that succeeded are kept, and the added spread (about 11° around the rim) is small next to the spread across my demos (about 20°): together about 23°. | Regenerate with `--grasp-sigma 0` and retrain. |
| H4 | **Too-clean demonstrations.** Scripted rollouts follow their plan almost perfectly, so the data never shows how to correct a small error. Teleoperators correct themselves all the time, which shows the policy how to recover. Once our policy is a few mm off, it is in states it has never seen. | **Not supported by this test.** Notebook 09 regenerated the 5 consistent demos on the same 650 scenes with and without DART noise (executed motion disturbed by a median 8 mm, labels = the teacher's corrections). With 450 episodes each: clean 67% [57-75], DART 64% [54-73] (prediction fixed beforehand: >= +15 points would confirm, <= +5 rejects). Caveat: with action chunks, a frame's 50 future labels include corrections of noise that has not happened yet, which the policy cannot predict; the DART run's training loss ended twice as high (0.24 vs 0.12), so this test is weaker than it looks (see H7). | Re-tested with longer training (notebook 10). |
| H5 | **Open-loop chunks during a precise grasp.** ACT predicts 50 actions and executes 25 of them (1.25 s) before looking again, so an error in the chunk's grasp position is not corrected before the fingers close. Our policy also needs about twice as many steps (median 136 vs 74 to succeed), hence more chunks. | **Rejected.** Executing 10 instead of 25 actions per prediction gives 50% (vs 57%), re-predicting every step 12%. Committing to one chunk is what keeps our policy on one grasp; looking more often only gives it more chances to switch. | - |
| H7 | **Under-training.** Every run trains 20k steps. That is ~140 passes over the 49 official demos but only ~12 over our 450 episodes. | **Rejected.** Notebook 10 trained the clean 450-episode set 3x longer. The training loss kept falling (L1 0.12 -> 0.07), but success did not rise; it fell: 20k steps 69% [59-77], 40k 61% [51-70], 60k 58% [48-67] (two evaluations each; the 20k checkpoint reproduces notebook 09's 67%). Fitting the demonstrations more closely makes the policy worse in closed loop. | Done (`results/act/ours_consistent_clean_long_s0/`). |
| H8 | **A teacher that needs precision the policy cannot see.** Our teacher plans from the simulator's exact object positions; if its strategy only works with millimetre accuracy, a policy that sees 128 px images cannot copy it. | **Rejected.** The same plans built from a wrong bowl position still succeed 94% of the time at 2 cm error and 90% at 3 cm; a wrong plate position is tolerated up to 1.5 cm (92%), 70% at 2 cm (`results/act/teacher_tolerance_consistent5.json`). The strategy is not fragile. | Done (`tools/teacher_tolerance.py`). |
| H9 | **The demonstrated approach.** The object-relative teacher reaches the rim along the last 15 cm of my hand's path, bent to end at the grasp and skimming just above the rim; when it closes differs between demos (median 54-72 steps; spread 24 steps over all episodes). The teleop policy comes down onto the rim and closes at an almost fixed moment (spread 2.5 steps). | **Supported by the behaviour data (notebook 11).** Where the gripper first closes, as distance from the bowl's axis (rim at 5.6 cm): teacher 5.1 cm; official policy 4.6 cm (10th-90th percentile 3.3-5.3) in successes; our consistent policy 5.5 cm in successes but 7.1 cm (5.5-10.9) in failures, after which it closes again 3-7 times without lifting. Grasp height is right in both (about 2 cm below the rim). So our policy re-tries but misplaces the grasp horizontally (`results/behaviour/`). The same plans with a straight-down approach: teacher 99%, closes at 4.4 cm, spread 1.2 steps (`results/behaviour/teacher_variants.json`). | **Confirmed (notebook 12).** Same 5 demos, same 650 scenes and perturbations, same 450-episode cut, same training; only the approach changed. My approach: 67% [57-75] (69% in a repeat). Approach planned straight down, everything else from my demo: **99% [95-100]** (98% and 100% in two evaluations); prediction fixed beforehand: >= 83% confirms. Fully scripted motion: 100% [96-100]. The policy now closes 4.3-4.4 cm from the bowl's axis, like the teacher, and missed grasps disappear (0 of 100 episodes never lift the bowl). Before that, notebook 11 had located the failure: our policy closed 7.1 cm from the axis in failures (rim at 5.6 cm), the official one 4.6 cm. | Done (`results/act/ours_consistent_vertical_s0/`, `..._intent_s0/`). |

H6 (training and evaluation images differ) was ruled out by construction: generated data and evaluation use
the same renderer, camera and resizing.

## How the policy executes its predictions (notebook 06)

- **How the policy executes its predictions matters for our data, not for teleop data** (same checkpoints,
  only the execution changed; `results/act/*/eval_n10.json`, `eval_te.json`):

  | Execution | official | ours_all | ours_consistent |
  |---|---|---|---|
  | 25 of 50 predicted actions, then predict again (default) | 97% (n=100) | 57% (n=100) | 59% (n=100) |
  | 10 of 50, then predict again | 96% | 50% | - |
  | predict every step, average all predictions for this step (temporal ensembling) | 100% | **12%** (33 of 44 failures never lift the bowl) | 32% |

  Looking more often does not help our policy, and averaging its predictions destroys it. With the approach
  planned (`ours_consistent_vertical`), averaging is harmless again (92%; default 99%). See H2, H5 and H9 below.

## Order of the experiments

| Notebook | Question | Fixed prediction | Outcome |
|---|---|---|---|
| 06 | Does executing fewer actions per chunk, or averaging predictions, change success? Are demo grasps consistent? | (diagnostic) | Ours: 50% with 10-step chunks, 12% averaged; teleop unaffected. Wrist-angle spread at the grasp 21° vs 5.6° |
| 08 | H2: do 5 demos with similar grasp sides close the gap? | >= 70% confirms, <= 60% rejects | 59%: rejected as the main cause |
| 09 | H4: does DART noise during generation help? Does more data help? | DART >= clean + 15 confirms, <= +5 rejects | Clean 67%, DART 64%: not supported |
| 10 | H7: are the policies under-trained? | 60k >= 20k + 10 confirms, within 5 rejects | 69 -> 61 -> 58%: rejected |
| 11 | Where and how do the policies fail? | (diagnostic) | Failed grasps close 7-8 cm from the bowl's axis (rim at 5.6 cm) and retry 3-7 times |
| (local) | H8: does the teacher need millimetre precision? | (physics only) | 90% at 3 cm bowl error: rejected |
| 12 | H9: does planning the approach close the gap? | >= 83% confirms, <= 72% rejects | 99%: confirmed; fully scripted 100% |
| 13 | Do all 10 demos (inconsistent grasp sides) also reach that level? | >= 90%: no extra cost | 98% |
| 14 | Does 67 -> 99 survive a second training seed? | planned >= 90%, copied <= 75% | 96% vs 66% |
| 15 | Is it the path's shape or how it is executed? | >= 90%: execution; <= 75%: shape | 82%: both contribute |

## Earlier findings, reread after the answer

- **Earlier findings, reread.** The collapse under averaging (H2's main evidence) belonged to the approach:
  with the planned approach, averaging costs 7 points, against 27-45 before. Consistent grasp sides (H2) and
  more data probably helped a little because they made the approach less varied; DART (H4) and longer training
  (H7) plausibly could not help because they did not change what the policy had to imitate. These are
  readings after the fact, not tests.
