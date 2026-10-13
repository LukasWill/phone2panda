# Table 1: why rollouts fail (checked by replaying single scenes, physics only)

**Absolute (0/260 main).** One global translation maps the human's average bowl start onto the sim's. It cannot work because the human scenes were laid out differently:
- The human's bowl start lies a median of 11 cm (range 2–37 cm) from the average of all demos.
- The human's bowl→plate vector points in every direction, 26–39 cm long. In the sim the bowl is 16 cm from the plate.

So the gripper closes where the bowl is not. The bowl starts about 16 cm from the plate and ends a median of 14 cm from it (5–95%: 10–18 cm): it is at most nudged, never carried. This is the negative control: replaying the hand's path without anchoring it to the objects transfers nothing.

**Object-relative, the two weak main demos.**
- `a4558c8abd_lefthand` (1/20). The human carried the bowl in an S-curve that swings about 9 cm to the robot's right and 20 cm up. Replayed faithfully, that path takes the held bowl into the cabinet that stands on the robot's right in this LIBERO scene, an obstacle the human scene did not have. From then on the arm is commanded +x and −z but does not move: the end-effector stays at z = 1.065 for about 100 steps. The release then happens about 11 cm above the table and the bowl lands 6–7 cm from the plate centre.
- `c521c4e934` (4/20). The grasp side is at 149° in the robot frame, the edge of the reachable range (0–150°). The arm stalls in the human's approach path: about 230 of 300 steps in "approach", and the episode times out.

Intent retargeting keeps only the grasp side and the place point, plans its own path, and succeeds on both demos (20/20). This is the trade-off in the table: the more of the human's motion we keep, the more of the human scene's assumptions we inherit.

**Natural-grasp set.** People naturally grasp the bowl on the side facing the robot's right: 177–200° in the robot frame, and 272° for `bd9e8d821d` and `dcbcbf700f`. Both retargeting strategies keep the grasp side, and the robot cannot lower onto that side in this scene. Success falls with the angle: 9–20/20 at about 180°, 4–14/20 at about 200°, and 0/20 at 272°. That is why the main dataset asks for a left-rim grasp.
