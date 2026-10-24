"""Teacher trajectories for assets/approach_shape_vs_execution.png (physics only, ~3 min).

    python tools/approach_figure_collect.py traj.pkl && python tools/approach_figure_plot.py traj.pkl assets/approach_shape_vs_execution.png
"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pathlib import Path
from p2p.sim.env import TaskEnv, BOWL_HEIGHT
from p2p.human.extract import load
from p2p.retarget.strategies import make_plan
from p2p.sim.scripted import WaypointFollower
DEMOS = "f9115215f9_lefthand,c5f525e833,9a63cb1538_lefthand,92e0b86bc6_lefthand,65ee6ec786".split(",")
root = Path(__file__).resolve().parents[1]
demos = [load(root / 'data/processed' / f'{n}.npz') for n in DEMOS]
env = TaskEnv(render=False, hard_reset=False)
out = []
for init in [0, 4, 8]:
    for d in demos:
        for strat in ['object_relative', 'object_relative_timed', 'object_relative_vertical']:
            env.reset(init_state_id=init)
            plan = make_plan(strat, d, env.scene())
            grasp = [w.pos for w in plan if w.name == 'close'][0]
            f = WaypointFollower(plan)
            E, G, B, done = [], [], [], False
            while not done:
                s = env.scene(); a = f.act(s)
                E.append(s.eef_pos); G.append(a[6]); B.append(s.bowl_pos)
                _, succ, done = env.step(a)
                if len(E) > 160: break
            out.append(dict(init=init, demo=d['name'], strat=strat, eef=np.array(E), grip=np.array(G), bowl=np.array(B),
                            grasp=grasp, rim_z=B[0][2] + BOWL_HEIGHT, success=bool(succ)))
pickle.dump(out, open(sys.argv[1], 'wb'))
print(len(out))
