"""How precise must the scripted teacher be? Plans built from a WRONG bowl or plate position (random
horizontal direction), executed in the true scene; 5 consistent demos x 10 evaluation scenes per error size.

    python tools/teacher_tolerance.py      # ~15 min on 2 CPU cores, physics only
"""
import sys, json, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dataclasses import replace
from pathlib import Path
DEMOS = "f9115215f9_lefthand,c5f525e833,9a63cb1538_lefthand,92e0b86bc6_lefthand,65ee6ec786".split(",")
ENV = None
def init():
    global ENV
    from p2p.sim.env import TaskEnv
    ENV = TaskEnv(render=False, hard_reset=False)
def roll(job):
    from p2p.retarget.strategies import make_plan
    from p2p.sim.scripted import WaypointFollower
    demo, init_id, which, delta, ang = job
    ENV.reset(init_state_id=init_id)
    s = ENV.scene()
    off = delta * np.array([np.cos(ang), np.sin(ang), 0.0])
    s_wrong = replace(s, bowl_pos=s.bowl_pos + off) if which == 'bowl' else replace(s, plate_pos=s.plate_pos + off)
    f = WaypointFollower(make_plan('object_relative', demo, s_wrong))
    done = False
    while not done:
        _, succ, done = ENV.step(f.act(ENV.scene()))
    return dict(which=which, delta=delta, success=succ)
if __name__ == '__main__':
    from p2p.human.extract import load
    root = Path(__file__).resolve().parents[1]
    demos = [load(root / 'data/processed' / f'{n}.npz') for n in DEMOS]
    rng = np.random.default_rng(0)
    jobs = []
    for which in ['bowl', 'plate']:
        for delta in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]:
            if which == 'plate' and delta == 0: continue
            for d in demos:
                for i in range(0, 50, 5):
                    jobs.append((d, i, which, delta, rng.uniform(0, 2 * np.pi)))
    with mp.get_context('spawn').Pool(2, initializer=init) as pool:
        res = pool.map(roll, jobs, chunksize=2)
    import pandas as pd
    df = pd.DataFrame(res)
    print(df.groupby(['which', 'delta']).success.agg(['mean', 'count']))
