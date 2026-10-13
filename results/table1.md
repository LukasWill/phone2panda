# Table 1: retargeting strategies, replayed in LIBERO-Goal's eval scenes (20 per demo)

Success rate with 95% Wilson interval over all rollouts. Rollouts of one demo are not independent (same human trajectory), so the per-demo column is the more honest unit.

| demos | strategy | success | demos with >= 50% success | most common failure: last waypoint reached |
|---|---|---|---|---|
| main (13) | absolute | 0% [0-2] (n=260) | 0/13 | carry (76) |
| main (13) | object_relative | 81% [76-85] (n=260) | 11/13 | release (25) |
| main (13) | intent | 100% [99-100] (n=260) | 13/13 | - |
| natural (6) | absolute | 0% [0-3] (n=120) | 0/6 | close (37) |
| natural (6) | object_relative | 41% [32-50] (n=120) | 3/6 | carry (19) |
| natural (6) | intent | 42% [33-51] (n=120) | 2/6 | release (39) |

## Per demo (successes out of 20)

| demo | group | grasp side, robot frame (deg) | absolute | object_relative | intent |
|---|---|---|---|---|---|
| 17c2b05951_lefthand | main | 98 | 0 | 18 | 20 |
| 30c08a055f_lefthand | main | 87 | 0 | 20 | 20 |
| 39a6539bbc_lefthand | main | 111 | 0 | 20 | 20 |
| 634b81c516_lefthand | main | 86 | 0 | 12 | 20 |
| 65ee6ec786 | main | 116 | 0 | 20 | 20 |
| 81c1cf2d63_lefthand | main | 104 | 1 | 20 | 20 |
| 92e0b86bc6_lefthand | main | 112 | 0 | 20 | 20 |
| 9a63cb1538_lefthand | main | 111 | 0 | 20 | 20 |
| a4558c8abd_lefthand | main | 108 | 0 | 1 | 20 |
| c521c4e934 | main | 149 | 0 | 4 | 20 |
| c5f525e833 | main | 107 | 0 | 20 | 20 |
| e53631acd6_lefthand | main | 143 | 0 | 20 | 20 |
| f9115215f9_lefthand | main | 106 | 0 | 16 | 20 |
| 0c3c9c0735_test2 | natural | 177 | 0 | 12 | 20 |
| 5783997456_test1 | natural | 181 | 0 | 19 | 9 |
| bd9e8d821d | natural | 272 | 0 | 0 | 0 |
| c7e4e24668 | natural | 196 | 0 | 14 | 12 |
| dcbcbf700f | natural | 272 | 0 | 0 | 0 |
| fd35966f27 | natural | 200 | 0 | 4 | 9 |
