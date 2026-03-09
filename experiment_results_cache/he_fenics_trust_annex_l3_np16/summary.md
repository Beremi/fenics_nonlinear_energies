# FEniCS custom trust-radius sweep

| Case | Trust region | Radius init | Completed steps | First failed step | Failure mode | Total Newton | Total linear | Total time [s] | Assembly [s] | PC init [s] | KSP solve [s] | Line search [s] | Final energy | Result |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ls_only | no | - | 24 | - | - | 1104 | 19083 | 73.113 | 14.844 | 8.105 | 36.051 | 12.660 | 93.705 | completed |
| tr_r0_2 | yes | 0.200 | 17 | 18 | hard-timeout | - | - | - | - | - | - | - | - | hard-timeout |
| tr_r0_5 | yes | 0.500 | 24 | - | - | 1346 | 25792 | 98.258 | 18.714 | 11.971 | 47.490 | 18.229 | 93.705 | completed |
| tr_r1_0 | yes | 1.000 | 24 | - | - | 1168 | 19565 | 81.813 | 16.428 | 8.917 | 38.481 | 16.402 | 93.705 | completed |
| tr_r2_0 | yes | 2.000 | 24 | - | - | 1117 | 17835 | 76.503 | 15.936 | 8.050 | 35.684 | 15.313 | 93.705 | completed |
