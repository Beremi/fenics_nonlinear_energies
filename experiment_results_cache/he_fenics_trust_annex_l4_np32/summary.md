# HE FEniCS Trust Sweep

| case | trust region | radius | completed steps | first failed step | failure mode | total Newton | total linear | total time [s] | assembly [s] | pc init [s] | ksp solve [s] | line search [s] | final energy | result |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ls_only | no | - | 3 | 4 | hard-timeout | - | - | - | - | - | - | - | - | hard-timeout |
| tr_r0_2 | yes | 0.200 | 4 | 5 | hard-timeout | - | - | - | - | - | - | - | - | hard-timeout |
| tr_r0_5 | yes | 0.500 | 4 | 5 | hard-timeout | - | - | - | - | - | - | - | - | hard-timeout |
| tr_r1_0 | yes | 1.000 | 4 | 5 | hard-timeout | - | - | - | - | - | - | - | - | hard-timeout |
| tr_r2_0 | yes | 2.000 | 4 | 5 | hard-timeout | - | - | - | - | - | - | - | - | hard-timeout |
