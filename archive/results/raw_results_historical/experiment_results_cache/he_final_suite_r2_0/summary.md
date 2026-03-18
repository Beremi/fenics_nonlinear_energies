# HE Final Suite Summary

Data note:
- This file is only the quick-reference index.
- Full aggregated data are in `summary.json` in the same directory.
- Full per-case data are in `*_steps*_l*_np*.json` and matching `*.log` files.
- Each per-case JSON stores per-step data in `result.steps`, with per-Newton details in `history` and per-Newton linear timing in `linear_timing`.

| Solver | Total steps | Level | MPI | Completed steps | Total Newton | Total linear | Total time [s] | Mean step [s] | Max step [s] | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fenics_custom | 24 | 3 | 8 | 24 | 889 | 14392 | 125.498 | 5.229 | 7.041 | completed |
| fenics_custom | 24 | 3 | 16 | 24 | 854 | 13643 | 57.904 | 2.413 | 3.326 | completed |
| fenics_custom | 24 | 3 | 32 | 24 | 850 | 13489 | 39.700 | 1.654 | 2.327 | completed |
| fenics_custom | 24 | 4 | 8 | 7 | 321 | 5346 | 503.517 | 62.940 | 101.072 | kill-switch |
| fenics_custom | 24 | 4 | 16 | 24 | 1111 | 22287 | 940.888 | 39.204 | 95.552 | completed |
| fenics_custom | 24 | 4 | 32 | 24 | 1246 | 31867 | 718.576 | 29.941 | 45.055 | completed |
| fenics_custom | 96 | 1 | 1 | 96 | 1155 | 16710 | 11.642 | 0.121 | 0.144 | completed |
| fenics_custom | 96 | 1 | 2 | 96 | 1140 | 16538 | 9.694 | 0.101 | 0.120 | completed |
| fenics_custom | 96 | 1 | 4 | 96 | 1141 | 16790 | 9.141 | 0.095 | 0.115 | completed |
| fenics_custom | 96 | 1 | 8 | 96 | 1120 | 16567 | 9.146 | 0.095 | 0.124 | completed |
| fenics_custom | 96 | 1 | 16 | 96 | 1143 | 16513 | 11.139 | 0.116 | 0.146 | completed |
| fenics_custom | 96 | 1 | 32 | 96 | 1149 | 16370 | 17.309 | 0.180 | 0.216 | completed |
| fenics_custom | 96 | 2 | 1 | 96 | 1507 | 24683 | 119.272 | 1.242 | 2.636 | completed |
| fenics_custom | 96 | 2 | 2 | 96 | 1502 | 23922 | 59.987 | 0.625 | 1.121 | completed |
| fenics_custom | 96 | 2 | 4 | 96 | 1519 | 24447 | 38.617 | 0.402 | 0.769 | completed |
| fenics_custom | 96 | 2 | 8 | 96 | 1536 | 24541 | 28.014 | 0.292 | 0.558 | completed |
| fenics_custom | 96 | 2 | 16 | 96 | 1538 | 24515 | 24.634 | 0.257 | 0.402 | completed |
| fenics_custom | 96 | 2 | 32 | 96 | 1522 | 24117 | 28.983 | 0.302 | 0.625 | completed |
| fenics_custom | 96 | 3 | 1 | 96 | 1885 | 27994 | 1305.398 | 13.598 | 16.336 | completed |
| fenics_custom | 96 | 3 | 2 | 96 | 1837 | 27527 | 621.271 | 6.472 | 9.015 | completed |
| fenics_custom | 96 | 3 | 4 | 96 | 1871 | 27911 | 374.171 | 3.898 | 5.148 | completed |
| fenics_custom | 96 | 3 | 8 | 96 | 1898 | 27971 | 255.687 | 2.663 | 3.345 | completed |
| fenics_custom | 96 | 3 | 16 | 96 | 1865 | 27715 | 118.992 | 1.239 | 1.588 | completed |
| fenics_custom | 96 | 3 | 32 | 96 | 1904 | 27801 | 84.894 | 0.884 | 1.028 | completed |
| fenics_custom | 96 | 4 | 16 | 96 | 1947 | 28417 | 1378.126 | 14.355 | 20.052 | completed |
| fenics_custom | 96 | 4 | 32 | 96 | 1898 | 27681 | 926.639 | 9.652 | 15.015 | completed |
| jax_petsc_element | 24 | 3 | 8 | 24 | 1104 | 17963 | 207.586 | 8.649 | 13.090 | completed |
| jax_petsc_element | 24 | 3 | 16 | 24 | 1134 | 18376 | 102.527 | 4.272 | 5.380 | completed |
| jax_petsc_element | 24 | 3 | 32 | 24 | 1113 | 17384 | 73.759 | 3.073 | 4.013 | completed |
| jax_petsc_element | 24 | 4 | 8 | 1 | 83 | 1152 | 178.306 | 89.153 | 101.098 | kill-switch |
| jax_petsc_element | 24 | 4 | 16 | 3 | 235 | 3990 | 267.968 | 66.992 | 101.256 | kill-switch |
| jax_petsc_element | 24 | 4 | 32 | 24 | 1902 | 39933 | 1208.325 | 50.347 | 70.577 | failed |
| jax_petsc_element | 96 | 1 | 1 | 96 | 1203 | 18044 | 14.453 | 0.151 | 0.208 | completed |
| jax_petsc_element | 96 | 1 | 32 | 96 | 1215 | 17898 | 18.908 | 0.197 | 0.246 | completed |
| jax_petsc_element | 96 | 2 | 1 | 96 | 1800 | 30930 | 127.092 | 1.324 | 7.319 | completed |
| jax_petsc_element | 96 | 2 | 32 | 96 | 1762 | 29659 | 40.085 | 0.418 | 2.665 | completed |
| jax_petsc_element | 96 | 3 | 1 | 96 | 2097 | 32609 | 1223.320 | 12.743 | 44.379 | completed |
| jax_petsc_element | 96 | 3 | 2 | 96 | 2108 | 32851 | 678.567 | 7.068 | 26.877 | completed |
| jax_petsc_element | 96 | 3 | 4 | 96 | 2096 | 33130 | 461.657 | 4.809 | 16.977 | completed |
| jax_petsc_element | 96 | 3 | 8 | 96 | 2071 | 32413 | 385.740 | 4.018 | 11.175 | completed |
| jax_petsc_element | 96 | 3 | 16 | 96 | 2120 | 32702 | 185.180 | 1.929 | 6.195 | completed |
| jax_petsc_element | 96 | 3 | 32 | 96 | 2073 | 31919 | 132.395 | 1.379 | 4.354 | completed |
| jax_petsc_element | 96 | 4 | 16 | 96 | 2373 | 37665 | 2530.093 | 26.355 | 37.959 | completed |
| jax_petsc_element | 96 | 4 | 32 | 96 | 2312 | 36351 | 1305.155 | 13.595 | 19.267 | completed |
