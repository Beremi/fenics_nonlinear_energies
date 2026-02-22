## 4) Parallel custom FEniCS tables (np=4,8,16)

These runs use the same custom setup as the final serial configuration except `--no_near_nullspace` was required for MPI runs because current near-nullspace construction crashes in parallel (PETSc SEGV during nullspace build).

Common settings for all tables below: `ksp_type=gmres`, `pc_type=hypre`, `ksp_rtol=1e-1`, `ksp_max_it=30`, skip explicit `nodal/vec`, `--pc_setup_on_ksp_cap`, `--no_near_nullspace`.

### Level 1 — Custom FEniCS MPI nproc=4

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1062 | 21 | 304 | 0.3464120000 | 1.74e-06 | Energy change converged |
| 2 | 0.1106 | 21 | 252 | 1.3856370000 | 1.60e-06 | Energy change converged |
| 3 | 0.1111 | 20 | 261 | 3.1173460000 | 2.27e-06 | Energy change converged |
| 4 | 0.1283 | 23 | 344 | 5.5401490000 | 2.44e-07 | Energy change converged |
| 5 | 0.1078 | 20 | 277 | 8.6505010000 | 2.53e-07 | Energy change converged |
| 6 | 0.1228 | 22 | 322 | 12.4422680000 | 1.76e-07 | Energy change converged |
| 7 | 0.1108 | 21 | 275 | 16.9115650000 | 7.20e-08 | Energy change converged |
| 8 | 0.1248 | 21 | 309 | 22.0617420000 | 1.39e-07 | Energy change converged |
| 9 | 0.1340 | 22 | 280 | 27.8989760000 | 9.75e-09 | Energy change converged |
| 10 | 0.1356 | 23 | 344 | 34.4265020000 | 9.63e-08 | Energy change converged |
| 11 | 0.1271 | 23 | 316 | 41.6441060000 | 9.32e-08 | Energy change converged |
| 12 | 0.1184 | 21 | 277 | 49.5500720000 | 5.37e-08 | Energy change converged |
| 13 | 0.1132 | 20 | 250 | 58.1426610000 | 1.18e-10 | Energy change converged |
| 14 | 0.1174 | 22 | 280 | 67.4207960000 | 7.73e-08 | Energy change converged |
| 15 | 0.1281 | 21 | 297 | 77.3830590000 | 1.43e-09 | Energy change converged |
| 16 | 0.1216 | 21 | 291 | 88.0180590000 | 1.59e-08 | Energy change converged |
| 17 | 0.1446 | 24 | 365 | 99.3269870000 | 4.98e-09 | Energy change converged |
| 18 | 0.1444 | 23 | 346 | 111.3262050000 | 1.74e-08 | Energy change converged |
| 19 | 0.1484 | 22 | 340 | 124.0155390000 | 2.76e-09 | Energy change converged |
| 20 | 0.1197 | 20 | 264 | 137.3923340000 | 6.09e-09 | Energy change converged |
| 21 | 0.1479 | 24 | 355 | 151.4552330000 | 8.31e-09 | Energy change converged |
| 22 | 0.1226 | 20 | 267 | 166.2042260000 | 3.18e-08 | Energy change converged |
| 23 | 0.1559 | 23 | 334 | 181.6387080000 | 1.20e-08 | Energy change converged |
| 24 | 0.1255 | 22 | 319 | 197.7551340000 | 3.29e-05 | Energy change converged |

Summary: total time = `3.0268 s`, total Newton iters = `520`, total linear iters = `7269`, max relative error = `3.29e-05`, mean relative error = `1.66e-06`.

### Level 1 — Custom FEniCS MPI nproc=8

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0916 | 21 | 314 | 0.3464120000 | 1.74e-06 | Energy change converged |
| 2 | 0.0941 | 22 | 299 | 1.3856370000 | 1.60e-06 | Energy change converged |
| 3 | 0.1213 | 21 | 296 | 3.1173390000 | 2.71e-08 | Energy change converged |
| 4 | 0.1384 | 23 | 341 | 5.5401500000 | 4.25e-07 | Energy change converged |
| 5 | 0.1328 | 23 | 371 | 8.6504990000 | 2.20e-08 | Energy change converged |
| 6 | 0.1313 | 23 | 361 | 12.4422660000 | 1.53e-08 | Energy change converged |
| 7 | 0.1282 | 23 | 346 | 16.9115640000 | 1.29e-08 | Energy change converged |
| 8 | 0.1086 | 21 | 288 | 22.0617410000 | 9.33e-08 | Energy change converged |
| 9 | 0.1229 | 23 | 303 | 27.8989760000 | 9.75e-09 | Energy change converged |
| 10 | 0.1147 | 21 | 293 | 34.4264990000 | 9.19e-09 | Energy change converged |
| 11 | 0.1095 | 23 | 310 | 41.6441020000 | 2.87e-09 | Energy change converged |
| 12 | 0.1192 | 22 | 303 | 49.5500710000 | 3.35e-08 | Energy change converged |
| 13 | 0.1007 | 20 | 246 | 58.1426610000 | 1.18e-10 | Energy change converged |
| 14 | 0.1179 | 23 | 314 | 67.4207910000 | 3.11e-09 | Energy change converged |
| 15 | 0.1103 | 21 | 290 | 77.3830610000 | 2.44e-08 | Energy change converged |
| 16 | 0.1252 | 23 | 345 | 88.0180580000 | 4.57e-09 | Energy change converged |
| 17 | 0.1384 | 24 | 364 | 99.3269870000 | 4.98e-09 | Energy change converged |
| 18 | 0.1360 | 23 | 352 | 111.3262030000 | 5.83e-10 | Energy change converged |
| 19 | 0.1500 | 23 | 339 | 124.0155390000 | 2.76e-09 | Energy change converged |
| 20 | 0.1181 | 22 | 278 | 137.3923340000 | 6.09e-09 | Energy change converged |
| 21 | 0.1303 | 23 | 320 | 151.4552330000 | 8.31e-09 | Energy change converged |
| 22 | 0.1182 | 22 | 296 | 166.2042280000 | 4.38e-08 | Energy change converged |
| 23 | 0.1494 | 23 | 330 | 181.6387120000 | 1.00e-08 | Energy change converged |
| 24 | 0.1030 | 20 | 283 | 197.7551220000 | 3.28e-05 | Energy change converged |

Summary: total time = `2.9101 s`, total Newton iters = `533`, total linear iters = `7582`, max relative error = `3.28e-05`, mean relative error = `1.54e-06`.

### Level 1 — Custom FEniCS MPI nproc=16

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1391 | 21 | 263 | 0.3464110000 | 1.14e-06 | Energy change converged |
| 2 | 0.1585 | 22 | 320 | 1.3856350000 | 1.59e-07 | Energy change converged |
| 3 | 0.1268 | 20 | 241 | 3.1173510000 | 3.88e-06 | Energy change converged |
| 4 | 0.1533 | 23 | 355 | 5.5401480000 | 6.38e-08 | Energy change converged |
| 5 | 0.1324 | 21 | 309 | 8.6505080000 | 1.06e-06 | Energy change converged |
| 6 | 0.1308 | 21 | 299 | 12.4422700000 | 3.37e-07 | Energy change converged |
| 7 | 0.1220 | 20 | 265 | 16.9115640000 | 1.29e-08 | Energy change converged |
| 8 | 0.1270 | 21 | 265 | 22.0617430000 | 1.84e-07 | Energy change converged |
| 9 | 0.1384 | 23 | 297 | 27.8989780000 | 8.14e-08 | Energy change converged |
| 10 | 0.1227 | 20 | 259 | 34.4265080000 | 2.71e-07 | Energy change converged |
| 11 | 0.1443 | 23 | 309 | 41.6441100000 | 1.89e-07 | Energy change converged |
| 12 | 0.1385 | 22 | 312 | 49.5500700000 | 1.33e-08 | Energy change converged |
| 13 | 0.1400 | 21 | 292 | 58.1426660000 | 8.61e-08 | Energy change converged |
| 14 | 0.1622 | 24 | 377 | 67.4207910000 | 3.11e-09 | Energy change converged |
| 15 | 0.1229 | 20 | 255 | 77.3830630000 | 5.03e-08 | Energy change converged |
| 16 | 0.1463 | 23 | 327 | 88.0180580000 | 4.57e-09 | Energy change converged |
| 17 | 0.1532 | 24 | 368 | 99.3269870000 | 4.98e-09 | Energy change converged |
| 18 | 0.1528 | 24 | 344 | 111.3262030000 | 5.83e-10 | Energy change converged |
| 19 | 0.1620 | 23 | 339 | 124.0155410000 | 1.89e-08 | Energy change converged |
| 20 | 0.1422 | 23 | 328 | 137.3923330000 | 1.19e-09 | Energy change converged |
| 21 | 0.1402 | 22 | 298 | 151.4552350000 | 2.15e-08 | Energy change converged |
| 22 | 0.1286 | 20 | 302 | 166.2042240000 | 1.97e-08 | Energy change converged |
| 23 | 0.1753 | 23 | 320 | 181.6387080000 | 1.20e-08 | Energy change converged |
| 24 | 0.1424 | 23 | 353 | 197.7551650000 | 3.30e-05 | Energy change converged |

Summary: total time = `3.4019 s`, total Newton iters = `527`, total linear iters = `7397`, max relative error = `3.30e-05`, mean relative error = `1.69e-06`.

### Level 2 — Custom FEniCS MPI nproc=4

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.2933 | 9 | 94 | 0.5824270000 | 1.87e+00 | Energy change converged |
| 2 | 1.2993 | 34 | 433 | 0.8108400000 | 8.74e-06 | Energy change converged |
| 3 | 1.1110 | 27 | 421 | 1.8243680000 | 5.90e-07 | Energy change converged |
| 4 | 1.0552 | 27 | 353 | 3.2432380000 | 2.04e-07 | Energy change converged |
| 5 | 1.1642 | 28 | 451 | 5.0672630000 | 1.19e-06 | Energy change converged |
| 6 | 0.9669 | 25 | 328 | 7.2960200000 | 2.69e-07 | Energy change converged |
| 7 | 1.1997 | 29 | 441 | 9.9289580000 | 5.26e-07 | Energy change converged |
| 8 | 1.0861 | 26 | 403 | 12.9658030000 | 1.48e-07 | Energy change converged |
| 9 | 1.0976 | 27 | 379 | 16.4069420000 | 2.96e-08 | Energy change converged |
| 10 | 1.1225 | 26 | 402 | 20.2529480000 | 5.16e-08 | Energy change converged |
| 11 | 1.0395 | 25 | 363 | 24.5041650000 | 1.71e-07 | Energy change converged |
| 12 | 1.2112 | 28 | 446 | 29.1606590000 | 1.85e-08 | Energy change converged |
| 13 | 1.0333 | 24 | 375 | 34.2223430000 | 3.96e-08 | Energy change converged |
| 14 | 1.0119 | 23 | 355 | 39.6889330000 | 1.14e-07 | Energy change converged |
| 15 | 1.1260 | 26 | 388 | 45.5598050000 | 5.00e-07 | Energy change converged |
| 16 | 1.6332 | 34 | 655 | 51.8204000000 | 5.36e-07 | Energy change converged |
| 17 | 1.0097 | 23 | 389 | 58.4802480000 | 1.37e-09 | Energy change converged |
| 18 | 1.0959 | 24 | 421 | 65.5436910000 | 2.57e-08 | Energy change converged |
| 19 | 1.0405 | 23 | 387 | 73.0099330000 | 2.15e-07 | Energy change converged |
| 20 | 1.1080 | 24 | 400 | 80.8777560000 | 1.33e-07 | Energy change converged |
| 21 | 1.1772 | 25 | 444 | 89.1458850000 | 3.28e-07 | Energy change converged |
| 22 | 1.1122 | 25 | 378 | 97.8129100000 | 1.50e-06 | Energy change converged |
| 23 | 1.1977 | 26 | 441 | 106.8769220000 | 4.47e-07 | Energy change converged |
| 24 | 1.3447 | 28 | 501 | 116.3323410000 | 3.43e-05 | Energy change converged |

Summary: total time = `26.5368 s`, total Newton iters = `616`, total linear iters = `9648`, max relative error = `1.87e+00`, mean relative error = `7.81e-02`.

### Level 2 — Custom FEniCS MPI nproc=8

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.6221 | 26 | 370 | 0.2027080000 | 7.11e-07 | Energy change converged |
| 2 | 0.6477 | 26 | 379 | 0.8108350000 | 2.57e-06 | Energy change converged |
| 3 | 0.6702 | 25 | 359 | 1.8243800000 | 5.99e-06 | Energy change converged |
| 4 | 0.6576 | 27 | 365 | 3.2432380000 | 2.04e-07 | Energy change converged |
| 5 | 0.6552 | 26 | 370 | 5.0672540000 | 5.90e-07 | Energy change converged |
| 6 | 0.6435 | 26 | 372 | 7.2960160000 | 2.79e-07 | Energy change converged |
| 7 | 0.6661 | 26 | 370 | 9.9289540000 | 1.23e-07 | Energy change converged |
| 8 | 0.6414 | 25 | 358 | 12.9658110000 | 7.65e-07 | Energy change converged |
| 9 | 0.6363 | 25 | 352 | 16.4069430000 | 9.06e-08 | Energy change converged |
| 10 | 0.6045 | 24 | 346 | 20.2529470000 | 2.25e-09 | Energy change converged |
| 11 | 0.6415 | 25 | 373 | 24.5041620000 | 4.89e-08 | Energy change converged |
| 12 | 0.6863 | 26 | 373 | 29.1606660000 | 2.22e-07 | Energy change converged |
| 13 | 0.6695 | 26 | 403 | 34.2223460000 | 1.27e-07 | Energy change converged |
| 14 | 0.6388 | 24 | 365 | 39.6889280000 | 1.24e-08 | Energy change converged |
| 15 | 0.6660 | 25 | 390 | 45.5597980000 | 3.46e-07 | Energy change converged |
| 16 | 1.0414 | 35 | 677 | 51.8203820000 | 1.88e-07 | Energy change converged |
| 17 | 0.6556 | 24 | 383 | 58.4802550000 | 1.18e-07 | Energy change converged |
| 18 | 0.6760 | 24 | 415 | 65.5436900000 | 1.05e-08 | Energy change converged |
| 19 | 0.6639 | 24 | 419 | 73.0099180000 | 9.09e-09 | Energy change converged |
| 20 | 0.6997 | 24 | 409 | 80.8777730000 | 3.43e-07 | Energy change converged |
| 21 | 0.6878 | 25 | 418 | 89.1458820000 | 2.94e-07 | Energy change converged |
| 22 | 0.7401 | 27 | 423 | 97.8128840000 | 1.23e-06 | Energy change converged |
| 23 | 0.7043 | 25 | 385 | 106.8769100000 | 5.59e-07 | Energy change converged |
| 24 | 1.3826 | 45 | 934 | 116.3238510000 | 1.07e-04 | Energy change converged |

Summary: total time = `16.9981 s`, total Newton iters = `635`, total linear iters = `10008`, max relative error = `1.07e-04`, mean relative error = `5.09e-06`.

### Level 2 — Custom FEniCS MPI nproc=16

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.4746 | 24 | 316 | 0.2027200000 | 5.99e-05 | Energy change converged |
| 2 | 0.5919 | 28 | 438 | 0.8108430000 | 1.24e-05 | Energy change converged |
| 3 | 0.5438 | 26 | 383 | 1.8243680000 | 5.90e-07 | Energy change converged |
| 4 | 0.6211 | 28 | 444 | 3.2432460000 | 2.67e-06 | Energy change converged |
| 5 | 0.5553 | 27 | 374 | 5.0672590000 | 3.97e-07 | Energy change converged |
| 6 | 0.6025 | 28 | 387 | 7.2960170000 | 1.42e-07 | Energy change converged |
| 7 | 0.5686 | 26 | 384 | 9.9289570000 | 4.25e-07 | Energy change converged |
| 8 | 0.5674 | 26 | 392 | 12.9658030000 | 1.48e-07 | Energy change converged |
| 9 | 0.5334 | 25 | 374 | 16.4069480000 | 3.95e-07 | Energy change converged |
| 10 | 0.5557 | 26 | 390 | 20.2529480000 | 5.16e-08 | Energy change converged |
| 11 | 0.5617 | 27 | 403 | 24.5041610000 | 8.07e-09 | Energy change converged |
| 12 | 0.5267 | 25 | 391 | 29.1606590000 | 1.85e-08 | Energy change converged |
| 13 | 0.5142 | 25 | 382 | 34.2223470000 | 1.56e-07 | Energy change converged |
| 14 | 0.5792 | 26 | 410 | 39.6889260000 | 6.28e-08 | Energy change converged |
| 15 | 0.5719 | 26 | 409 | 45.5597920000 | 2.15e-07 | Energy change converged |
| 16 | 0.9602 | 39 | 757 | 51.8203790000 | 1.30e-07 | Energy change converged |
| 17 | 0.5257 | 24 | 383 | 58.4802490000 | 1.57e-08 | Energy change converged |
| 18 | 0.5258 | 24 | 410 | 65.5436900000 | 1.05e-08 | Energy change converged |
| 19 | 0.6479 | 25 | 445 | 73.0099120000 | 7.31e-08 | Energy change converged |
| 20 | 0.5806 | 25 | 444 | 80.8777530000 | 9.56e-08 | Energy change converged |
| 21 | 0.6375 | 26 | 504 | 89.1458740000 | 2.04e-07 | Energy change converged |
| 22 | 0.6581 | 28 | 509 | 97.8128410000 | 7.95e-07 | Energy change converged |
| 23 | 0.5592 | 26 | 425 | 106.8769000000 | 6.52e-07 | Energy change converged |
| 24 | 1.1500 | 47 | 1008 | 116.3238720000 | 1.07e-04 | Energy change converged |

Summary: total time = `14.6130 s`, total Newton iters = `657`, total linear iters = `10762`, max relative error = `1.07e-04`, mean relative error = `7.78e-06`.
