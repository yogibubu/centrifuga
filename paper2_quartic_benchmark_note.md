# Paper 2 quartic benchmark note

This note contains two different kinds of reference values:

- `H22`: validated benchmark-level channel numbers used for code verification.
- `H12H30`, `H30H30`: working-reference channel numbers used for reconstruction
  and diagnostics while those implementations are still under active
  development.

All quartic constants below are S-reduced and reported in kHz. The benchmark-consistent axis override is inferred from the Gaussian quartic benchmark itself.

## H2O

Gaussian quartic S = {'DJ': 9514.856962, 'DJK': 625332.3797, 'DK': -583360.0865, 'd1': 110833.4519, 'd2': -109452.3504}
CeDiTT order2 S = {'DJ': 9514.85674, 'DK': -583360.070987, 'DJK': 625332.364341, 'd1': 110833.449023, 'd2': -109452.347591}
H22 = {'DJ': -67.650101, 'DK': 287.818749, 'DJK': -246.335379, 'd1': -97.130484, 'd2': 65.305719}
H12H30 = {'DJ': -25.200069, 'DK': -108.701921, 'DJK': 134.767919, 'd1': 2.040719, 'd2': -15.680396}
H30H30 = {'DJ': -3995824.208894, 'DK': -4704491.521416, 'DJK': 8571508.653867, 'd1': -863292.154481, 'd2': -865791.978084}
Total-order2 = {'DJ': -3995917.059088, 'DK': -4704312.403107, 'DJK': 8571397.084819, 'd1': -863387.244528, 'd2': -865742.352483}

## H2S

Gaussian quartic S = {'DJ': 43377.34743, 'DJK': -183504.6271, 'DK': 170170.1568, 'd1': 12701.62534, 'd2': 8112.165724}
CeDiTT order2 S = {'DJ': 43377.34748, 'DK': 170170.156943, 'DJK': -183504.627534, 'd1': 12701.625348, 'd2': 8112.165746}
H22 = {'DJ': -5.917341, 'DK': -3.88703, 'DJK': 1.331355, 'd1': -4.153058, 'd2': 1.469627}
H12H30 = {'DJ': 2.170509, 'DK': 11.88606, 'DJK': -14.012266, 'd1': 0.134087, 'd2': 0.876202}
H30H30 = {'DJ': -102207.191033, 'DK': 256529.210998, 'DJK': -165183.974944, 'd1': -33370.375589, 'd2': 117.451329}
Total-order2 = {'DJ': -102210.937975, 'DK': 256537.209596, 'DJK': -165196.655389, 'd1': -33374.394592, 'd2': 119.797137}

## H2CO

Gaussian quartic S = {'DJ': 64.223625, 'DJK': 1261.589002, 'DK': 18442.65629, 'd1': 9.10916, 'd2': -2.041246}
CeDiTT order2 S = {'DJ': 64.223626, 'DK': 18442.656281, 'DJK': 1261.589006, 'd1': 9.10916, 'd2': -2.041246}
H22 = {'DJ': -0.003691, 'DK': -8.736904, 'DJK': -0.17658, 'd1': -0.001452, 'd2': 0.000783}
H12H30 = {'DJ': 0.000308, 'DK': -0.928541, 'DJK': -0.000473, 'd1': 6.9e-05, 'd2': -2.1e-05}
H30H30 = {'DJ': 0.0, 'DK': 0.0, 'DJK': -0.0, 'd1': 0.0, 'd2': 0.0}
Total-order2 = {'DJ': -0.003383, 'DK': -9.665491, 'DJK': -0.177056, 'd1': -0.001383, 'd2': 0.000761}

## H2CS

Gaussian quartic S = {'DJ': 17.760571, 'DJK': 487.63486, 'DK': 21748.52487, 'd1': 1.065891, 'd2': -0.138248}
CeDiTT order2 S = {'DJ': 17.760571, 'DK': 21748.524656, 'DJK': 487.634855, 'd1': 1.065891, 'd2': -0.138248}
H22 = {'DJ': -0.000532, 'DK': -11.435701, 'DJK': -0.037965, 'd1': -0.000114, 'd2': 5e-05}
H12H30 = {'DJ': -2.4e-05, 'DK': -0.922467, 'DJK': -0.010106, 'd1': -3e-06, 'd2': 2e-06}
H30H30 = {'DJ': 0.002906, 'DK': 6.539727, 'DJK': -6.542633, 'd1': 0.0, 'd2': 0.001453}
Total-order2 = {'DJ': 0.00235, 'DK': -5.818497, 'DJK': -6.590705, 'd1': -0.000116, 'd2': 0.001505}
