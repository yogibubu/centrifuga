# H2O quartic VPT4 diagnostic

All quartic constants are S-reduced and reported in kHz.
The benchmark-consistent axis override is `{'a': 2, 'b': 0, 'c': 1}`.

## Standard benchmark

Gaussian quartic S:
- DJ = 9514.856962
- DJK = 625332.3797
- DK = -583360.0865
- d1 = 110833.4519
- d2 = -109452.3504

CeDiTT order2 S:
- DJ = 9514.856740
- DJK = 625332.364341
- DK = -583360.070987
- d1 = 110833.449023
- d2 = -109452.347591

Maximum absolute standard-sector error relative to Gaussian:
- 0.016 kHz

Conclusion:
- The standard quartic sector is numerically under control for H2O once the benchmark axis convention is enforced.
- The water anomaly is therefore not caused by projection, axis conventions, or the standard VPT2 baseline.

## Fourth-order channel decomposition

H22:
- DJ = -67.650101
- DJK = -246.335379
- DK = 287.818749
- d1 = -97.130484
- d2 = 65.305719

H12H30:
- DJ = -0.025200
- DJK = 0.134768
- DK = -0.108702
- d1 = 0.002041
- d2 = -0.015680

H30H30:
- DJ = -3995824.208894
- DJK = 8571508.653867
- DK = -4704491.521416
- d1 = -863292.154481
- d2 = -865791.978084

Total VPT4:
- DJ = -3986.402202
- DJK = 9196.729449
- DK = -5287.672474
- d1 = -752.553796
- d2 = -975.194700

## Relative scale with respect to order2 baseline

H22 / order2:
- DJ: -7.11e-3
- DJK: -3.94e-4
- DK: -4.93e-4
- d1: -8.76e-4
- d2: -5.97e-4

H12H30 / order2:
- DJ: -2.65e-6
- DJK: 2.16e-7
- DK: 1.86e-7
- d1: 1.84e-8
- d2: 1.43e-7

H30H30 / order2:
- DJ: -4.20e-1
- DJK: 1.37e-2
- DK: 8.06e-3
- d1: -7.79e-3
- d2: 7.91e-3

## Interpretation

1. `H22` is small in every quartic constant. It cannot explain the failure of the standard VPT2 picture for water.
2. `H12H30` is negligible on the scale of the standard baseline and cannot explain the water anomaly.
3. The entire pathological behavior of H2O at quartic VPT4 level is driven by `H30H30`.
4. Within `H30H30`, the dominant diagonal scaffolds are `diag_0_iii_iii` and `diag_1_iii_iii`, with a secondary contribution from `diag_1_iii_iij_0`; the remaining scaffolds are much smaller.
5. Therefore the immediate technical problem for Paper 2 is not the standard sector, and not even the first geometric breakdown channel `H22`, but the quantitative formulation of the cubic-cubic anharmonic channel.
6. This is consistent with the fact that H2O is the molecule for which published VPT/VCI quartic comparisons are known to be most problematic.

## Immediate consequence for Paper 2

- H2CO and H2CS can be used as regular benchmark systems.
- H2S is an intermediate case where `H30H30` is already numerically important.
- H2O must be treated as the decisive stress test for the high-anharmonic channel.
- The paper should not claim predictive full-VPT4 quartic performance until the H2O cubic-cubic sector is understood quantitatively.
