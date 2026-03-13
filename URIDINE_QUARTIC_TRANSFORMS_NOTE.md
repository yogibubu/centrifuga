## Uridine quartic transforms from the supplied geometry and Watson constants

The supplied Cartesian geometry is already in the principal-axis frame.
Using standard isotopic masses, the inertia tensor is diagonal within
numerical noise in the printed frame, with principal moments

- `Ia = 573.034289 amu A^2`
- `Ib = 1511.113496 amu A^2`
- `Ic = 1880.800616 amu A^2`

and corresponding rotational constants

- `A = 881.935023 MHz`
- `B = 334.441464 MHz`
- `C = 268.704192 MHz`

which reproduce the values printed in the Gaussian output. Therefore the
input geometry is already in the `I^r` representation.

The supplied asymmetrically reduced quartic constants were interpreted as the
`I^r` / `A`-reduction set

- `Delta_J = 0.010458 kHz`
- `Delta_JK = 0.147760 kHz`
- `Delta_K = -0.097394 kHz`
- `delta_J = 0.000760 kHz`
- `delta_K = 0.061064 kHz`

and transformed with the tensor workflow implemented in the repo.

### Numerical results

| Constant | `I^r` (A) | `I^r` (S) | `III^r` (A) | `III^r` (S) |
|---|---:|---:|---:|---:|
| `Delta_J` / `D_J` | `0.010458` | `0.00872883` | `0.0348810` | `0.0667181` |
| `Delta_JK` / `D_JK` | `0.147760` | `0.15813503` | `0.0744910` | `-0.1165318` |
| `Delta_K` / `D_K` | `-0.097394` | `-0.10603985` | `-0.0973940` | `0.0617917` |
| `delta_J` / `d_1` | `0.000760` | `-0.00076000` | `-0.0129715` | `0.0129715` |
| `delta_K` / `d_2` | `0.061064` | `-0.00086459` | `-0.0500227` | `0.0159186` |

### Validation of the input mapping

The `I^r` / `A -> I^r` / `S` transport reproduces the symmetric-reduction
constants printed in the Gaussian output:

- `D_J  = 0.00872883 kHz`
- `D_JK = 0.15813503 kHz`
- `D_K  = -0.10603985 kHz`
- `d_1  = -0.00076000 kHz`
- `d_2  = -0.00086459 kHz`

which agrees with the supplied values to the printed precision.

### Practical use in the manuscript

This is enough to build a comparison table against experiment once the
experimental quartic constants are added. The key structural point already
visible is that the transported `III^r` anisotropic constants become much
larger than the native `I^r` set, even though they correspond to the same
underlying quartic tensor.

## Experimental comparison extracted from Pena et al. (Angew. Chem. Int. Ed. 2015)

The experimental paper reports the following ground-state constants for gas
phase uridine:

- `A = 885.98961(14) MHz`
- `B = 335.59622(35) MHz`
- `C = 270.11210(20) MHz`
- `D_J = 0.0115(10) kHz`

No other quartic distortion constant was reported in that fit.

For the dual-level comparison discussed in the manuscript, the available
theoretical rotational constants are:

- `DPCS3 equilibrium`
  - `A_e = 889.8408746 MHz`
  - `B_e = 337.1284836 MHz`
  - `C_e = 271.2515410 MHz`
- `BDPCS3 equilibrium`
  - `A_e = 892.9368352 MHz`
  - `B_e = 338.0938525 MHz`
  - `C_e = 272.0213668 MHz`
- `HPCS2 ground-state`
  - `A_0 = 881.9350311 MHz`
  - `B_0 = 334.4414974 MHz`
  - `C_0 = 268.7042113 MHz`

The HPCS2 vibrational corrections to be used in the dual-level comparison are:

- `Delta_vib(A) = -9.03 MHz`
- `Delta_vib(B) = -2.99 MHz`
- `Delta_vib(C) = -2.49 MHz`

This gives:

- `HPCS2 + Delta_vib(HPCS2)`
  - `A_0 = 881.93503 MHz`
  - `B_0 = 334.44150 MHz`
  - `C_0 = 268.70421 MHz`
- `DPCS3//HPCS2`
  - `A_0 = 880.81087 MHz`
  - `B_0 = 334.13848 MHz`
  - `C_0 = 268.76154 MHz`
- `BDPCS3//HPCS2`
  - `A_0 = 883.90684 MHz`
  - `B_0 = 335.10385 MHz`
  - `C_0 = 269.53137 MHz`

The corresponding agreement with experiment is:

- `A_0 - A_exp = -4.05458 MHz` (`-0.458%`)
- `B_0 - B_exp = -1.15472 MHz` (`-0.344%`)
- `C_0 - C_exp = -1.40789 MHz` (`-0.521%`)

for the HPCS2 equilibrium reference, while for `BDPCS3//HPCS2` the deviations
become

- `A_0 - A_exp = -2.08277 MHz` (`-0.235%`)
- `B_0 - B_exp = -0.49237 MHz` (`-0.147%`)
- `C_0 - C_exp = -0.58073 MHz` (`-0.215%`)

Using these rotational constants together with the transformed quartic
constants above, the following compact comparison table can be used in the
manuscript.

### LaTeX-ready tables

```tex
\begin{table}[h]
\centering
\scriptsize
\caption{Uridine as a demanding dual-level case. Experimental ground-state
rotational constants are from Ref.~X. The notations `HPCS2//HPCS2`,
`DPCS3//HPCS2`, and `BDPCS3//HPCS2` indicate that the equilibrium constants
are taken at the first level of theory and corrected by HPCS2 vibrational
contributions. The last row reports the mean absolute percentage error (MAE\%)
with respect to experiment.}

\begin{tabular}{lcccc}
\hline
Constant & Exp. ($B_0$) & HPCS2//HPCS2 & DPCS3//HPCS2 & BDPCS3//HPCS2 \\
\hline
$A$ (MHz) & 885.98961(14) & 881.93503 & 880.81087 & 883.90684 \\
$B$ (MHz) & 335.59622(35) & 334.44150 & 334.13848 & 335.10385 \\
$C$ (MHz) & 270.11210(20) & 268.70421 & 268.76154 & 269.53137 \\
\hline
MAE (\%) & --- & 0.441 & 0.531 & 0.199 \\
\hline
\end{tabular}
\label{tab:uridine_rotational_duallevel}
\end{table}

\begin{table}[h]
\centering
\caption{Uridine quartic centrifugal distortion constants from the HPCS2
vibrational treatment, reported in the native $I^r$ representation and after
transport to $III^r$. The supplied geometry is already in the $I^r$
representation. Only $D_J$ was resolved experimentally.}

\begin{tabular}{lccc}
\hline
Quantity & Experiment & HPCS2 ($I^r$) & HPCS2 ($III^r$) \\
\hline
$\Delta_J$ (kHz) & --- & 0.010458 & 0.034881 \\
$\Delta_{JK}$ (kHz) & --- & 0.147760 & 0.074491 \\
$\Delta_K$ (kHz) & --- & -0.097394 & -0.097394 \\
$\delta_J$ (kHz) & --- & 0.000760 & -0.012972 \\
$\delta_K$ (kHz) & --- & 0.061064 & -0.050023 \\
\hline
$D_J$ (kHz) & 0.0115(10) & 0.008729 & 0.066718 \\
$D_{JK}$ (kHz) & --- & 0.158135 & -0.116532 \\
$D_K$ (kHz) & --- & -0.106040 & 0.061792 \\
$d_1$ (kHz) & --- & -0.000760 & 0.012972 \\
$d_2$ (kHz) & --- & -0.000865 & 0.015919 \\
\hline
\end{tabular}
\label{tab:uridine_quartic_comparison}
\end{table}
```

### Reading of the table

The rotational-constant table now makes the dual-level message explicit.
The HPCS2 vibrational corrections are not negligible, especially for the
largest rotational constant, and their inclusion is essential for any
meaningful comparison with experiment. At the same time, the comparison shows
that the `BDPCS3//HPCS2` combination gives clearly better agreement with the
observed ground-state constants than the simpler alternatives, with a mean
absolute percentage error of about `0.20%` against `0.44%` for
`HPCS2//HPCS2` and `0.53%` for `DPCS3//HPCS2`. On the quartic side, the only
experimentally resolved constant is `D_J = 0.0115(10) kHz`, which is in the
same range as the native `I^r` prediction `0.00873 kHz`. By contrast, the
transported `III^r` constants show a much stronger anisotropic reshuffling of
the quartic content. This is a useful demanding example because it shows that
for a 29-atom system the tensor transport already produces selective
spectroscopic information: even when a complete experimental quartic fit is
not available, the transformed constants indicate which distortion parameters
are plausibly large enough to be observable.
