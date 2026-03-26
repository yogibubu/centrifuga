# Linear `Dv` Aliev Equations Check Sheet

This note records the equations currently transcribed from the scanned linear
molecule formulas into
[`linear_dv_aliev_terms.py`](/Users/vincenzobarone/centrifugal/linear_dv_aliev_terms.py).

The purpose is review, not final manuscript typography.

Current working approximation:
- quartic input can be given either as a full 4-index field or as a reduced
  3-index field `k_(ii jk)`
- quartic contributions with four mutually different indices are set to zero
  whenever only the reduced field is available
- all repeated-index quartic patterns are kept

Program interface:
- the general driver is [`scripts/linear_aliev_cli.py`](/Users/vincenzobarone/centrifugal/scripts/linear_aliev_cli.py)
- it accepts `--quartic-mode full|reduced|none|auto`

Current Gaussian bootstrap convention for `zeta_nt`:
- working default: `component_tb`
- equivalent small-branch alternative: `component_ua`
- diagnostic-only alternatives: `component_ta`, `component_ub`, `maxabs`, `norm`

Reason:
- the scanned definition identifies `zeta_nt = zeta^x_(n,tb)`
- a systematic doublet scan on `C2H2` shows two numerically small equivalent
  branches (`component_tb`, `component_ua`) and a large branch
  (`component_ta`, `component_ub`, `maxabs`), while `norm` is the least
  compatible with the paper-aligned interpretation

Current Gaussian bootstrap convention for `C_n`:
- recommended default: `B_n^(xx) = -alpha_perp`
- rationale: matching
  `B_v = B_e + sum_n B_n^(xx) <q_n^2>`
  with
  `B_v = B_e - sum_n alpha_n (v_n + 1/2)`
  and `<q_n^2> = v_n + 1/2`
  gives directly `B_n^(xx) = -alpha_n`

## Definitions

`C_n = - B * B_n^(xx) / omega_n`

`X^(nt) = 2 B zeta_(nt) sqrt(omega_n omega_t) / (omega_n^2 - omega_t^2)`

`X_(nt) = B zeta_(nt) (omega_n^2 + omega_t^2) / (sqrt(omega_n omega_t) (omega_n^2 - omega_t^2))`

`r_n = 4 omega_n C_n (D_J / B - B^2 / omega_n^2)
     + (1/2) sum_(n',n'') k'_(n n' n'') C_(n') C_(n'')`

`r_(n n') = 3 omega_n omega_(n') C_n C_(n') / (4 B)
          + (1/2) sum_(n'') k'_(n n' n'') C_(n'')`

`r_(t t') = (1/2) sum_n k'_(t t' n) C_n`

## Auxiliary Families

`F^(n n') = B^2 sum_t zeta_(n t) zeta_(n' t) sqrt(omega_n omega_(n'))
          * (omega_n^2 + omega_(n')^2 - 2 omega_t^2)
          / ((omega_n^2 - omega_t^2) (omega_(n')^2 - omega_t^2))`

`F_(n n') = r_(n n')
          + B^2 sum_t zeta_(n t) zeta_(n' t)
          * (omega_n^2 omega_(n')^2 - omega_t^4)
          / (omega_t sqrt(omega_n omega_(n')) (omega_n^2 - omega_t^2) (omega_(n')^2 - omega_t^2))`

`U_(t t') = r_(t t') / (4 (omega_t + omega_(t')))
          + (B^2 / 8) sum_n zeta_(n t) zeta_(n t')
          * [ (omega_t + omega_n)^2 (omega_(t') - omega_n)^2
            + (omega_t - omega_n)^2 (omega_(t') + omega_n)^2 ]
          / (omega_n sqrt(omega_t omega_(t')) (omega_t^2 - omega_n^2) (omega_(t')^2 - omega_n^2))`

`V_(t t') = r_(t t') / (4 (omega_t - omega_(t')))
          + (B^2 / 8) sum_n zeta_(n t) zeta_(n t')
          * [ (omega_t + omega_(t'))^2 (omega_(t') + omega_n)^2 (omega_t + omega_(t') - 2 omega_n)
            - (omega_t - omega_(t'))^2 (omega_(t') - omega_n)^2 (omega_t + omega_(t') + 2 omega_n) ]
          / (omega_n sqrt(omega_t omega_(t')) (omega_t^2 - omega_n^2) (omega_(t')^2 - omega_n^2) (omega_t - omega_(t')))`

## Quartic Scalar

`L = -8 D_J^3 / B^2
   + (1/24) sum_(n,n',n'',n''') k'_(n n' n'' n''') C_n C_(n') C_(n'') C_(n''')
   + 8 B D_J sum_n (C_n^2 / omega_n)
   - sum_n (r_n^2 / (2 omega_n))`

## `beta_n`

`beta_n = 4 omega_n^2 C_n^2 (B / omega_n^2 - D_J / B^2)
        + (1/4) sum_(n',n'') k4_beta_n C_(n') C_(n'')
        - sum_(n') [k'_(n n n n') / (2 omega_(n'))] r_(n')
        - 4 B sum_(n',t) C_(n') zeta_(n' t)
          * [ (3 omega_n^2 + omega_t^2) omega_n^(3/2) C_n zeta_(n t)
            + (omega_n^2 + omega_t^2) omega_n^(3/2) C_n r_(n n') ]
          / [ sqrt(omega_n omega_t) (omega_n^2 - omega_t^2) ]
        + 4 sum_(n',t) [ X_(n t) X_(n' t) F^(n n') + X^(n t) X^(n' t) F_(n n') ]
        - 4 sum_(n',t) [ X_(n t) X^(n' t) F_(n' n) + X^(n t) X_(n' t) F^(n' n) ]
        - 16 sum_(t,t') [ U_(t t') (omega_t + omega_(t')) + V_(t t') (omega_(t') - omega_t) ]`

## `beta_t`

`beta_t = (1/4) sum_(n,n') k4_beta_t C_n C_(n')
        + B sum_(n,n') sqrt(omega_t^2 / (omega_n omega_(n'))) zeta_(n t) zeta_(n' t)
        - sum_n [k'_(t t n) / (2 omega_n)] r_n
        - 4 B sum_(n,n') C_n zeta_(n t)
          * [ 2 omega_t^2 omega_(n')^(3/2) C_(n') zeta_(n' t)
            + (3 omega_t^2 + omega_(n')^2) omega_(n')^(3/2) C_(n') r_(n n') ]
          / [ omega_t sqrt(omega_(n')) (omega_t^2 - omega_(n')^2) ]
        + 4 sum_(n,n') [ X_(n t) X_(n' t) F^(n n') + X^(n t) X^(n' t) F_(n n') ]
        - 4 sum_(n,n') [ X_(n t) X^(n' t) F_(n n') + X^(n t) X_(n' t) F^(n n') ]
        - 16 sum_(t') [ U_(t t') (omega_t + omega_(t')) + V_(t t') (omega_(t') - omega_t) ]`

## Final Linear State Dependence

`Dv(v) = D_J - sum_n beta_n (v_n + 1/2) - sum_t beta_t (v_t + 1)`
