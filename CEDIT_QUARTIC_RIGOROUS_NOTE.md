# CeDiTT Quartic Note: Rigorous 3+3 Structure and Pseudoinverse Gauge Fixing

## 1. Quartic tensor and reduced matrix

Work in the compressed pair-pair quartic basis

```text
tau = (tau_aaaa, tau_bbbb, tau_cccc, tau_aabb, tau_aacc, tau_bbcc).
```

The Gaussian-consistent reduced matrix is

```text
Tau' =
[ tau_aaaa      tau_aabb/2    tau_aacc/2 ]
[ tau_aabb/2    tau_bbbb      tau_bbcc/2 ]
[ tau_aacc/2    tau_bbcc/2    tau_cccc   ].
```

This is a real symmetric `3x3` tensor in `R^3`.

Its complete spectral parameterization is therefore

```text
Tau' = R diag(lambda_1, lambda_2, lambda_3) R^T
```

with:

- `lambda_1, lambda_2, lambda_3`: eigenvalues,
- `R in SO(3)`: orientation of the eigenframe.

This gives a natural `3+3` tensorial description.

## 2. Watson map and gauge family

The quartic Watson constants form a 5-vector, while the compressed quartic
tensor has 6 components. For a fixed reduction (A or S), the linear map

```text
tau(6) -> Watson(5)
```

has rank 5. Therefore its nullspace is 1-dimensional.

This means:

- a given Watson quartic parameter set does **not** determine a unique `tau`;
- instead it determines an affine 1-parameter family of tensors

```text
tau + c n,
```

where `n` spans the nullspace of the forward map.

## 3. Explicit null direction

Using the A-reduction formulas with

```text
T = Tau' / 4,
Sigma = (2A-B-C)/(B-C),
```

the null direction in compressed pair-pair coordinates is

```text
n = (0, 0, 0, 1, (Sigma-1)/2, -(Sigma+1)/2)
```

up to overall scaling.

Equivalently, in `Tau'` form the null direction is

```text
N' =
[ 0            1/2              (Sigma-1)/4 ]
[ 1/2          0               -(Sigma+1)/4 ]
[ (Sigma-1)/4  -(Sigma+1)/4     0           ].
```

The same nullspace is shared by A and S reductions because the A<->S quartic
conversion is invertible at fixed rotational constants.

## 4. Important distinction

The null direction above is **not** a physicality condition on quartic tensors.

It does **not** mean that physical quartic tensors satisfy a constraint

```text
F(Tau') = 0.
```

Instead, it means that:

- all tensors in the affine family `Tau' + c N'` correspond to the same Watson
  quartic constants;
- the Moore-Penrose inverse selects one specific representative in that family.

So the scalar equation

```text
2 Tau'12 + (Sigma-1) Tau'13 - (Sigma+1) Tau'23 = 0
```

is the **pseudoinverse gauge-fixing condition**, not a condition of physical
admissibility.

## 5. Consequence for the 3+2 problem

The `3+3` spectral description of `Tau'` is the correct tensorial starting
point.

The reduction to 5 parameters is obtained because Watson spectroscopy sees only
the quotient space

```text
Sym(3) / span(N').
```

Therefore the CeDiTT-side `3+2` structure should be understood as:

- `3` genuinely spatial quantities, naturally chosen as the eigenvalues of
  `Tau'`,
- `2` additional coordinates describing the orbit of `Tau'` modulo the
  1-parameter null family selected by the Watson projection.

## 6. What is currently established in code

The workflow now exposes:

- the `3+3` spectral coordinates of `Tau'`,
- the pseudoinverse gauge-fixing scalar,
- reconstruction from spectral coordinates back to `Tau'`.

This is mathematically rigorous and numerically closed.

## 7. What remains open

What is still open is the final CeDiTT-side choice of the two residual
coordinates. That choice must:

1. be well defined on the quotient by the null direction,
2. preserve the interpretation of the three eigenvalues as spatial invariants,
3. remain convenient for representation transforms and app usage.

So the correct statement is:

- `3+3` is the fundamental tensorial structure,
- `3+2` is a reduced coordinate choice on the Watson quotient,
- the pseudoinverse constraint is one possible gauge-fixing, not the physics
  itself.
