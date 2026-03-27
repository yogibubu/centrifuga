# Linear Aliev: Proof Structure For `uv_parallel`

This note isolates the only point that now matters for the parallel `uv`
sector:

- why `principal_direction` is not an admissible scalar input for the operator
  families `X/F/U/V`;
- why the off-diagonal reduction `pair_offdiag` is the correct working choice
  under the present linear-molecule canonical basis.

The goal is not to restate the whole audit. The goal is to write the argument
as a short derivation.

## 1. The Object We Actually Have

For a linear molecule with a doubly-degenerate perpendicular pair, the Gaussian
bootstrap gives, for each parallel mode `n` and each degenerate pair `t`, a
real `2x2` Coriolis block

\[
M_{nt}=
\begin{pmatrix}
\zeta_{x,n,t_a} & \zeta_{x,n,t_b}\\
\zeta_{y,n,t_a} & \zeta_{y,n,t_b}
\end{pmatrix}.
\]

In the actual benchmark data (`C2H2`) these blocks are numerically:

- nearly traceless,
- nearly symmetric,
- dominated by a large diagonal-antisymmetric part plus a much smaller
  off-diagonal symmetric part.

So, to excellent approximation, each block can be written as

\[
M_{nt} \approx a_{nt} S_1 + s_{nt} S_2,
\]

with

\[
S_1=
\begin{pmatrix}
1 & 0\\
0 & -1
\end{pmatrix},
\qquad
S_2=
\begin{pmatrix}
0 & 1\\
1 & 0
\end{pmatrix},
\]

that is,

\[
M_{nt}\approx
\begin{pmatrix}
a_{nt} & s_{nt}\\
s_{nt} & -a_{nt}
\end{pmatrix}.
\]

The quantities used in the current code are exactly

\[
a_{nt} = \frac{M_{11}-M_{22}}{2},
\qquad
s_{nt} = \frac{M_{12}+M_{21}}{2}.
\]

## 2. How The Block Transforms Under A Rotation Of The Degenerate Basis

If the degenerate pair basis is rotated by an angle `\theta`,

\[
M'_{nt}=R(\theta)^T M_{nt} R(\theta),
\qquad
R(\theta)=
\begin{pmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{pmatrix},
\]

then the coefficients transform as

\[
a'_{nt}=a_{nt}\cos 2\theta + s_{nt}\sin 2\theta,
\]
\[
s'_{nt}=-a_{nt}\sin 2\theta + s_{nt}\cos 2\theta.
\]

So `(a_{nt},s_{nt})` is a spin-2 doublet under rotations of the degenerate
subspace. Its Euclidean norm

\[
\rho_{nt}=\sqrt{a_{nt}^2+s_{nt}^2}
\]

is invariant, but the individual components `a_{nt}` and `s_{nt}` are not.

This is the key structural fact.

## 3. Why `principal_direction` Is Wrong For The Operator Sector

The current `principal_direction` reduction is, in practice, the largest
eigenvalue of the symmetric block. For a traceless symmetric `2x2` matrix this
is just

\[
\pm \rho_{nt}.
\]

So `principal_direction` does **not** return a component of the block. It
returns its invariant norm.

But the Aliev formulas do not ask for an invariant norm of a `2x2` block.
They ask for a scalar quantity written as

\[
\zeta_{nt},
\]

and the scanned definition used during the reconstruction was of the form

\[
\zeta_{nt}=\zeta^x_{n,tb},
\]

that is: a **specific component**, not the norm of the whole degenerate block.

Therefore `principal_direction` is conceptually wrong for the operator sector:

- it replaces the required component by the block norm;
- it mixes the large diagonal-antisymmetric amplitude `a_{nt}` into the scalar
  fed into `X/F/U/V`;
- quadratic operator terms then scale like `\rho_{nt}\rho_{n't}` instead of
  the required componentwise product.

That is exactly why the parallel `U_{nn'}/V_{nn'}` sector blows up.

### Numerical discriminator

For `C2H2`:

- `(n=0,t=0)`:
  - `s_{nt} = 5.710266677840147e-4`
  - `principal_direction = 2.162362193843213e-1`
  - ratio `~ 379`

- `(n=1,t=1)`:
  - `s_{nt} = 1.1932249053611407e-1`
  - `principal_direction = 1.0`
  - ratio `~ 8.38`

So `principal_direction` is not a mild convention change. It injects the wrong
scale into the operator sector.

## 4. Why `pair_offdiag` Is The Correct Working Choice

The off-diagonal reduction is

\[
\texttt{pair\_offdiag}(M_{nt}) = s_{nt} = \frac{M_{12}+M_{21}}{2}.
\]

This is not invariant under an arbitrary rotation of the degenerate basis.
However, once the pair basis has been canonicalized, it becomes the component
matching the scanned component notation.

That is exactly the role of the current rotational-derivative/canonical-pair
machinery:

- first build a canonical basis for each degenerate pair,
- then evaluate the Coriolis block in that canonical basis,
- then read the off-diagonal component in that canonical basis.

Under this condition, `pair_offdiag` is not an arbitrary projection. It is the
component candidate consistent with the notation `\zeta^x_{n,tb}`.

## 5. Consequence For `uv_parallel`

With `principal_direction`, the off-diagonal `U_{nn'}/V_{nn'}` contribution is
catastrophic:

- `C2H2`: about `1.6\times 10^{-3}`
- `HCN`: about `1.56\times 10^{-7}`

With `pair_offdiag`, the same sector drops to the sane range:

- `C2H2`: about `10^{-8}`
- `HCN`: about `10^{-12}\dots 10^{-14}`

So the correct conclusion is:

1. the parallel `uv` term itself is not the problem;
2. the problem was the wrong scalar reduction fed into the operator sector;
3. `principal_direction` must be excluded from `X/F/U/V`;
4. in the present canonical linear branch, `pair_offdiag` is the correct
   working reduction for the operator sector.

## 6. What Is Proved, And What Is Not

What is proved here:

- `principal_direction` is incompatible with a component-type Aliev scalar in
  the operator sector;
- the `uv_parallel` blowup follows directly from that incompatibility;
- `pair_offdiag` is the unique correct choice **within the present canonical
  basis construction**, because it isolates the component rather than the norm.

What is not yet proved in an absolute literature sense:

- that the current canonical pair basis is exactly the same physical basis used
  in the original Aliev derivation.

So this note gives a complete proof for the current implementation framework,
and a strong physical argument for the chosen working branch. The only residual
publication-level task is to connect the present canonical pair basis
explicitly to the original Aliev basis convention.
