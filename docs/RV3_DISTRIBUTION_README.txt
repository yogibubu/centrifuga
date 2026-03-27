RV3 Distribution

RV3 = Ro-Vibrational Van Vleck

Current live scope:
- geometry from xyz / Gaussian log / Gaussian fchk
- Hessian from Gaussian fchk or text Cartesian Hessian via xyz+hessian
- harmonic model, point group, rotational constants, normal-mode symmetry
- order-2 quartic route
- order-3 cubic / sextic / H22 diagnostics
- Gaussian alpha parser
- alpha from harmonic model + semi-diagonal cubic input
- order-4 linear branch through the existing linear Aliev backend
- integrated vibro-rotational report
- export of JSON / report / CSV

Not implemented yet:
- Der2 / Der3 / Der4 formal readers
- general non-linear order-4 branch
- GVPT2 vibrational and ro-vibrational layers

Entry points:
- RV3.app
- or CLI: python RV3.py --help

