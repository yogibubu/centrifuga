#!/usr/bin/env python3
"""Exact rotational-derivative dependency audit for the Paper1 octic block.

Goal:
    determine whether mu2 can appear in the octic branch, assuming the already
    established sextic result that H06 depends on mu1 but not mu2.

This is a symbolic dependency graph, not a numerical calculation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dep:
    rot: frozenset[str]
    pot: frozenset[str]
    harm: frozenset[str]

    def union(self, *others: "Dep") -> "Dep":
        rot = set(self.rot)
        pot = set(self.pot)
        harm = set(self.harm)
        for other in others:
            rot |= set(other.rot)
            pot |= set(other.pot)
            harm |= set(other.harm)
        return Dep(frozenset(rot), frozenset(pot), frozenset(harm))


ZERO = Dep(frozenset(), frozenset(), frozenset())


def paper1_rotderivative_dependency_map() -> dict[str, Dep]:
    """Return the exact dependency graph used in the octic audit."""
    out: dict[str, Dep] = {}

    # Primitive families
    out["mu1"] = Dep(frozenset({"mu1"}), frozenset(), frozenset())
    out["mu2"] = Dep(frozenset({"mu2"}), frozenset(), frozenset())
    out["phi3"] = Dep(frozenset(), frozenset({"phi3"}), frozenset())
    out["phi4"] = Dep(frozenset(), frozenset({"phi4"}), frozenset())
    out["zeta"] = Dep(frozenset(), frozenset(), frozenset({"zeta"}))
    out["omega"] = Dep(frozenset(), frozenset(), frozenset({"omega"}))
    out["B"] = Dep(frozenset(), frozenset(), frozenset({"B"}))

    # Paper1/Table I side used in the previous linear audit
    out["C"] = out["mu1"]
    out["R_k"] = out["C"]
    out["R_klm"] = out["C"]

    # Eq. (101): visible sectors are C^2 + k4 + zeta^2
    out["R_kl"] = ZERO.union(out["C"], out["phi4"], out["zeta"], out["omega"], out["B"])
    out["R'_kl"] = ZERO.union(out["R_kl"], out["phi4"], out["R_k"], out["omega"])
    out["R_tilde_k"] = out["R_k"]

    # Table IV / Table V auxiliaries
    out["X"] = ZERO.union(out["R_kl"], out["R'_kl"], out["R_tilde_k"], out["H02"] if "H02" in out else ZERO, out["omega"])
    out["U"] = ZERO.union(out["R_kl"], out["R'_kl"], out["R_tilde_k"], out["X"], out["omega"])
    out["V"] = ZERO.union(out["R_kl"], out["R'_kl"], out["R_tilde_k"], out["X"], out["omega"])

    # Pure rotational harmonic blocks
    out["H02"] = ZERO.union(out["B"])
    out["H04"] = ZERO.union(out["mu1"], out["omega"])

    # Table V notation block
    out["E"] = ZERO.union(out["R'_kl"], out["R_tilde_k"], out["X"], out["H02"])
    out["F"] = ZERO.union(out["R'_kl"], out["R_tilde_k"], out["X"], out["H02"])

    # Sextic theorem: no mu2 survives in H06
    out["H06"] = ZERO.union(out["mu1"], out["phi3"], out["omega"], out["B"])

    # Odd pure-vibrational generators: no rotational-derivative tensors
    out["S03"] = ZERO.union(out["phi3"], out["omega"])
    out["S05"] = ZERO.union(out["phi3"], out["phi4"], out["omega"])
    out["S07"] = ZERO.union(out["phi3"], out["phi4"], out["omega"])

    # Bare H08 from visible Table V structure
    out["H08_bare"] = ZERO.union(
        out["R_k"], out["R_klm"], out["R'_kl"], out["E"], out["F"], out["U"], out["V"], out["phi3"], out["phi4"], out["zeta"], out["omega"], out["B"]
    )

    # Eq. (99) completion
    out["term_H08"] = out["H08_bare"]
    out["term_S03S03S03H02"] = ZERO.union(out["S03"], out["H02"])
    out["term_S03S03H04"] = ZERO.union(out["S03"], out["H04"])
    out["term_S03H06"] = ZERO.union(out["S03"], out["H06"])
    out["term_S05H04"] = ZERO.union(out["S05"], out["H04"])
    out["term_S05S03H02"] = ZERO.union(out["S05"], out["S03"], out["H02"])
    out["term_S07H02"] = ZERO.union(out["S07"], out["H02"])
    out["tilde_H08"] = ZERO.union(
        out["term_H08"],
        out["term_S03S03S03H02"],
        out["term_S03S03H04"],
        out["term_S03H06"],
        out["term_S05H04"],
        out["term_S05S03H02"],
        out["term_S07H02"],
    )
    return out


def paper1_octic_mu2_verdict() -> dict[str, object]:
    """Return the exact mu2 verdict under the sextic no-mu2 theorem."""
    dep = paper1_rotderivative_dependency_map()
    source_terms = (
        "term_H08",
        "term_S03S03S03H02",
        "term_S03S03H04",
        "term_S03H06",
        "term_S05H04",
        "term_S05S03H02",
        "term_S07H02",
    )
    return {
        "source_terms": source_terms,
        "mu2_in_each_term": {term: ("mu2" in dep[term].rot) for term in source_terms},
        "mu2_in_tilde_H08": ("mu2" in dep["tilde_H08"].rot),
        "rotational_derivative_support_tilde_H08": tuple(sorted(dep["tilde_H08"].rot)),
        "potential_support_tilde_H08": tuple(sorted(dep["tilde_H08"].pot)),
        "harmonic_support_tilde_H08": tuple(sorted(dep["tilde_H08"].harm)),
        "assumption": "H06 depends on mu1 but not mu2",
    }


def main() -> None:
    out = paper1_octic_mu2_verdict()
    print(out)


if __name__ == "__main__":
    main()
    out["R_klm"] = out["C"]
