"""theorysim — MPTE theory-validation simulation (JBES revision add-on).

ISOLATED package. Validates the linear estimator the inferential theory describes
(Theorems 1-3) and bridges it to the nonlinear model, via experiments E1-E4.

Isolation contract: this package never modifies or overwrites the existing empirical
pipeline. It lives only under src/theorysim/, src/config/cfg_theorysim.yaml,
outputs/theorysim/, reports/theory_validation/, and tests/test_theorysim_*.py, and
writes outputs only under outputs/theorysim/. Reuse of existing code is by import or by
copying patterns, never by editing originals. See src/theorysim/README.md.
"""
