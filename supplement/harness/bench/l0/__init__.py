"""PWM-Bench L0 — Mental-Model Fidelity (the substrate lens).

Evaluates the substrate AS a mental model (Forrester; Ha & Schmidhuber): do its
anchors (selected concepts) and relations generalize the real system, rather
than memorize surface instances? See spec §3A. The headline is the
embedding-only vs embedding+graph delta on concepts that TRANSCEND perception
(identity/activity), where raw appearance is insufficient — perceptual concepts
(place) are embedding-redundant and do not test model fidelity.
"""
from .keystone import run_keystone
from .oddoneout import run_l0, run_oddoneout

__all__ = ["run_keystone", "run_l0", "run_oddoneout"]
