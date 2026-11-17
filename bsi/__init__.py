"""bsi — Belief-based System Identification.

LLMs propose hypotheses; code does the inference (posterior, EIG); explicit rules decide.
"""
from .hypotheses import BackoffDirichletOther, DirichletOther, FixedDist, FunctionHypothesis, Hypothesis, Observation
from .posterior import Posterior

__all__ = ["Hypothesis", "Observation", "DirichletOther", "BackoffDirichletOther", "FixedDist", "FunctionHypothesis", "Posterior"]
