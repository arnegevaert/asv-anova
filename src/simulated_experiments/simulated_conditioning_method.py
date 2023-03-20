from pddshapley.sampling import ConditioningMethod
from pddshapley.signature import FeatureSubset
from numpy import typing as npt
from typing import Callable
import numpy as np


class SimulatedConditioningMethod(ConditioningMethod):
    """
    This class implements a conditioning method
    for the specific simulated data experiments.
    ("Causal explanations of unfair discrimination" in Frye et al. 2020)
    """
    def __init__(self, **kwargs) -> None:
        self.rng = np.random.default_rng(0)
        super().__init__(None, **kwargs)
    
    def _conditional_expectation(self, feature_subset: FeatureSubset,
                                    value: npt.NDArray,
                                    model: Callable[[npt.NDArray], npt.NDArray],
                                    num_samples=100, **kwargs) -> npt.NDArray:
            if len(value) > 0:
                value = feature_subset.expand_columns(value, 3)[0, :]
            # X[:, 1] is independent of both other features
            data = np.zeros((num_samples, 3))
            if 1 not in feature_subset:
                data[:, 1] = self.rng.normal(0, 1, num_samples)
            else:
                data[:, 1] = value[1]
            
            # X[:, 0] and X[:, 2] behave symmetrically:
            # Both marginals are uniform, both conditionals are 0.8 - 0.2
            # (follows from Bayes rule)
            for i, j in [(0, 2), (2, 0)]:
                if i not in feature_subset:
                    if j in feature_subset:
                        prob = 0.8 if value[j] == 0 else 0.2
                    else:
                        prob = 0.5
                    data[:, i] = self.rng.binomial(1, prob, num_samples)
                else:
                    data[:, i] = value[i]
            return model(data)