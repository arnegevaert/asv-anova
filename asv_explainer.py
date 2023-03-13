"""
This module implements Asymmetric Shapley values using permutation sampling.
Reference: Frye et al. 2020. Asymmetric Shapley Values: incorporating causal 
            knowledge into model-agnostic explainability. 
"""
import numpy as np
from typing import List, Callable, Dict
from pddshapley.sampling.conditioning_method import ConditioningMethod
from pddshapley.signature import FeatureSubset
from numpy import typing as npt
from tqdm import tqdm
from joblib import Parallel


def _random_linear_extension(partial_order: List[List[int]],
                             num_features: int,
                             incomparables: List[int]) -> List[int]:
    result = np.zeros(num_features, dtype=int)
    shuffled_indices = np.arange(num_features)
    np.random.shuffle(shuffled_indices)

    incomparable_indices = shuffled_indices[:len(incomparables)]
    comparable_indices = shuffled_indices[len(incomparables):]

    result[incomparable_indices] = np.random.permutation(incomparables)

    comparables: List[int] = []
    for group in partial_order:
        comparables.extend(np.random.permutation(group))
    
    result[comparable_indices] = comparables
    return result.tolist()


class ASVExplainer:
    def __init__(self, model: Callable[[npt.NDArray], npt.NDArray],
                 conditioning_method: ConditioningMethod,
                 num_outputs: int,
                 memo=True) -> None:
        self.model = model
        self.conditioning_method = conditioning_method
        self.num_outputs = num_outputs
        self.memo = memo

    def _explain_row(self, row: npt.NDArray, partial_order: List[List[int]],
                     incomparables: List[int], num_permutations: int = 100,
                     num_samples: int = 100):
        assert len(row.shape) == 1, "Row must be a 1D array"

        memo: Dict[FeatureSubset, npt.NDArray] = {}
        contributions = np.zeros((row.shape[0], self.num_outputs))
        for _ in range(num_permutations):
            linear_extension = _random_linear_extension(partial_order, row.shape[0], incomparables)
            for i, feature in enumerate(linear_extension):
                before = FeatureSubset(*linear_extension[:i])
                after = FeatureSubset(*linear_extension[:i + 1])
                row_before = before.get_columns(row.reshape(1, -1)).reshape(-1)
                row_after = after.get_columns(row.reshape(1, -1)).reshape(-1)

                if self.memo:
                    if FeatureSubset(*before) not in memo.keys():
                        memo[FeatureSubset(*before)] = np.average(
                            self.conditioning_method.conditional_expectation(
                            FeatureSubset(*before), row_before, self.model,
                            num_samples=num_samples), axis=0)
                    if FeatureSubset(*after) not in memo.keys():
                        memo[FeatureSubset(*after)] = np.average(
                            self.conditioning_method.conditional_expectation(
                            FeatureSubset(*after), row_after, self.model,
                            num_samples=num_samples), axis=0)
                    marginal_contribution = memo[FeatureSubset(*after)] - memo[FeatureSubset(*before)]
                else:
                    avg_before = np.average(
                        self.conditioning_method.conditional_expectation(
                            FeatureSubset(*before), row_before, self.model,
                            num_samples=num_samples),
                        axis=0)
                    avg_after = np.average(
                        self.conditioning_method.conditional_expectation(
                            FeatureSubset(*after), row_after, self.model,
                            num_samples=num_samples),
                        axis=0)
                    marginal_contribution = avg_after - avg_before
                contributions[feature, ...] += marginal_contribution
        return contributions / num_permutations

    def explain(self, data: npt.NDArray, num_permutations: int = 100,
                num_samples: int = 100,
                partial_order: List[List[int]] = None):

        if partial_order is None:
            partial_order = [list(range(data.shape[1]))]

        incomparables = np.ones(data.shape[1])
        for group in partial_order:
            for i in group:
                incomparables[i] = 0

        return np.array([self._explain_row(row, partial_order,
                                           np.where(incomparables == 1)[0], 
                                           num_permutations, num_samples)
                         for row in tqdm(data)])
            