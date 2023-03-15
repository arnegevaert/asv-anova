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
from joblib import Parallel, delayed


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
                 num_outputs: int) -> None:
        self.model = model
        self.conditioning_method = conditioning_method
        self.num_outputs = num_outputs

    def _explain_row(self, row: npt.NDArray, partial_order: List[List[int]],
                     incomparables: List[int], max_n_permutations: int):
        assert len(row.shape) == 1, "Row must be a 1D array"

        contributions = np.zeros((row.shape[0], self.num_outputs))
        running_avg = np.zeros_like(contributions)
        consecutive_close = 0
        global_expectation = np.average(
                self.conditioning_method.conditional_expectation(
                    FeatureSubset(), np.array([]), self.model),
                axis=0)
        for p in range(max_n_permutations):
            linear_extension = _random_linear_extension(partial_order, row.shape[0], incomparables)
            avg_before = global_expectation
            for i, feature in enumerate(linear_extension):
                after = FeatureSubset(*linear_extension[:i + 1])
                row_after = after.get_columns(row.reshape(1, -1)).reshape(-1)

                avg_after = np.average(
                        self.conditioning_method.conditional_expectation(
                            FeatureSubset(*after), row_after, self.model),
                        axis=0)
                contributions[feature, ...] += avg_after - avg_before
                avg_before = avg_after
            prev_running_avg = running_avg.copy()
            running_avg = contributions / (p + 1)
            if np.allclose(prev_running_avg, running_avg, atol=1e-4, rtol=1e-3):
                consecutive_close += 1
            else:
                consecutive_close = 0
            if consecutive_close == 10:
                return running_avg
        return contributions / max_n_permutations

    def explain(self, data: npt.NDArray, max_n_permutations: int,
                partial_order: List[List[int]] = None):

        if partial_order is None:
            partial_order = [list(range(data.shape[1]))]

        incomparables = np.ones(data.shape[1])
        for group in partial_order:
            for i in group:
                incomparables[i] = 0

        res = Parallel(n_jobs=-1)(
            delayed(self._explain_row)(row, partial_order, 
                                       np.where(incomparables == 1)[0],
                                       max_n_permutations)
                                       for row in tqdm(data))
        return np.array(res)