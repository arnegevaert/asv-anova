from pddshapley.sampling import ConditioningMethod
from pddshapley.signature import FeatureSubset
from numpy import typing as npt
import numpy as np
from typing import Callable


def _sample_gaussian_conditional(mean: npt.NDArray, cov: npt.NDArray,
                                 feature_subset: FeatureSubset,
                                 value: npt.NDArray, num_samples: int) -> npt.NDArray:
    """
    Sample from the conditional distribution of a Gaussian distribution
    given a subset of features.

    Parameters
    ----------
    mean : npt.NDArray
        Mean of the Gaussian distribution.
    cov : npt.NDArray
        Covariance matrix of the Gaussian distribution.
    feature_subset : FeatureSubset
        Subset of features that are given.
    value : npt.NDArray
        Values of the given features.
    num_samples : int
        Number of samples to draw.

    Returns
    -------
    npt.NDArray
        Sampled values.
    """
    # Compute conditional mean and covariance
    # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    # for details
    if np.linalg.cond(cov) > 1 / np.finfo(cov.dtype).eps:
        # If matrix is singular, add small value to diagonal
        cov += np.eye(cov.shape[0]) * 1e-6
    x = feature_subset.features
    y = np.setdiff1d(np.arange(mean.shape[0]), x)
    cov_xx = cov[x, :][:, x]
    cov_xy = cov[x, :][:, y]
    cov_yx = cov[y, :][:, x]
    cov_yy = cov[y, :][:, y]
    mean_x = mean[x]
    mean_y = mean[y]
    cov_cond = cov_yy - cov_yx @ np.linalg.inv(cov_xx) @ cov_xy
    mean_cond = mean_y + cov_yx @ np.linalg.inv(cov_xx) @ (value - mean_x)

    # Sample from conditional distribution
    return np.random.multivariate_normal(mean_cond, cov_cond, num_samples)


class CategoricalGaussianConditioningMethod(ConditioningMethod):
    def __init__(self, train_data: npt.NDArray,
                 categorical_features: FeatureSubset,
                 **kwargs) -> None:
        super().__init__(train_data, **kwargs)
        self.categorical_features = categorical_features
        self.numerical_features = FeatureSubset(*np.setdiff1d(
            np.arange(train_data.shape[1]), categorical_features.features))
        self.train_data = train_data
        self.mean = np.mean(train_data[:, self.numerical_features.features].astype(float), axis=0)
        self.cov = np.cov(train_data[:, self.numerical_features.features].astype(float), rowvar=False)
    
    def _conditional_expectation(self, feature_subset: FeatureSubset,
                                 value: npt.NDArray,
                                 model: Callable[[npt.NDArray], npt.NDArray],
                                 num_samples=100, **kwargs) -> npt.NDArray:
        if len(feature_subset) < self.train_data.shape[1]:
            # Filter on categorical features
            categorical_features = self.categorical_features.intersection(
                feature_subset).features
            if len(categorical_features) > 0:
                # Extract the categorical columns from value
                categ_rel_idx = np.where(np.isin(feature_subset.features, categorical_features))[0]
                subset_idx = self.train_data[:, categorical_features] ==\
                    value[categ_rel_idx].reshape(1, -1)
                if np.any(np.all(subset_idx, axis=1)):
                    conditioned_data = self.train_data[np.all(subset_idx, axis=1), :]
                else:
                    conditioned_data = self.train_data
            else:
                conditioned_data = self.train_data

            numerical_features = self.numerical_features.intersection(
                feature_subset)
            if len(numerical_features) > 0:
                # Sample given categorical features
                result = conditioned_data[np.random.choice(
                    conditioned_data.shape[0], num_samples, replace=True), :]

                # Fit Gaussian on all numerical features, given the categorical
                """
                numerical_conditioned = conditioned_data[:, self.numerical_features.features].astype(float)
                mean = np.mean(numerical_conditioned, axis=0)
                cov = np.cov(numerical_conditioned, rowvar=False)
                """

                # For numerical features that are not given,
                # sample from conditional distribution
                value_num = value[np.where(np.isin(
                    feature_subset.features, numerical_features.features))[0]]\
                        .astype(float)
                non_given_numerical = np.setdiff1d(
                    self.numerical_features.features, feature_subset.features
                )
                if len(non_given_numerical) > 0:
                    numerical_features_rel = FeatureSubset(*np.where(np.isin(
                        self.numerical_features.features, numerical_features.features))[0])
                    result[:, non_given_numerical] = _sample_gaussian_conditional(
                        self.mean, self.cov, numerical_features_rel, value_num, num_samples)
                
                # For numerical features that are given,
                # fix the given value
                result[:, numerical_features.features] = value_num
            else:
                return model(conditioned_data[np.random.choice(
                    conditioned_data.shape[0], num_samples, replace=True), :])
            return model(result)
        return model(value.reshape(1, -1))