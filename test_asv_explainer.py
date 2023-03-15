from typing import cast
from pddshapley.sampling import IndependentConditioningMethod,\
    GaussianConditioningMethod, KernelConditioningMethod
from experiments.util import eval
import numpy as np
from experiments.synthetic.multilinear_polynomial import RandomMultilinearPolynomial
import pandas as pd
from sklearn.model_selection import train_test_split
from asv_explainer import ASVExplainer


if __name__ == "__main__":
    # Generate input data
    num_features = 3
    np.random.seed(0)
    #num_features = 2
    mean = np.zeros(num_features)
    cov = np.diag(np.ones(num_features))
    X = np.random.multivariate_normal(mean, cov, size=1000).astype(np.float32)

    # Create model and compute ground truth Shapley values
    model = RandomMultilinearPolynomial(num_features, [-1, -1, -1])
    print(model)
    y = model(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    true_values = np.expand_dims(model.shapley_values(X_test), -1)

    num_permutations = 100

    # Compute ASV-SHAP values
    asv_explainer = ASVExplainer(model, IndependentConditioningMethod(X_train), 1)
    asv_values = asv_explainer.explain(X_test, max_n_permutations=num_permutations)
    eval.print_metrics(asv_values, true_values, "ASV", "Ground truth")
