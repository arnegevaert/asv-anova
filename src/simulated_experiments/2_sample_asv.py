import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from asv_explainer import ASVExplainer
from simulated_conditioning_method import SimulatedConditioningMethod
from time import time
import argparse
import joblib
import json
import numpy as np
import pandas as pd


def sample_asv(model, X_test, partial_order):
    model_fn = model.predict_proba
    explainer = ASVExplainer(model_fn,
                            SimulatedConditioningMethod(),
                            num_outputs=2)
    start_t = time()
    result =  explainer.explain(X_test.values, max_n_permutations=1000,
                                partial_order=partial_order)
    end_t = time()
    return result, end_t - start_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, default="data/simulated",
                        help="Directory to store the results in. This should"
                        "be the same as the output directory in 1_train_model.py.")
    args = parser.parse_args()

    result_dir = os.path.join(args.output_dir, "results", "sampling")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    model_fair = joblib.load(os.path.join(args.output_dir, "model_fair.pkl"))
    model_unfair = joblib.load(os.path.join(args.output_dir, "model_unfair.pkl"))
    test_data = pd.read_csv(os.path.join(args.output_dir, "test_data.csv"))

    # Partial order: feature 2 (department) must come before feature 0 (gender)
    # Resulting ASVs for gender show the influence of the gender feature *after*
    # the department feature is known.
    partial_order = [[2], [0]]
    X_test = test_data[["x1", "x2", "x3"]]
    meta = {}

    # Compute ASVs for fair labels
    values, runtime = sample_asv(model_fair, X_test, partial_order)
    np.save(os.path.join(result_dir, "values_fair.npy"), values)
    meta["fair"] = {"runtime": runtime}

    # Compute ASVs for unfair labels
    values, runtime = sample_asv(model_unfair, X_test, partial_order)
    np.save(os.path.join(result_dir, "values_unfair.npy"), values)
    meta["unfair"] = {"runtime": runtime}

    with open(os.path.join(result_dir, "meta.json"), "w") as f:
        json.dump(meta, f)