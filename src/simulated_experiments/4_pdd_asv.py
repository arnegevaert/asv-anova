"""
This script computes Asymmetric Shapley Values using PDD-SHAP.
The experiment is analogous to 2_sample_asv.py.
"""
import numpy as np
import joblib
import argparse
import os
import time
import json
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, default="data/simulated")
    args = parser.parse_args()

    pdd_fair = joblib.load(os.path.join(args.output_dir, "pdd_fair.pkl"))
    pdd_unfair = joblib.load(os.path.join(args.output_dir, "pdd_unfair.pkl"))
    test_data = pd.read_csv(os.path.join(args.output_dir, "test_data.csv"))

    # Partial order: feature 2 (department) must come before feature 0 (gender)
    # Resulting ASVs for gender show the influence of the gender feature *after*
    # the department feature is known.
    partial_order = [[2], [0]]
    X_test = test_data[["x1", "x2", "x3"]].values
    meta = {}

    # Compute ASVs for fair labels
    start_t = time.time()
    fair_values = pdd_fair.shapley_values(X_test, partial_ordering=partial_order)
    end_t = time.time()
    meta["runtime_fair"] = end_t - start_t

    # Compute ASVs for unfair labels
    start_t = time.time()
    unfair_values = pdd_unfair.shapley_values(X_test, partial_ordering=partial_order)
    end_t = time.time()
    meta["runtime_unfair"] = end_t - start_t
    
    result_dir = os.path.join(args.output_dir, "results", "pdd")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    np.save(os.path.join(result_dir, "values_fair.npy"), fair_values)
    np.save(os.path.join(result_dir, "values_unfair.npy"), unfair_values)
    with open(os.path.join(result_dir, "meta.json"), "w") as f:
        json.dump(meta, f)