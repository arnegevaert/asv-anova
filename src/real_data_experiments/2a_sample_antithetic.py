"""
This script samples symmetric Shapley values using antithetic sampling
(PermutationExplainer in shap package)
"""
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from shap import PermutationExplainer
from time import time
import argparse
import joblib
from util import DATASETS
import json


def sample_antithetic(model, ds_type, n_outputs, X_background, X_test):
    model_fn = model.predict_proba if ds_type == "classification" else model.predict
    explainer = PermutationExplainer(model_fn, X_background.values)
    start_t = time()
    result =  explainer(X_test.values).values
    end_t = time()
    return result, end_t - start_t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="adult",
                        choices=DATASETS.keys(), help="Dataset to use.")
    parser.add_argument("-o", "--output-dir", type=str, default="data",
                        help="Directory to store the results in. This should"
                        "be the same as the output directory in 1_train_model.py.")
    args = parser.parse_args()

    data_dir = os.path.join(args.output_dir, args.dataset)
    result_dir = os.path.join(data_dir, "results", "antithetic")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    model = joblib.load(os.path.join(data_dir, "model.pkl"))
    background_set = joblib.load(os.path.join(data_dir, "background_set.pkl"))
    exp_set = joblib.load(os.path.join(data_dir, "exp_set.pkl"))

    values, runtime = sample_antithetic(model, DATASETS[args.dataset]["type"],
                                DATASETS[args.dataset]["n_outputs"],
                                background_set, exp_set)

    joblib.dump(values, os.path.join(result_dir, "values_antithetic.pkl"))
    with open(os.path.join(result_dir, f"meta_antithetic.json"), "w") as f:
        json.dump({"runtime": runtime}, f)
