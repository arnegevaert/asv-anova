"""
This script samples Asymmetric Shapley Values (ASV) using the ASVExplainer.
It does this for varying partial orders: for i = 0, ..., d//2, we sample ASV
where the first i features should precede the other d-i features.
"""
import os
from asv_explainer import ASVExplainer
from pddshapley.sampling import IndependentConditioningMethod
from time import time
import argparse
import joblib
from util import DATASETS
import json


def sample_asv(model, ds_type, n_outputs, X_background, X_test, partial_order):
    model_fn = model.predict_proba if ds_type == "classification" else model.predict
    explainer = ASVExplainer(model_fn,
                            IndependentConditioningMethod(X_background),
                            num_outputs=n_outputs)
    start_t = time()
    result =  explainer.explain(X_test.values, max_n_permutations=1000,
                                partial_order=partial_order)
    end_t = time()
    return result, end_t - start_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="adult")
    parser.add_argument("-o", "--output-dir", type=str, default="data")
    args = parser.parse_args()

    data_dir = os.path.join(args.output_dir, args.dataset)
    result_dir = os.path.join(data_dir, "results", "sampling")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    model = joblib.load(os.path.join(data_dir, "model.pkl"))
    background_set = joblib.load(os.path.join(data_dir, "background_set.pkl"))
    exp_set = joblib.load(os.path.join(data_dir, "exp_set.pkl"))

    for i in range(background_set.shape[1] // 2):
        print(f"{i}/{background_set.shape[1] // 2}...")
        if i == 0:
            partial_order = None
        else:
            all_columns = list(range(background_set.shape[1]))
            partial_order = [all_columns[:i], all_columns[i:]]

        values, runtime = sample_asv(model, DATASETS[args.dataset]["type"],
                                     DATASETS[args.dataset]["n_outputs"],
                                    background_set, exp_set, partial_order)
        joblib.dump(values, os.path.join(result_dir, f"values_{i}.pkl"))
        with open(os.path.join(result_dir, f"meta_{i}.json"), "w") as f:
            json.dump({"runtime": runtime, "partial_order": partial_order}, f)