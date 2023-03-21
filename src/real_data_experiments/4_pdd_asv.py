"""
This script computes Asymmetric Shapley Values using PDD-SHAP.
The experiment is analogous to 2_sample_asv.py.
"""
import joblib
import argparse
import os
import time
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="adult")
    parser.add_argument("-o", "--output-dir", type=str, default="data")
    args = parser.parse_args()

    data_dir = os.path.join(args.output_dir, args.dataset)
    pdd = joblib.load(os.path.join(data_dir, "pdd.pkl"))
    exp_set = joblib.load(os.path.join(data_dir, "exp_set.pkl"))
    
    result_dir = os.path.join(data_dir, "results", "pdd")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    columns = exp_set.columns
    for i in range(len(columns) // 2 + 1):
        print(f"{i}/{len(columns) // 2 + 1}...")
        if i == 0:
            partial_order = None
        else:
            all_columns = list(range(len(columns)))
            partial_order = [all_columns[:i], all_columns[i:]]
        
        start_t = time.time()
        values = pdd.shapley_values(exp_set, partial_ordering=partial_order, project=True)
        end_t = time.time()

        joblib.dump(values, os.path.join(result_dir, f"values_{i}.pkl"))
        with open(os.path.join(result_dir, f"meta_{i}.json"), "w") as f:
            json.dump({"runtime": end_t - start_t, "partial_order": partial_order}, f)