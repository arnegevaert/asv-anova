"""
This script plots the results of the experiments.
"""
import argparse
import os
import joblib
from sklearn import metrics
import numpy as np
from scipy.stats import spearmanr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="adult")
    parser.add_argument("-o", "--data-dir", type=str, default="data")
    args = parser.parse_args()

    sampling_results_dir = os.path.join(args.data_dir, args.dataset, "results", "sampling")
    pdd_results_dir = os.path.join(args.data_dir, args.dataset, "results", "pdd")

    for filename in sorted(os.listdir(sampling_results_dir)):
        if filename.endswith(".pkl"):
            sampling_values = joblib.load(os.path.join(sampling_results_dir, filename))
            pdd_values = joblib.load(os.path.join(pdd_results_dir, filename))

            i = int(filename.split("_")[1].split(".")[0])

            avg_r2 = np.average(
                [metrics.r2_score(sampling_values[..., j].flatten(), pdd_values[..., j].flatten())
                 for j in range(sampling_values.shape[-1])])

            corrs = []
            for j in range(sampling_values.shape[0]):
                for k in range(sampling_values.shape[-1]):
                    corrs.append(spearmanr(pdd_values[j, :, k], sampling_values[j, :, k])[0])
            avg_corr = np.average(corrs)

            print(f"{i}: R2:{avg_r2:.4f} Corr:{avg_corr:.4f}")
