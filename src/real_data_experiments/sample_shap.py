"""
This script compares the SHAP values produced by PDD-SHAP, my own sampling
implementation, and shap.PermutationSampler. This comparison only works for
empty partial orders, i.e. when the partial order is None. It serves as a
sanity check for the implementation of PDD-SHAP and my own sampling implemenation.
"""
import os
import numpy as np
from sklearn import metrics
import argparse
import joblib
from shap import PermutationExplainer
from scipy.stats import spearmanr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="adult")
    parser.add_argument("-o", "--output-dir", type=str, default="data")
    args = parser.parse_args()

    data_dir = os.path.join(args.output_dir, args.dataset)

    model = joblib.load(os.path.join(data_dir, "model.pkl"))
    background_set = joblib.load(os.path.join(data_dir, "background_set.pkl"))
    exp_set = joblib.load(os.path.join(data_dir, "exp_set.pkl"))

    explainer = PermutationExplainer(model.predict_proba, background_set.values)
    shap_values = explainer(exp_set.values).values

    pdd_values = joblib.load(os.path.join(data_dir, "results", "pdd", "values_0.pkl"))
    sampling_values = joblib.load(os.path.join(data_dir, "results", "sampling", "values_0.pkl"))

    corr_shap_sampling = []
    corr_shap_pdd = []
    corr_pdd_sampling = []
    for i in range(sampling_values.shape[0]):
        for j in range(sampling_values.shape[-1]):
            corr_shap_sampling.append(spearmanr(shap_values[i, :, j], sampling_values[i, :, j])[0])
            corr_shap_pdd.append(spearmanr(shap_values[i, :, j], pdd_values[i, :, j])[0])
            corr_pdd_sampling.append(spearmanr(pdd_values[i, :, j], sampling_values[i, :, j])[0])

    print(f"Correlation shap-sampling: {np.average(corr_shap_sampling):.4f}")
    print(f"Correlation shap-pdd: {np.average(corr_shap_pdd):.4f}")
    print(f"Correlation pdd-sampling: {np.average(corr_pdd_sampling):.4f}")

    avg_r2 = np.average(
        [metrics.r2_score(shap_values[..., j], sampling_values[..., j])
            for j in range(sampling_values.shape[-1])])
    print(f"R2 sampling: {avg_r2:.4f}")
    
    avg_r2 = np.average(
        [metrics.r2_score(shap_values[..., j], pdd_values[..., j])
            for j in range(sampling_values.shape[-1])])
    print(f"R2 PDD: {avg_r2:.4f}")
    
    avg_r2 = np.average(
        [metrics.r2_score(sampling_values[..., j], pdd_values[..., j])
            for j in range(sampling_values.shape[-1])])
    print(f"R2 PDD-sampling: {avg_r2:.4f}")