"""
This script plots the results of the experiments.
"""
import argparse
import json
import os
import joblib
from sklearn import metrics
import numpy as np
from scipy.stats import spearmanr
from matplotlib import pyplot as plt


def get_results(data_dir, dataset):
    indices = sorted([
        int(filename.split("_")[1].split(".")[0])
        for filename in os.listdir(
            os.path.join(data_dir, dataset, "results", "sampling"))
            if filename.endswith(".pkl")])

    symmetric_sampling_results = joblib.load(
        os.path.join(data_dir, dataset, "results", "sampling", "values_0.pkl")
    )
    asymmetric_sampling_results = joblib.load(
        os.path.join(data_dir, dataset, "results", "sampling", 
                        f"values_{indices[-1]}.pkl")
    )
    symmetric_pdd_results = joblib.load(
        os.path.join(data_dir, dataset, "results", "pdd", "values_0.pkl")
    )
    asymmetric_pdd_results = joblib.load(
        os.path.join(data_dir, dataset, "results", "pdd",
                        f"values_{indices[-1]}.pkl")
    )
    return symmetric_sampling_results, asymmetric_sampling_results, \
            symmetric_pdd_results, asymmetric_pdd_results


def score_barplot(scores, ylabel, output_filename):
    fig, ax = plt.subplots(layout="constrained")
    width=0.35
    rects = ax.bar(np.arange(len(scores)) - width/2, [scores[ds][0] for ds in datasets],
                   width=width, label="Symmetric")
    ax.bar_label(rects, padding=3, fmt='{:.2f}')
    
    rects = ax.bar(np.arange(len(scores)) + width/2, [scores[ds][1] for ds in datasets],
                   width=width, label="Asymmetric")
    ax.bar_label(rects, padding=3, fmt='{:.2f}')

    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", fancybox=True,
              shadow=False, bbox_to_anchor=(0.5, 1.1), ncols=2)
    plt.savefig(output_filename)


def plot_r2(data_dir, datasets, output_dir):
    r2_scores = {}
    for ds in datasets:
        symmetric_sampling_results, asymmetric_sampling_results, \
            symmetric_pdd_results, asymmetric_pdd_results = get_results(
                data_dir, ds)
        avg_r2_symm = np.average(
            [metrics.r2_score(symmetric_sampling_results[..., j].flatten(),
                                symmetric_pdd_results[..., j].flatten())
                 for j in range(symmetric_sampling_results.shape[-1])])
        avg_r2_asymm = np.average(
            [metrics.r2_score(asymmetric_sampling_results[..., j].flatten(),
                                asymmetric_pdd_results[..., j].flatten())
                 for j in range(asymmetric_sampling_results.shape[-1])])
        r2_scores[ds] = (avg_r2_symm, avg_r2_asymm)
    score_barplot(r2_scores, "R2 score", os.path.join(output_dir, "r2_scores.png"))

def plot_corr(data_dir, datasets, output_dir):
    corrs = {}
    for ds in datasets:
        symmetric_sampling_results, asymmetric_sampling_results, \
            symmetric_pdd_results, asymmetric_pdd_results = get_results(
                data_dir, ds)
        avg_corr_symm = np.average(
            [[spearmanr(symmetric_sampling_results[j, :, k],
                        symmetric_pdd_results[j, :, k])[0]
                 for j in range(symmetric_sampling_results.shape[0])]
                 for k in range(symmetric_sampling_results.shape[-1])])
        avg_corr_asymm = np.average(
            [[spearmanr(asymmetric_sampling_results[j, :, k],
                        asymmetric_pdd_results[j, :, k])[0]
                 for j in range(asymmetric_sampling_results.shape[0])]
                 for k in range(asymmetric_sampling_results.shape[-1])])
        corrs[ds] = (avg_corr_symm, avg_corr_asymm)
    score_barplot(corrs, "Spearman correlation", os.path.join(output_dir, "corrs.png"))


def plot_train_time(data_dir, datasets, output_dir):
    times = {}
    for ds in datasets:
        with open(os.path.join(data_dir, ds, "pdd_meta.json")) as json_file:
            meta = json.load(json_file)
            times[ds] = meta["runtime"]
    
    fig, ax = plt.subplots(layout="constrained")
    width=0.35
    sorted_datasets = sorted(datasets, key=lambda x: -times[x])
    rects = ax.bar(np.arange(len(times)), [times[ds] for ds in sorted_datasets],
                   width=width)
    ax.bar_label(rects, padding=3, fmt='{:.2f}')

    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(sorted_datasets)
    ax.set_ylabel("Training time (s)")
    plt.savefig(os.path.join(output_dir, "train_time.png"))



def plot_inference_speed(data_dir, datasets, output_dir):
    speeds = {}
    for ds in datasets:
        exp_set = joblib.load(os.path.join(data_dir, ds, "exp_set.pkl"))
        n_samples = exp_set.shape[0]
        indices = sorted([
            int(filename.split("_")[1].split(".")[0])
            for filename in os.listdir(
                os.path.join(data_dir, ds, "results", "sampling"))
                if filename.endswith(".pkl")])
        ds_speeds = []
        with open(os.path.join(data_dir, ds, "results", "pdd",
                               "meta_0.json")) as json_file:
            meta = json.load(json_file)
            ds_speeds.append(meta["runtime"] / n_samples)
        with open(os.path.join(data_dir, ds, "results", "pdd", 
                               f"meta_{indices[-1]}.json")) as json_file:
            meta = json.load(json_file)
            ds_speeds.append(meta["runtime"] / n_samples)
        with open(os.path.join(data_dir, ds, "results", "sampling",
                               "meta_0.json")) as json_file:
            meta = json.load(json_file)
            ds_speeds.append(meta["runtime"] / n_samples)
        with open(os.path.join(data_dir, ds, "results", "sampling", 
                               f"meta_{indices[-1]}.json")) as json_file:
            meta = json.load(json_file)
            ds_speeds.append(meta["runtime"] / n_samples)
        speeds[ds] = ds_speeds
    
    fig, ax = plt.subplots(layout="constrained")
    width=0.2
    multiplier = 0
    x = np.arange(len(speeds))
    for i, label in enumerate(["Symmetric-PDD", "Asymmetric-PDD", 
                               "Symmetric-Sampling", "Asymmetric-Sampling"]):
        rects = ax.bar(x + multiplier * width,
                       [speeds[ds][i] for ds in datasets],
                       width=width, label=label)
        multiplier += 1
    ax.set_yscale("log")
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Inference speed (s/sample)")
    ax.legend(loc="upper center", fancybox=True,
                shadow=False, bbox_to_anchor=(0.5, 1.15), ncols=2)
    plt.savefig(os.path.join(output_dir, "inference_speed.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, default="data/maxsize_3")
    parser.add_argument("-o", "--output-dir", type=str, default="plots")
    args = parser.parse_args()

    datasets = os.listdir(args.data_dir)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    plot_r2(args.data_dir, datasets, args.output_dir)
    plot_corr(args.data_dir, datasets, args.output_dir)
    plot_train_time(args.data_dir, datasets, args.output_dir)
    plot_inference_speed(args.data_dir, datasets, args.output_dir)