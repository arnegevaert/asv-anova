import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="data/simulated")
    parser.add_argument("-o", "--output_dir", type=str, default="plots")
    args = parser.parse_args()

    fair_values = np.load(os.path.join(args.data_dir, "results", "pdd", "values_fair.npy"))
    unfair_values = np.load(os.path.join(args.data_dir, "results", "pdd", "values_unfair.npy"))
    test_data = pd.read_csv(os.path.join(args.data_dir, "test_data.csv"))
    fair_labels = test_data["y_fair"].astype(int).values
    unfair_labels = test_data["y_unfair"].astype(int).values

    fig, ax = plt.subplots(layout="constrained")
    fair_plot_values = fair_values[np.arange(len(fair_values)), :,  fair_labels]
    unfair_plot_values = unfair_values[np.arange(len(unfair_values)), :,  unfair_labels]

    fair_plot = ax.bar(np.arange(3)*2 - 0.35, fair_plot_values.mean(axis=0), 0.6, label="Fair data")
    unfair_plot = ax.bar(np.arange(3)*2 + 0.35, unfair_plot_values.mean(axis=0), 0.6, label="Unfair data")
    plt.legend()

    ax.set_ylabel("Global ASV")
    ax.set_xticks(np.arange(0, 6, 2), ["Gender", "Test score", "Department"])
    plt.savefig(os.path.join(args.output_dir, "fairness.png"))