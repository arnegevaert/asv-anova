import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="data/simulated")
    parser.add_argument("-o", "--output_dir", type=str, default="plots")
    args = parser.parse_args()

    fair_values = np.load(os.path.join(args.data_dir, "results", "sampling", "values_fair.npy"))
    unfair_values = np.load(os.path.join(args.data_dir, "results", "sampling", "values_unfair.npy"))

    fig, ax = plt.subplots(layout="constrained")
    fair_plot = ax.boxplot(np.abs(fair_values[..., 0]), positions=np.arange(3)*2 - 0.35, widths=0.6,
                           patch_artist=True, showfliers=False)
    unfair_plot = ax.boxplot(np.abs(unfair_values[..., 0]), positions=np.arange(3)*2 + 0.35, widths=0.6,
                             patch_artist=True, showfliers=False)

    # fill with colors
    colors = ['pink', 'lightblue']
    for bplot, color in zip((fair_plot, unfair_plot), colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)
    
    #use plot function to draw a small line to name the legend.
    plt.plot([], c="pink", label="Fair data")
    plt.plot([], c="lightblue", label="Unfair data")
    plt.legend()

    ax.set_ylabel("Absolute ASV")
    ax.set_xticks(np.arange(0, 6, 2), ["Gender", "Test score", "Department"])
    plt.show()