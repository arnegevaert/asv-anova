"""
This script trains a Partial Dependence Decomposition on a random subsample
of the training data and saves it to disk. Note that in the synthetic data setting,
the background set is only used for collocation points. Conditional expectations
are computed using the data generating model (SimulatedConditioningMethod).
"""
import time
import argparse
import os
import joblib
from pddshapley import PartialDependenceDecomposition
from pddshapley.sampling import RandomSubsampleCollocation
from simulated_conditioning_method import SimulatedConditioningMethod
import pandas as pd
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, default="data/simulated")
    args = parser.parse_args()

    model_fair = joblib.load(os.path.join(args.output_dir, "model_fair.pkl"))
    model_unfair = joblib.load(os.path.join(args.output_dir, "model_unfair.pkl"))
    train_data = pd.read_csv(os.path.join(args.output_dir, "train_data.csv"))
    background_set = train_data[["x1", "x2", "x3"]].sample(1000)

    meta = {}

    for model, name in ((model_fair, "fair"), (model_unfair, "unfair")):
        print("Training PDD on model:", name)
        pdd = PartialDependenceDecomposition(
            model.predict_proba,
            RandomSubsampleCollocation(),
            SimulatedConditioningMethod(),
            estimator_type="tree",
        )
        start_t = time.time()
        pdd.fit(background_set, n_jobs=1)
        end_t = time.time()

        joblib.dump(pdd, os.path.join(args.output_dir, f"pdd_{name}.pkl"))
        meta["runtime_" + name] = end_t - start_t

    with open(os.path.join(args.output_dir, "pdd_meta.json"), "w") as f:
        json.dump(meta, f)
