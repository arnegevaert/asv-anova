"""
This script trains a Partial Dependence Decomposition on the background set
and saves it to disk.
"""
import time
import argparse
import os
import joblib
from pddshapley import PartialDependenceDecomposition
from pddshapley.sampling import IndependentConditioningMethod, RandomSubsampleCollocation
from util import DATASETS
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="adult")
    parser.add_argument("-o", "--output-dir", type=str, default="data")
    parser.add_argument("-s", "--max-size", type=int, default=4)
    args = parser.parse_args()

    data_dir = os.path.join(args.output_dir, args.dataset)
    model = joblib.load(os.path.join(data_dir, "model.pkl"))
    background_set = joblib.load(os.path.join(data_dir, "background_set.pkl"))

    model_fn = model.predict_proba \
        if DATASETS[args.dataset]["type"] == "classification" else model.predict

    pdd = PartialDependenceDecomposition(
        model_fn,
        RandomSubsampleCollocation(),
        IndependentConditioningMethod(background_set.values),
        estimator_type="tree",
    )
    start_t = time.time()
    pdd.fit(background_set, max_size=args.max_size)
    end_t = time.time()

    joblib.dump(pdd, os.path.join(data_dir, f"pdd-{args.max_size}.pkl"))
    with open(os.path.join(data_dir, f"pdd-{args.max_size}_meta.json"), "w") as f:
        json.dump({"runtime": end_t - start_t}, f)
