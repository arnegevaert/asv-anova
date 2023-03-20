import argparse
import pandas as pd
import os
import numpy as np
from numpy.random import default_rng
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import joblib


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_fair(x2, x3, rng):
    probs = sigmoid(x2 + 2*(1-x3) - 1)
    return rng.binomial(1, probs)


def generate_unfair(x1, x2, x3, rng):
    x4 = np.zeros(x1.shape)
    x4[x1 == 0] = rng.binomial(1, 1/3, size=(x1 == 0).sum())
    x4[x1 == 1] = rng.binomial(1, 2/3, size=(x1 == 1).sum())
    probs = sigmoid(x2 + 2*(1-x3) + 2*x4 - 2)
    return rng.binomial(1, probs)


def train_model(X_train, y_train, X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(10, 10), validation_fraction=0.25,
                          early_stopping=True)
    model.fit(X_train, y_train)
    print(f"Train balanced accuracy: "
          f"{balanced_accuracy_score(y_train, model.predict(X_train)):.3f}")
    print(f"Test balanced accuracy: "
          f"{balanced_accuracy_score(y_test, model.predict(X_test)):.3f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", type=int, default=10000)
    parser.add_argument("-o", "--output-dir", type=str, default="data/simulated")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate and save data
    rng = default_rng()
    x1 = rng.binomial(1, 0.5, size=args.num_samples)
    x2 = rng.normal(0, 1, size=args.num_samples)
    x3 = np.zeros(args.num_samples)
    x3[x1 == 0] = rng.binomial(1, 0.8, size=(x1 == 0).sum())
    x3[x1 == 1] = rng.binomial(1, 0.2, size=(x1 == 1).sum())
    y_fair = generate_fair(x2, x3, rng)
    y_unfair = generate_unfair(x1, x2, x3, rng)

    data = np.stack([x1, x2, x3, y_fair, y_unfair], axis=1)
    data = pd.DataFrame(data, columns=["x1", "x2", "x3", "y_fair", "y_unfair"])
    train_data, test_data = train_test_split(data, test_size=0.25)
    train_data.to_csv(os.path.join(args.output_dir, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(args.output_dir, "test_data.csv"), index=False)


    # Train and save models
    print("Fair model")
    model_fair = train_model(train_data[["x1", "x2", "x3"]].values,
                             train_data["y_fair"].values,
                             test_data[["x1", "x2", "x3"]].values,
                             test_data["y_fair"].values)
    joblib.dump(model_fair, os.path.join(args.output_dir, f"model_fair.pkl"))

    print("Unfair model")
    model_unfair = train_model(train_data[["x1", "x2", "x3"]].values,
                               train_data["y_unfair"].values,
                               test_data[["x1", "x2", "x3"]].values,
                               test_data["y_unfair"].values)
    joblib.dump(model_unfair, os.path.join(args.output_dir, f"model_unfair.pkl"))