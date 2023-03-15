"""
This script trains a model on a dataset and saves it to disk.
It also saves the training and test set to disk,
and a background set and explanation set.
The explanation set is a subset of the test set.
"""
import argparse
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os
from util import DATASETS


def train_model(dataset):
    ds = datasets.fetch_openml(data_id=DATASETS[dataset]["id"], parser="auto")
    X, y = ds.data.dropna(), ds.target[~ds.data.isna().any(axis=1)]

    categ_columns = X.select_dtypes('category').columns
    num_columns = X.select_dtypes(['int64', 'float64']).columns
    X_copy = X.copy()
    X_copy[categ_columns] = OrdinalEncoder().fit_transform(X[categ_columns])
    X_copy[num_columns] = X[num_columns].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X_copy, y, test_size=0.2, random_state=42)

    if DATASETS[dataset]["type"] == "classification":
        clf = GradientBoostingClassifier()
        clf.fit(X_train.values, y_train)
        print(f"Train bal acc: {balanced_accuracy_score(y_train, clf.predict(X_train.values)):.3f}")
        print(f"Test bal acc: {balanced_accuracy_score(y_test, clf.predict(X_test.values)):.3f}")
        return clf, X_train, X_test, y_train, y_test
    if DATASETS[dataset]["type"] == "regression":
        reg = GradientBoostingRegressor()
        reg.fit(X_train.values, y_train)
        print(f"Train R2: {r2_score(y_train, reg.predict(X_train.values)):.3f}")
        print(f"Test R2: {r2_score(y_test, reg.predict(X_test.values)):.3f}")
        return clf, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="adult")
    parser.add_argument("-o", "--output-dir", default="data")
    parser.add_argument("-b", "--background-size", type=int, default=100)
    parser.add_argument("-e", "--exp-size", type=int, default=100)
    args = parser.parse_args()

    model, X_train, X_test, y_train, y_test = train_model(args.dataset)

    out_dir = os.path.join(args.output_dir, args.dataset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    joblib.dump(X_train, os.path.join(out_dir, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(out_dir, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(out_dir, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(out_dir, "y_test.pkl"))
    
    background_set = X_train.sample(args.background_size)
    if len(X_test) > args.exp_size:
        exp_set = X_test.sample(args.exp_size)
    else:
        exp_set = X_test
    joblib.dump(background_set, os.path.join(out_dir, "background_set.pkl"))
    joblib.dump(exp_set, os.path.join(out_dir, "exp_set.pkl"))