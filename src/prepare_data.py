import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def load_data(dataset_path="../dataset/creditcard.csv"):
    # Step 1 -> Loading the dataset, removing the duplicates
    csv = pd.read_csv(dataset_path)
    csv.drop_duplicates(inplace=True)
    return csv


def split_data(df, test_size=0.2, random_state=42):
    # Step 2 -> Split the training data from test data
    # we need every row as X (features) but "Class"
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, columns_to_scale=["Time", "Amount"]):
    # Step 3 -> two of our features have large values, we need to scale them to avoid feature bullying
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train, X_test, scaler


def save(X_train, X_test, y_train, y_test, scaler, output_dir="../data/processed"):
    # Step 4 -> After this, training reads from data/processed/ instead of re-processing every run.
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train.to_numpy())
    np.save(os.path.join(output_dir, "X_test.npy"), X_test.to_numpy())
    np.save(os.path.join(output_dir, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(output_dir, "y_test.npy"), y_test.to_numpy())

    # Step 5 -> Save the scaler we just defined
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print(f"Saved processed data to {output_dir}")


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    save(X_train, X_test, y_train, y_test, scaler)
