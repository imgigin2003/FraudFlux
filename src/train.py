import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier


def load_processed_data(processed_path="../data/processed"):
    # Step 1 -> from the previous step, we load the data we processed
    X_train = np.load(f"{processed_path}/X_train.npy")
    X_test = np.load(f"{processed_path}/X_test.npy")
    y_train = np.load(f"{processed_path}/y_train.npy")
    y_test = np.load(f"{processed_path}/y_test.npy")

    return X_train, X_test, y_train, y_test


def train_baseline_model(X_train, y_train):
    # Step 3-1 -> Creating our baseline with logistic regression
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    # Step 3-2 -> Creating our improved model with DecisionTree
    model = DecisionTreeClassifier(
        class_weight="balanced", random_state=42, max_depth=5
    )
    model.fit(X_train, y_train)
    return model


# def train_decision_tree_tuned(X_train, y_train):
#     # Step 3-2 -> Creating our improved model with DecisionTree & GridSearch
#     param_grid = {
#         "max_depth": [3, 5, 7, 10],
#         "min_samples_leaf": [1, 5, 10, 20],
#     }

#     f2_scorer = make_scorer(fbeta_score, beta=2)
#     model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
#     grid_search = GridSearchCV(
#         estimator=model, param_grid=param_grid, cv=5, scoring=f2_scorer, n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)

#     print(f"Best Params: {grid_search.best_params_}")
#     print(f"Best F2 Score: {grid_search.best_score_:.4f}")

#     return grid_search.best_estimator_


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        class_weight="balanced", random_state=42, n_estimators=150, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def save_baseline_model(model, output_dir="../models"):
    # Step 4-1 -> saving the baseline model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "baseline_model.pkl"))
    print(f"Model Saved to {output_dir}")


def save_tree_model(model, output_dir="../models"):
    # Step 4-2 -> saving the Decision Tree model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "decision_tree_model.pkl"))
    print(f"Model Saved to {output_dir}")


# def save_tree_tuned_model(model, output_dir="../models"):
#     # Step 4-3 -> saving the Decision Tree Tuned model
#     os.makedirs(output_dir, exist_ok=True)
#     joblib.dump(model, os.path.join(output_dir, "decision_tree_tuned_model.pkl"))
#     print(f"Model Saved to {output_dir}")


def save_random_forest_model(model, output_dir="../models"):
    # Step 4-4 -> saving the Random Forest model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "random_forest.pkl"))
    print(f"Model Saved to {output_dir}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()

    baseline_model = train_baseline_model(X_train, y_train)
    save_baseline_model(baseline_model)

    decision_tree_model = train_decision_tree(X_train, y_train)
    save_tree_model(decision_tree_model)

    # decision_tree_tuned_model = train_decision_tree_tuned(X_train, y_train)
    # save_tree_tuned_model(decision_tree_tuned_model)

    random_forest_model = train_random_forest(X_train, y_train)
    save_random_forest_model(random_forest_model)
