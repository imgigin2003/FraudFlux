import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
from train import load_processed_data
import joblib
import numpy as np


def load_trained_baseline_model(model_path="../models/baseline_model.pkl"):
    # Step 1-1 -> Loading our baseline model
    model = joblib.load(model_path)
    return model


def load_trained_decision_tree_model(model_path="../models/decision_tree_model.pkl"):
    # Step 1-2 -> Loading our decision tree model
    model = joblib.load(model_path)
    return model


# def load_trained_decision_tree_tuned_model(
#     model_path="../models/decision_tree_tuned_model.pkl",
# ):
#     # Step 1-3 -> Loading our tuned decision tree model
#     model = joblib.load(model_path)
#     return model


def load_trained_random_forest(model_path="../models/random_forest.pkl"):
    # Step 1-3 -> Loading our Random Forest model
    model = joblib.load(model_path)
    return model


def eval_baseline(model, X_test, y_test, threshold):
    print(f"\n--- Logistic Regression ---")
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"--- Threshold: {threshold} ---")
    print(f"Accuracy: {accuracy*100:.4f}%")
    print(f"Precision Score: {precision*100:.4f}%")
    print(f"Recall Score: {recall*100:.4f}%")
    print(f"F1 Score: {f1*100:.4f}%")
    print(f"Confusion Matrix: \n{cm}")


def eval_decision_tree(model, X_test, y_test, threshold):
    print(f"\n--- Decision Tree ---")
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"--- Threshold: {threshold} ---")
    print(f"Accuracy: {accuracy*100:.4f}%")
    print(f"Precision Score: {precision*100:.4f}%")
    print(f"Recall Score: {recall*100:.4f}%")
    print(f"F1 Score: {f1*100:.4f}%")
    print(f"Confusion Matrix: \n{cm}")


# def eval_decition_tree_tuned(model, X_test, y_test, threshold):
#     print("\n --- Decision Tree (Tuned with GridSearch) ---")
#     probs = model.predict_proba(X_test)[:, 1]
#     y_pred = (probs >= threshold).astype(int)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)

#     print(f"--- Threshold: {threshold} ---")
#     print(f"Accuracy: {accuracy*100:.4f}%")
#     print(f"Precision Score: {precision*100:.4f}%")
#     print(f"Recall Score: {recall*100:.4f}%")
#     print(f"F1 Score: {f1*100:.4f}%")
#     print(f"Confusion Matrix: \n{cm}")


def eval_random_forest(model, X_test, y_test, threshold):
    print("\n --- Random Forest ---")
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"--- Threshold: {threshold} ---")
    print(f"Accuracy: {accuracy*100:.4f}%")
    print(f"Precision Score: {precision*100:.4f}%")
    print(f"Recall Score: {recall*100:.4f}%")
    print(f"F1 Score: {f1*100:.4f}%")
    print(f"Confusion Matrix: \n{cm}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()

    baseline_model = load_trained_baseline_model()
    eval_baseline(baseline_model, X_test, y_test, threshold=0.7)

    decision_tree_model = load_trained_decision_tree_model()
    eval_decision_tree(decision_tree_model, X_test, y_test, threshold=0.9)

    # decision_tree_tuned_model = load_trained_decision_tree_tuned_model()
    # for t in [0.85, 0.90, 0.9021, 0.93, 0.95, 0.97, 0.99]:
    #     eval_decition_tree_tuned(decision_tree_tuned_model, X_test, y_test, t)

    random_forest_model = load_trained_random_forest()
    eval_random_forest(random_forest_model, X_test, y_test, threshold=0.15)
