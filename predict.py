import joblib
import pandas as pd


def load_model(model_path="./models/random_forest.pkl"):
    # Step 1 -> Loading the trained model
    model = joblib.load(model_path)
    return model


def load_scaler(scaler_path="./data/processed/scaler.pkl"):
    # Step 2 -> Loading the saved scaler
    scaler = joblib.load(scaler_path)
    return scaler


def predict(
    data_path, model, scaler, threshold=0.15, columns_to_scale=["Time", "Amount"]
):
    # Step 4 -> using everything we have, we make the prediction
    df = pd.read_csv(data_path)

    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.transform(df[columns_to_scale])

    df_scaled.to_numpy()

    probs = model.predict_proba(df_scaled)[:, 1]
    predictions = (probs >= threshold).astype(int)

    df["fraud_probability"] = probs
    df["prediction"] = predictions

    for i, row in df.iterrows():
        label = "FRAUD" if row["prediction"] == 1 else "Not Fraud"
        print(
            f"Transaction {i}: {row['fraud_probability']*100:.1f}% fraud probability -> {label}"
        )

    return probs, predictions


if __name__ == "__main__":
    model = load_model()
    scaler = load_scaler()

    dataset = "./dataset/testcreditcard.csv"

    probability, prediction = predict(dataset, model, scaler)
