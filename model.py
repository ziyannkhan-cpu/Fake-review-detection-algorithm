
import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from preprocessing import clean_text


# ---------------- CONFIGURATION ----------------

DATA_PATH = os.path.join("data", "final_labeled_fake_reviews.csv")
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACT_DIR, "model_rf.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.pkl")
ENCODERS_PATH = os.path.join(ARTIFACT_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")


CATEGORICAL_COLS = [
    "asin",
    "parent_asin",
    "user_id",
    "verified_purchase"
]

SCALE_COLS = [
    "asin",
    "parent_asin",
    "user_id",
    "rating",
    "helpful_vote",
    "hour",
    "day_of_week",
    "date"
]


# ---------------- TRAINING FUNCTION ----------------

def main():

    if not os.path.exists("fake_review.csv"):
        raise FileNotFoundError(
            f"Dataset not found: {"fake_review.csv"}"
        )

    print("Loading data...")

    data = pd.read_csv("fake_review.csv")

    # Feature engineering

    data["date"] = pd.to_datetime(data["timestamp"])

    data["hour"] = data["date"].dt.hour

    data["day_of_week"] = data["date"].dt.dayofweek

    data.drop(
        columns=["timestamp", "images"],
        inplace=True,
        errors="ignore"
    )

    data.dropna(inplace=True)

    x = data.drop(
        columns=["label", "user_timestamp", "user_review_burst"]
    )

    y = data["label"]

    # Combine title and review text

    x["tag"] = (
        x["title"].astype(str) + " " +
        x["text"].astype(str)
    )

    x.drop(
        columns=["title", "text"],
        inplace=True
    )

    # Text cleaning

    print("Cleaning text...")

    x_text_series = x["tag"].apply(clean_text)

    # NLP vectorization

    print("Vectorizing text...")

    vectorizer = CountVectorizer(max_features=5000)

    x_text = vectorizer.fit_transform(x_text_series)

    x.drop(columns=["tag"], inplace=True)

    # Encode categorical columns

    print("Encoding categorical features...")

    encoders = {}

    for col in CATEGORICAL_COLS:

        enc = LabelEncoder()

        x[col] = enc.fit_transform(x[col].astype(str))

        encoders[col] = enc

    # Date conversion

    x["date"] = (
        x["date"].dt.strftime("%Y%m%d").astype(int)
    )

    # Numerical scaling

    print("Scaling numerical features...")

    scaler = MinMaxScaler()

    x[SCALE_COLS] = scaler.fit_transform(
        x[SCALE_COLS]
    )

    # Preserve exact feature order

    feature_columns = list(x.columns)

    # Combine NLP and numerical features

    X_final = hstack(
        (x_text, csr_matrix(x.to_numpy())),
        format="csr"
    )

    # Train-test split

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=0.33,
        random_state=42
    )

    # Train model

    print("Training Random Forest Classifier...")

    model_rf = RandomForestClassifier(
        max_depth=100,
        random_state=42,
        n_jobs=-1
    )

    model_rf.fit(X_train, y_train)

    # Evaluate

    y_pred_rf = model_rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred_rf)

    print(f"Model accuracy: {acc:.4f}")

    # Save artifacts

    print("Saving model artifacts...")

    joblib.dump(model_rf, MODEL_PATH)

    joblib.dump(vectorizer, VECTORIZER_PATH)

    joblib.dump(encoders, ENCODERS_PATH)

    joblib.dump(scaler, SCALER_PATH)

    joblib.dump(feature_columns, FEATURES_PATH)

    print("All artifacts saved successfully!")


# ---------------- LOAD MODEL ----------------

def load_model():

    required_files = [
        MODEL_PATH,
        VECTORIZER_PATH,
        ENCODERS_PATH,
        SCALER_PATH,
        FEATURES_PATH
    ]

    for file in required_files:

        if not os.path.exists(file):
            raise FileNotFoundError(
                f"Missing artifact: {file}. Run python model.py first."
            )

    model_rf = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    encoders = joblib.load(ENCODERS_PATH)

    scaler = joblib.load(SCALER_PATH)

    feature_columns = joblib.load(FEATURES_PATH)

    return (
        model_rf,
        vectorizer,
        encoders,
        scaler,
        feature_columns
    )


# ---------------- PREDICTION FUNCTION ----------------

def predict_review(
    title,
    text,
    asin,
    parent_asin,
    user_id,
    verified_purchase,
    rating,
    helpful_vote,
    timestamp
):

    (
        model_rf,
        vectorizer,
        encoders,
        scaler,
        feature_columns
    ) = load_model()

    # Prepare input

    input_data = pd.DataFrame([{

        "asin": str(asin),

        "parent_asin": str(parent_asin),

        "user_id": str(user_id),

        "verified_purchase": str(verified_purchase),

        "rating": rating,

        "helpful_vote": helpful_vote,

        "timestamp": timestamp

    }])

    # Feature engineering

    input_data["date"] = pd.to_datetime(
        input_data["timestamp"]
    )

    input_data["hour"] = input_data["date"].dt.hour

    input_data["day_of_week"] = input_data["date"].dt.dayofweek

    input_data.drop(
        columns=["timestamp"],
        inplace=True
    )

    # Text processing

    combined_text = str(title) + " " + str(text)

    cleaned_text = clean_text(combined_text)

    x_text = vectorizer.transform([cleaned_text])

    # Encode categorical features

    for col in CATEGORICAL_COLS:

        encoder = encoders[col]

        value = str(input_data[col].iloc[0])

        if value in encoder.classes_:

            input_data[col] = encoder.transform([value])

        else:

            # Unknown category handling
            input_data[col] = -1

    # Date conversion

    input_data["date"] = (
        input_data["date"].dt.strftime("%Y%m%d").astype(int)
    )

    # Scale numerical features

    input_data[SCALE_COLS] = scaler.transform(
        input_data[SCALE_COLS]
    )

    # Match training feature order

    input_data = input_data[feature_columns]

    # Combine text and numerical features

    X_final = hstack(
        (x_text, csr_matrix(input_data.to_numpy())),
        format="csr"
    )

    # Prediction

    prediction = model_rf.predict(X_final)[0]

    probabilities = model_rf.predict_proba(X_final)[0]

    classes = model_rf.classes_

    confidence = max(probabilities) * 100

    probability_dict = {
        str(label): float(probability) * 100
        for label, probability in zip(classes, probabilities)
    }

    return prediction, confidence, probability_dict


# ---------------- MAIN ----------------

if __name__ == "__main__":

    main()
