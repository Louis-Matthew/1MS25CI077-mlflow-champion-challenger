
import os

import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

TRACKING_URI = "http://127.0.0.1:5000"

REGISTERED_MODEL_NAME = "NewsSentimentClassifier"

MODEL_ALIAS = "champion"

NEW_DATA_PATH = "data/new_data.csv"

OUTPUT_DIR = "outputs"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "new_data_predictions.csv"
)


# ==========================================
# 2. CONNECT TO MLFLOW
# ==========================================

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(
    tracking_uri=TRACKING_URI
)

print("Connected to MLflow successfully.")


# ==========================================
# 3. RETRIEVE CHAMPION MODEL INFORMATION
# ==========================================

print("\nFinding the Champion model...")

champion_version = client.get_model_version_by_alias(
    name=REGISTERED_MODEL_NAME,
    alias=MODEL_ALIAS
)

model_version = champion_version.version

model_tags = champion_version.tags

hf_model_id = model_tags.get(
    "hf_model_id",
    "Not available"
)

print("\nChampion model information:")

print("Registered model:", REGISTERED_MODEL_NAME)

print("Alias:", MODEL_ALIAS)

print("Model version:", model_version)

print("Hugging Face model ID:", hf_model_id)


# ==========================================
# 4. LOAD CHAMPION MODEL FROM MLFLOW
# ==========================================

model_uri = (
    f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
)

print("\nLoading Champion model...")

champion_model = mlflow.pyfunc.load_model(
    model_uri
)

print("Champion model loaded successfully.")


# ==========================================
# 5. LOAD NEW INPUT EXAMPLES FROM CSV
# ==========================================

print("\nLoading new data from CSV...")

if not os.path.exists(NEW_DATA_PATH):
    raise FileNotFoundError(
        f"New data file not found: {NEW_DATA_PATH}"
    )

new_data_df = pd.read_csv(
    NEW_DATA_PATH
)

# Check whether the required column exists.

if "text" not in new_data_df.columns:
    raise ValueError(
        "new_data.csv must contain a column named 'text'."
    )

# Check for missing or empty text.

new_data_df["text"] = (
    new_data_df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

if (new_data_df["text"] == "").any():
    raise ValueError(
        "new_data.csv contains missing or empty text examples."
    )

# The assignment requires 5–10 new examples.

if not 5 <= len(new_data_df) <= 10:
    raise ValueError(
        "new_data.csv must contain between 5 and 10 examples."
    )

new_texts = new_data_df["text"].tolist()

print("New data loaded successfully.")

print("Number of new examples:", len(new_texts))


# ==========================================
# 6. GENERATE PREDICTIONS USING MLFLOW
# ==========================================

print("\nGenerating predictions using MLflow...")

# IMPORTANT:
# Predictions are made through the registered MLflow model.
# No Hugging Face pipeline is imported or called here.

raw_predictions = champion_model.predict(
    new_texts
)

print("Predictions generated successfully.")


# ==========================================
# 7. VALIDATE AND NORMALIZE PREDICTIONS
# ==========================================

# The logged Transformers model returns a DataFrame
# containing the columns "label" and "score".

if not isinstance(raw_predictions, pd.DataFrame):
    raise TypeError(
        "Unexpected prediction output type: "
        f"{type(raw_predictions)}"
    )

if "label" not in raw_predictions.columns:
    raise ValueError(
        "The MLflow prediction output has no label column."
    )

if "score" not in raw_predictions.columns:
    raise ValueError(
        "The MLflow prediction output has no score column."
    )

if len(raw_predictions) != len(new_texts):
    raise ValueError(
        "Number of predictions does not match input examples."
    )


# Convert model-specific labels into standard labels.

label_mapping = {

    "LABEL_0": "negative",

    "LABEL_1": "positive",

    "NEGATIVE": "negative",

    "POSITIVE": "positive"

}

predicted_labels = []

for raw_label in raw_predictions["label"]:

    normalized_label = str(raw_label).upper()

    if normalized_label not in label_mapping:
        raise ValueError(
            f"Unexpected sentiment label: {raw_label}"
        )

    predicted_labels.append(
        label_mapping[normalized_label]
    )


# Extract confidence scores.

confidence_scores = (
    raw_predictions["score"].tolist()
)


# ==========================================
# 8. CREATE RESULTS DATAFRAME
# ==========================================

results_df = pd.DataFrame({

    "text": new_texts,

    "predicted_sentiment": predicted_labels,

    "confidence": confidence_scores,

    "model_version": str(model_version),

    "hf_model_id": hf_model_id

})


# ==========================================
# 9. SAVE PREDICTIONS TO CSV
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPredictions saved successfully.")


# ==========================================
# 10. DISPLAY FINAL PREDICTION RESULTS
# ==========================================

print("\n========== CHAMPION PREDICTIONS ==========")

print(
    results_df.to_string(index=False)
)

print("\n========== MODEL INFORMATION ==========")

print("Registered model:", REGISTERED_MODEL_NAME)

print("Model alias:", MODEL_ALIAS)

print("Model version:", model_version)

print("Hugging Face model ID:", hf_model_id)

print("\nNumber of predictions:", len(results_df))

print("Output file:", OUTPUT_FILE)

print("\nChampion model inference completed successfully.")
