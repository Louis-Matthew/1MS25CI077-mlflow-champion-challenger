import json
import os
import time

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import sklearn
import torch
import transformers

from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

TASK_NAME = "sentiment"
MODEL_A_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B_ID = "textattack/bert-base-uncased-SST-2"

DATASET_PATH = "data/eval_inputs.csv"
SAMPLE_SIZE = 200
RANDOM_SEED = 42
BATCH_SIZE = 8

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Champion-Challenger-Sentiment"

MINIMUM_IMPROVEMENT = 0.01
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "GPU" if DEVICE == 0 else "CPU"

print("Evaluation initialized.")
print("Model A:", MODEL_A_ID)
print("Model B:", MODEL_B_ID)
print("Evaluation sample size:", SAMPLE_SIZE)
print("Device:", DEVICE_NAME)


# ==========================================
# 2. LOAD EVALUATION DATA
# ==========================================

def load_evaluation_data():
    df = pd.read_csv(DATASET_PATH)

    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must contain text and label columns.")

    if len(df) < SAMPLE_SIZE:
        raise ValueError(f"Expected at least {SAMPLE_SIZE} rows.")

    eval_df = df.head(SAMPLE_SIZE).copy()
    eval_df["label"] = eval_df["label"].str.lower()

    if not set(eval_df["label"]).issubset({"positive", "negative"}):
        raise ValueError("Unexpected sentiment labels found.")

    if len(eval_df) != SAMPLE_SIZE:
        raise ValueError("Evaluation dataset must contain 200 rows.")

    print("\nEvaluation examples:", len(eval_df))
    print(eval_df["label"].value_counts())

    return eval_df


eval_df = load_evaluation_data()
evaluation_texts = eval_df["text"].tolist()
actual_labels = eval_df["label"].tolist()


# ==========================================
# 3. LOAD HUGGING FACE MODELS
# ==========================================

def load_models():
    print("\nLoading Model A...")
    model_a = pipeline(
        "sentiment-analysis",
        model=MODEL_A_ID,
        device=DEVICE
    )

    print("Model A loaded successfully.")

    print("\nLoading Model B...")
    model_b = pipeline(
        "sentiment-analysis",
        model=MODEL_B_ID,
        device=DEVICE
    )

    print("Model B loaded successfully.")

    return model_a, model_b


model_a, model_b = load_models()


# ==========================================
# 4. INFERENCE AND LABEL NORMALIZATION
# ==========================================

def normalize_prediction(raw_label, model_name):
    if model_name == MODEL_B_ID:
        mapping = {
            "LABEL_0": "negative",
            "LABEL_1": "positive"
        }

        if raw_label not in mapping:
            raise ValueError(f"Unexpected Model B label: {raw_label}")

        return mapping[raw_label]

    label = raw_label.lower()

    if label not in {"positive", "negative"}:
        raise ValueError(f"Unexpected Model A label: {raw_label}")

    return label


def run_inference(model_pipeline, texts, model_name):
    start_time = time.perf_counter()

    raw_results = model_pipeline(
        texts,
        batch_size=BATCH_SIZE,
        truncation=True
    )

    total_time = time.perf_counter() - start_time

    predictions = [
        {
            "prediction": normalize_prediction(
                result["label"], model_name
            ),
            "confidence": float(result["score"])
        }
        for result in raw_results
    ]

    average_latency = total_time / len(texts)

    return predictions, total_time, average_latency


print("\nRunning Model A inference...")
predictions_a, total_time_a, avg_latency_a = run_inference(
    model_a, evaluation_texts, MODEL_A_ID
)
print("Model A inference completed.")

print("\nRunning Model B inference...")
predictions_b, total_time_b, avg_latency_b = run_inference(
    model_b, evaluation_texts, MODEL_B_ID
)
print("Model B inference completed.")

pred_a = [p["prediction"] for p in predictions_a]
pred_b = [p["prediction"] for p in predictions_b]


# ==========================================
# 5. CALCULATE METRICS
# ==========================================

def calculate_metrics(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision_macro": precision_score(
            actual, predicted, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            actual, predicted, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            actual, predicted, average="macro", zero_division=0
        )
    }


metrics_a = calculate_metrics(actual_labels, pred_a)
metrics_b = calculate_metrics(actual_labels, pred_b)

metrics_a.update({
    "total_inference_time": total_time_a,
    "average_latency": avg_latency_a
})

metrics_b.update({
    "total_inference_time": total_time_b,
    "average_latency": avg_latency_b
})

print("\n========== MODEL A METRICS ==========")
for name, value in metrics_a.items():
    print(f"{name}: {value:.4f}")

print("\n========== MODEL B METRICS ==========")
for name, value in metrics_b.items():
    print(f"{name}: {value:.4f}")


# ==========================================
# 6. SAVE PREDICTIONS AND METRICS
# ==========================================

results_df = eval_df.copy()

results_df["actual_label"] = actual_labels
results_df["model_a_prediction"] = pred_a
results_df["model_a_confidence"] = [
    p["confidence"] for p in predictions_a
]
results_df["model_b_prediction"] = pred_b
results_df["model_b_confidence"] = [
    p["confidence"] for p in predictions_b
]

predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")
results_df.to_csv(predictions_path, index=False)

metrics_summary = {
    "model_a": {"model_id": MODEL_A_ID, **metrics_a},
    "model_b": {"model_id": MODEL_B_ID, **metrics_b},
    "evaluation_sample_count": len(eval_df)
}

metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics_summary, file, indent=4)

print("\nPredictions and metrics saved successfully.")


# ==========================================
# 7. LOG MODEL RUNS TO MLFLOW
# ==========================================

common_params = {
    "task": TASK_NAME,
    "evaluation_sample_count": len(eval_df),
    "random_seed": RANDOM_SEED,
    "device": DEVICE_NAME,
    "batch_size": BATCH_SIZE,
    "dataset": "news_sentiment_analysis.csv"
}

common_tags = {
    "student_usn": "1MS25CI077",
    "dataset_name": "news_sentiment_analysis.csv",
    "mlflow_version": mlflow.__version__,
    "transformers_version": transformers.__version__,
    "torch_version": torch.__version__,
    "sklearn_version": sklearn.__version__
}


def log_model_evaluation(run_name, model_id, role, metrics, model_pipeline):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **common_params,
            "hf_model_id": model_id,
            "model_role": role
        })

        mlflow.log_metrics(metrics)
        mlflow.set_tags(common_tags)

        mlflow.log_artifact(predictions_path)
        mlflow.log_artifact(metrics_path)

        # Log the actual Hugging Face model.
        mlflow.transformers.log_model(
            transformers_model=model_pipeline,
            artifact_path="model"
        )

        print(f"\nMLflow model logged: {run_name}")


log_model_evaluation(
    "A_sentiment_distilbert",
    MODEL_A_ID,
    "initial_champion",
    metrics_a,
    model_a
)

log_model_evaluation(
    "B_sentiment_bert",
    MODEL_B_ID,
    "challenger",
    metrics_b,
    model_b
)

print("\nBoth model runs logged successfully.")


# ==========================================
# 8. CHAMPION-CHALLENGER DECISION
# ==========================================

f1_difference = metrics_b["macro_f1"] - metrics_a["macro_f1"]

if f1_difference >= MINIMUM_IMPROVEMENT:
    champion = "Model B"
    champion_model_id = MODEL_B_ID
    decision_reason = (
        "Model B has a macro-F1 improvement of at least 0.01."
    )

elif f1_difference <= -MINIMUM_IMPROVEMENT:
    champion = "Model A"
    champion_model_id = MODEL_A_ID
    decision_reason = (
        "Model A has a macro-F1 improvement of at least 0.01."
    )

elif avg_latency_a < avg_latency_b:
    champion = "Model A"
    champion_model_id = MODEL_A_ID
    decision_reason = (
        "Macro-F1 difference is below 0.01; Model A is faster."
    )

elif avg_latency_b < avg_latency_a:
    champion = "Model B"
    champion_model_id = MODEL_B_ID
    decision_reason = (
        "Macro-F1 difference is below 0.01; Model B is faster."
    )

else:
    champion = "Model A"
    champion_model_id = MODEL_A_ID
    decision_reason = (
        "Macro-F1 difference is below 0.01 and latency is tied; "
        "retain Model A."
    )


comparison_results = {
    "champion": champion,
    "champion_model_id": champion_model_id,
    "decision_reason": decision_reason,
    "minimum_improvement_margin": MINIMUM_IMPROVEMENT,
    "model_a_macro_f1": metrics_a["macro_f1"],
    "model_b_macro_f1": metrics_b["macro_f1"],
    "macro_f1_difference_b_minus_a": f1_difference,
    "model_a_average_latency": avg_latency_a,
    "model_b_average_latency": avg_latency_b
}

decision_path = os.path.join(OUTPUT_DIR, "comparison_decision.json")

with open(decision_path, "w", encoding="utf-8") as file:
    json.dump(comparison_results, file, indent=4)


# ==========================================
# 9. GENERATE EVALUATION PLOTS
# ==========================================

class_labels = ["negative", "positive"]

cm_a = confusion_matrix(actual_labels, pred_a, labels=class_labels)
cm_b = confusion_matrix(actual_labels, pred_b, labels=class_labels)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, matrix, title in [
    (axes[0], cm_a, "Model A - DistilBERT"),
    (axes[1], cm_b, "Model B - BERT")
]:
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(class_labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_labels)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center", color="black"
            )

fig.colorbar(image, ax=axes)
plt.tight_layout()

confusion_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
plt.savefig(confusion_path, dpi=300)
plt.close()


# Model comparison chart
metric_names = ["Accuracy", "Macro-F1"]
values_a = [metrics_a["accuracy"], metrics_a["macro_f1"]]
values_b = [metrics_b["accuracy"], metrics_b["macro_f1"]]

x = range(len(metric_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    [i - width / 2 for i in x], values_a, width,
    label="Model A - DistilBERT"
)

ax.bar(
    [i + width / 2 for i in x], values_b, width,
    label="Model B - BERT"
)

ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.set_xticks(list(x))
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()

comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
plt.savefig(comparison_path, dpi=300)
plt.close()


# ==========================================
# 10. SAVE FIVE DISAGREEMENT EXAMPLES
# ==========================================

disagreements = results_df[
    results_df["model_a_prediction"] != results_df["model_b_prediction"]
].copy()

disagreement_path = os.path.join(
    OUTPUT_DIR, "disagreement_examples.csv"
)

disagreement_examples = disagreements.head(5).copy()

# Preserve existing analyst comments when the script is rerun.
if os.path.exists(disagreement_path):
    previous = pd.read_csv(disagreement_path)

    if {"text", "analyst_comment"}.issubset(previous.columns):
        comments = (
            previous.drop_duplicates("text")
            .set_index("text")["analyst_comment"]
        )
        disagreement_examples["analyst_comment"] = (
            disagreement_examples["text"].map(comments).fillna("")
        )
    else:
        disagreement_examples["analyst_comment"] = ""
else:
    disagreement_examples["analyst_comment"] = ""

disagreement_examples.to_csv(disagreement_path, index=False)

print("\nEvaluation artifacts generated:")
print(confusion_path)
print(comparison_path)
print(disagreement_path)
print("Number of disagreements:", len(disagreements))

if len(disagreement_examples) < 5:
    print("Warning: Fewer than five disagreement examples were found.")


# ==========================================
# 11. LOG COMPARISON RUN TO MLFLOW
# ==========================================

with mlflow.start_run(run_name="compare_sentiment"):
    mlflow.log_params({
        "task": TASK_NAME,
        "model_a": MODEL_A_ID,
        "model_b": MODEL_B_ID,
        "primary_metric": "macro_f1",
        "minimum_improvement_margin": MINIMUM_IMPROVEMENT,
        "margin": MINIMUM_IMPROVEMENT,
        "tie_breaker": "average_latency",
        "evaluation_sample_count": len(eval_df)
    })

    mlflow.log_metrics({
        "model_a_macro_f1": metrics_a["macro_f1"],
        "model_b_macro_f1": metrics_b["macro_f1"],
        "macro_f1_difference": f1_difference,
        "delta_primary_metric": f1_difference,
        "model_a_accuracy": metrics_a["accuracy"],
        "model_b_accuracy": metrics_b["accuracy"],
        "model_a_average_latency": avg_latency_a,
        "model_b_average_latency": avg_latency_b
    })

    mlflow.set_tags({
        **common_tags,
        "selected_champion": champion,
        "winner": champion,
        "decision_reason": decision_reason
    })

    for artifact in [
        decision_path,
        metrics_path,
        predictions_path,
        confusion_path,
        comparison_path,
        disagreement_path
    ]:
        mlflow.log_artifact(artifact)

    print("\nComparison run logged successfully.")


# ==========================================
# 12. DISPLAY FINAL DECISION
# ==========================================

print("\n========== FINAL COMPARISON DECISION ==========")
print("Champion:", champion)
print("Champion model ID:", champion_model_id)
print("Reason:", decision_reason)
print(f"Macro-F1 difference (B - A): {f1_difference:.4f}")

print("\nAll evaluation files are saved in the outputs folder.")
