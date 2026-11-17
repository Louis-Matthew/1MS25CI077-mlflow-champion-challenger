
import json
import time

import mlflow
from mlflow.tracking import MlflowClient


# ==========================================
# 1. CONFIGURATION
# ==========================================

TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "Champion-Challenger-Sentiment"

REGISTERED_MODEL_NAME = "NewsSentimentClassifier"

MODEL_A_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B_ID = "textattack/bert-base-uncased-SST-2"

DECISION_FILE = "outputs/comparison_decision.json"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)


# ==========================================
# 2. READ THE CHAMPION DECISION
# ==========================================

with open(DECISION_FILE, "r", encoding="utf-8") as file:
    decision = json.load(file)

champion_model_id = decision["champion_model_id"]

print("Champion selected:", decision["champion"])
print("Champion model ID:", champion_model_id)


# ==========================================
# 3. FIND THE LATEST SUCCESSFUL MODEL RUNS
# ==========================================

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError("MLflow experiment was not found.")


def find_latest_model_run(run_name):

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=20
    )

    for run in runs:

        if run.info.status != "FINISHED":
            continue

        artifacts = client.list_artifacts(
            run.info.run_id,
            "model"
        )

        has_mlmodel_file = any(
            item.path.replace("\\", "/").endswith("MLmodel")
            for item in artifacts
        )

        if has_mlmodel_file:
            return run

    raise ValueError(
        f"No completed model artifact found for {run_name}."
    )


run_a = find_latest_model_run("A_sentiment_distilbert")
run_b = find_latest_model_run("B_sentiment_bert")

print("\nModel A run ID:", run_a.info.run_id)
print("Model B run ID:", run_b.info.run_id)


# ==========================================
# 4. REGISTER BOTH MODELS
# ==========================================

model_specs = [
    {
        "run": run_a,
        "model_id": MODEL_A_ID,
        "role": "Model A",
    },
    {
        "run": run_b,
        "model_id": MODEL_B_ID,
        "role": "Model B",
    },
]

registered_versions = {}

for spec in model_specs:

    run_id = spec["run"].info.run_id

    model_uri = f"runs:/{run_id}/model"

    print(f"\nRegistering {spec['role']}...")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    version_number = model_version.version

    # Wait for the registered model version to become READY.
    for attempt in range(120):

        current_version = client.get_model_version(
            name=REGISTERED_MODEL_NAME,
            version=version_number
        )

        if current_version.status == "READY":
            break

        if current_version.status == "FAILED_REGISTRATION":
            raise RuntimeError(
                f"Registration failed for {spec['role']}."
            )

        time.sleep(1)

    else:
        raise TimeoutError(
            f"Registration timed out for {spec['role']}."
        )

    # Store useful metadata on the model version.
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version_number,
        "hf_model_id",
        spec["model_id"]
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version_number,
        "model_role",
        spec["role"]
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version_number,
        "student_usn",
        "1MS25CI077"
    )

    registered_versions[spec["model_id"]] = version_number

    print(
        f"{spec['role']} registered successfully "
        f"as version {version_number}."
    )


# ==========================================
# 5. ASSIGN CHAMPION AND CHALLENGER ALIASES
# ==========================================

if champion_model_id not in registered_versions:
    raise ValueError(
        "Champion model ID does not match either registered model."
    )

challenger_model_id = (
    MODEL_A_ID
    if champion_model_id == MODEL_B_ID
    else MODEL_B_ID
)

champion_version = registered_versions[champion_model_id]

challenger_version = registered_versions[challenger_model_id]


client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "champion",
    champion_version
)

client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "challenger",
    challenger_version
)


# ==========================================
# 6. DISPLAY REGISTRY RESULTS
# ==========================================

print("\n========== MODEL REGISTRY RESULTS ==========")

print("Registered model name:", REGISTERED_MODEL_NAME)

print(
    "Champion alias:",
    f"version {champion_version}",
    champion_model_id
)

print(
    "Challenger alias:",
    f"version {challenger_version}",
    challenger_model_id
)

print("\nBoth models registered and aliases assigned successfully.")
