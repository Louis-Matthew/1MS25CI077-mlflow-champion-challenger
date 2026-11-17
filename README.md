# 1MS25CI077-mlflow-champion-challenger
# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Institution:** Ramaiah Institute of Technology  
**USN:** 1MS25CI077  
**Project:** MLOps with MLflow Mini Project  
**Task:** Sentiment Classification

---

## 1. Task and Models

### Objective

The objective of this project is to compare two pretrained Hugging Face sentiment classification models using identical evaluation inputs, track their performance using MLflow, and select a Champion model using a predefined decision rule.

### Task

- **Task:** Binary Sentiment Classification
- **Classes:** Positive and Negative
- **Evaluation inputs:** `data/eval_inputs.csv`

### Dataset

- **Source:** Lecturer-provided news sentiment analysis dataset.
- **Original records:** 3,500.
- **Evaluation records:** 200.
- **Positive examples:** 100.
- **Negative examples:** 100.
- **Duplicate handling:** Duplicate URLs removed before sampling.
- **Neutral examples:** Excluded because both selected models perform binary sentiment classification.
- **Random seed:** 42.
- **Input text:** Combined news title and description.
- **Text preprocessing:** HTML entities decoded and whitespace normalized.
- **Evaluation file:** `data/eval_inputs.csv`.

Both models will receive exactly the same 200 evaluation texts in the same order.

### Model A — Initial Champion

- **Hugging Face Model ID:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Role:** Initial Champion
- **Description:** DistilBERT-based binary sentiment classification model.

### Model B — Challenger

- **Hugging Face Model ID:** `textattack/bert-base-uncased-SST-2`
- **Role:** Challenger
- **Description:** BERT-based binary sentiment classification model.
- **Special handling:** Its `LABEL_0` and `LABEL_1` outputs must be mapped to the correct sentiment labels.

---

## 2. How to Run

The following commands will be finalized and tested as the project implementation is completed.

### Environment Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Start MLflow Tracking Server

```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

### Run Model Evaluation

```powershell
python evaluate.py
```

### Register Models and Assign Aliases

```powershell
python register_champion.py
```

### Predict on New Data

```powershell
python predict_from_mlflow.py
```

---

## 3. Champion–Challenger Decision Rule

The decision rule is defined before the final experiment run and will not be changed after examining the results.

### Primary Metric

Macro-F1 score is the primary evaluation metric because it calculates the F1 score independently for each sentiment class and averages them equally.

- Higher Macro-F1 is better.
- Lower inference latency is better.

### Minimum Improvement Margin

The Challenger must improve Macro-F1 by at least **0.01 (one percentage point)** over the current Champion to qualify for replacement based on the primary metric.

This margin is selected to avoid promoting a model based on a very small performance difference.

### Tie-Breaker

If the difference in Macro-F1 is less than 0.01 in either direction, the model with lower average inference latency will be preferred.

If both models have effectively equal latency, the existing Champion (Model A) will remain Champion.

### Final Decision Logic

1. If Model B's Macro-F1 exceeds Model A's by at least 0.01, Model B becomes Champion.
2. If Model A's Macro-F1 exceeds Model B's by at least 0.01, Model A remains Champion.
3. If the difference is less than 0.01, select the faster model based on average inference latency.
4. If both performance and latency are effectively equal, retain Model A.

The final decision will be recorded in the MLflow comparison run and justified using evaluation metrics and inference time.

---

## 4. Results Table

The following table will be completed after evaluating both models on the same saved evaluation dataset.

| Metric | Model A | Model B | Difference |
|---|---|---|---|
| Accuracy | TBD | TBD | TBD |
| Precision | TBD | TBD | TBD |
| Recall | TBD | TBD | TBD |
| Macro-F1 | TBD | TBD | TBD |
| Average latency (ms) | TBD | TBD | TBD |
| Total inference time (s) | TBD | TBD | TBD |

---

## 5. Final Champion Decision

**Selected Champion:** To be determined after evaluation.

The final explanation will discuss:

- Whether the Challenger improves on the initial Champion.
- The difference in Macro-F1.
- Whether the improvement is meaningful given the evaluation sample size.
- The trade-offs in inference speed, model size and classification errors.
- Why the final model was selected according to the predefined decision rule.

The final explanation will be written after the experiment and supported by the MLflow results.

---

## 6. Limitations and Declaration

### Limitations

- The evaluation uses a limited subset of data rather than the entire dataset.
- Model performance may vary with different evaluation samples.
- Inference latency may vary depending on hardware and system load.
- Small differences in evaluation metrics may not indicate a meaningful improvement.

### Declaration

This project is an individual academic mini-project.

AI assistance was used for guidance, understanding concepts, debugging and development support. The implementation, experiment results and final decisions will be reviewed and understood by the student.

All code and design choices will be studied and explained during the viva.

---

**Project Repository:** 1MS25CI077-mlflow-champion-challenger