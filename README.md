# 1MS25CI077-mlflow-champion-challenger

# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Institution:** Ramaiah Institute of Technology  
**USN:** 1MS25CI077  
**Project:** MLOps with MLflow Mini Project  
**Task:** Binary Sentiment Classification

---

## 1. Task and Models

### Objective

Compare two pretrained Hugging Face sentiment classification models using identical evaluation inputs, track their performance using MLflow, and select a Champion using a predefined decision rule.

### Task and Dataset

- **Task:** Binary Sentiment Classification
- **Classes:** Positive and Negative
- **Dataset:** Lecturer-provided News Sentiment Analysis dataset
- **Original dataset size:** 3,500 records
- **Evaluation subset size:** 200 records
- **Positive examples:** 100
- **Negative examples:** 100
- **Evaluation inputs:** `data/eval_inputs.csv`
- **New prediction inputs:** `data/new_data.csv`
- **Random seed:** 42
- **Duplicate handling:** Duplicate URLs removed before sampling.
- **Neutral examples:** Excluded because both selected models perform binary sentiment classification.
- **Input text:** Combined news title and description.
- **Text preprocessing:** HTML entities decoded and whitespace normalized.

Both models receive exactly the same 200 evaluation texts in the same order.

### Model A — Initial Champion

- **Hugging Face Model ID:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Role:** Initial Champion
- **Architecture:** DistilBERT
- **Approximate model size:** 67 million parameters

### Model B — Challenger

- **Hugging Face Model ID:** `textattack/bert-base-uncased-SST-2`
- **Role:** Challenger
- **Architecture:** BERT
- **Approximate model size:** 110 million parameters
- **Special handling:** The model outputs `LABEL_0` and `LABEL_1`, which are mapped to negative and positive, respectively.

---

## 2. Project Structure

```text
1MS25CI077-mlflow-champion-challenger/
│
├── data/
│   ├── news_sentiment_analysis.csv
│   ├── eval_inputs.csv
│   └── new_data.csv
│
├── outputs/
│   ├── predictions.csv
│   ├── metrics_summary.json
│   ├── comparison_decision.json
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── disagreement_examples.csv
│   └── new_data_predictions.csv
│
├── screenshots/
│
├── prepare_data.py
├── evaluate.py
├── register_champion.py
├── predict_from_mlflow.py
├── requirements.txt
├── README.md
└── .gitignore
```

Local MLflow databases and artifact storage are excluded from Git using `.gitignore`.

---

## 3. How to Run

### Step 1: Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 2: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 3: Start the MLflow tracking server

Run this command in a separate terminal:

```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open the MLflow dashboard:

`http://127.0.0.1:5000`

### Step 4: Prepare the evaluation dataset

```powershell
python prepare_data.py
```

This generates the fixed 200-example evaluation dataset at `data/eval_inputs.csv`.

### Step 5: Evaluate both models

```powershell
python evaluate.py
```

This performs inference, calculates classification metrics, logs both models and their metrics to MLflow, generates comparison artifacts, and records the Champion decision.

### Step 6: Register models and assign aliases

```powershell
python register_champion.py
```

This registers both models under the same registered model name and assigns the `champion` and `challenger` aliases.

### Step 7: Predict on new data

```powershell
python predict_from_mlflow.py
```

This loads the Champion through its MLflow alias, reads the new examples from `data/new_data.csv`, generates predictions, and saves them in `outputs/new_data_predictions.csv`.

---

## 4. Champion–Challenger Decision Rule

The decision rule was defined before the final evaluation and was not changed after examining the results.

### Primary Metric

Macro-F1 is the primary metric because it calculates F1 independently for each class and averages the scores equally.

- Higher Macro-F1 is better.
- Lower inference latency is better.

### Minimum Improvement Margin

The minimum improvement margin is **0.01 (one percentage point)**.

### Tie-Breaker

- If Model B improves Macro-F1 over Model A by at least 0.01, Model B becomes Champion.
- If Model A improves Macro-F1 over Model B by at least 0.01, Model A remains Champion.
- If the absolute Macro-F1 difference is below 0.01, the faster model is selected using average inference latency.
- If latency is also tied, Model A remains Champion.

---

## 5. Evaluation Results

Both models were evaluated on the same 200 saved evaluation examples using CPU inference.

| Metric | Model A: DistilBERT | Model B: BERT | Difference (B - A) |
|---|---:|---:|---:|
| Accuracy | 0.4850 | 0.5500 | +0.0650 |
| Macro Precision | 0.4848 | 0.5501 | +0.0653 |
| Macro Recall | 0.4850 | 0.5500 | +0.0650 |
| Macro-F1 | 0.4834 | 0.5498 | +0.0664 |
| Average Latency (ms) | 47.5 | 91.6 | +44.1 |
| Total Inference Time (s) | 9.5035 | 18.3254 | +8.8219 |

### Evaluation Artifacts

- `outputs/predictions.csv`: Predictions and confidence scores for both models.
- `outputs/metrics_summary.json`: Classification metrics and inference timing.
- `outputs/comparison_decision.json`: Final Champion decision and supporting values.
- `outputs/confusion_matrices.png`: Confusion matrices for both models.
- `outputs/model_comparison.png`: Comparison of Accuracy and Macro-F1.
- `outputs/disagreement_examples.csv`: Five selected examples where the models disagree, with qualitative analyst comments.

A total of 27 disagreements were identified between the two models.

---

## 6. Final Champion Decision

**Selected Champion: Model B — `textattack/bert-base-uncased-SST-2`**

Model B, the Challenger, outperformed the initial Champion, Model A, on the 200-example evaluation dataset. Model B achieved a Macro-F1 score of 0.5498, compared with 0.4834 for Model A. The improvement was 0.0664, or 6.64 percentage points, exceeding the predeclared minimum improvement margin of 0.01. Model B also achieved higher accuracy (0.5500 compared with 0.4850), corresponding to 110 correct predictions versus 97 for Model A on the 200 examples.

However, Model B was slower. Its average inference latency was 91.6 milliseconds per example, compared with 47.5 milliseconds for Model A. Its total inference time was 18.3254 seconds, while Model A required 9.5035 seconds. Model B is also larger, with approximately 110 million parameters compared with approximately 67 million for Model A.

Despite these speed and size trade-offs, Model B was selected because the primary decision metric is Macro-F1, and its improvement exceeded the predefined threshold. The latency tie-breaker does not apply in this case.

The improvement is meaningful under the project's decision rule, but it should not be interpreted as proof of general superiority. The evaluation contains only 200 balanced examples, and no statistical significance test or confidence interval was calculated. Therefore, performance should be checked on a larger and more diverse dataset before using the model in a real-world application.

---

## 7. Limitations and Declaration

### Limitations

- The evaluation uses 200 examples rather than the complete original dataset.
- Neutral sentiment examples were excluded because the selected models perform binary classification.
- Both models were evaluated using a balanced subset of 100 positive and 100 negative examples.
- Model performance may vary with different evaluation samples.
- Inference latency may vary depending on hardware, system load, and execution environment.
- The results do not establish statistical significance or guarantee generalization to all news articles.
- The 27 disagreement examples show that the models can interpret the same news text differently.
- The dataset's original sentiment labels may not always be unambiguous from the available text.

### Declaration

This project is an individual academic mini-project.

AI assistance through ChatGPT was used for guidance, understanding concepts, debugging, code development support, and documentation preparation.

The implementation, experiment results, model comparison, and final decisions are to be reviewed and understood by the student. All code and design choices will be studied and explained during the viva.

---

## 8. MLflow Model Registry

**Registered Model Name:** `NewsSentimentClassifier`

| Version | Hugging Face Model | Alias |
|---|---|---|
| Version 1 | `distilbert-base-uncased-finetuned-sst-2-english` | `challenger` |
| Version 2 | `textattack/bert-base-uncased-SST-2` | `champion` |

Both models are registered under a common name, and predictions are loaded through the Champion alias:

```python
models:/NewsSentimentClassifier@champion
```

---

## 9. New-Data Predictions

The project uses a separate input dataset containing eight new news headlines:

`data/new_data.csv`

The `predict_from_mlflow.py` script loads the Champion through the MLflow registry alias and generates predictions without directly importing or calling the Hugging Face pipeline.

The output is saved as:

`outputs/new_data_predictions.csv`

The output includes the input text, predicted sentiment, confidence score, registered model version, and Hugging Face model ID.

---

**Project Repository:** `1MS25CI077-mlflow-champion-challenger`