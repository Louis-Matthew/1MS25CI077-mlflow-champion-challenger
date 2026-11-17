
import pandas as pd
from pathlib import Path
import html

# 1. Define file paths
input_file = Path("data/news_sentiment_analysis.csv")
output_file = Path("data/eval_inputs.csv")

# 2. Load the original dataset
df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

# 3. Remove duplicate news articles using their URLs
df = df.drop_duplicates(subset=["URL"]).copy()

# 4. Keep only positive and negative sentiment records
df["Sentiment"] = df["Sentiment"].str.strip().str.lower()

df = df[df["Sentiment"].isin(["positive", "negative"])].copy()

# 5. Check that enough records are available
class_counts = df["Sentiment"].value_counts()

print("\nAvailable records after filtering:")
print(class_counts)

if class_counts.get("positive", 0) < 100:
    raise ValueError("Not enough positive records.")

if class_counts.get("negative", 0) < 100:
    raise ValueError("Not enough negative records.")

# 6. Select 100 records from each sentiment class
positive = df[df["Sentiment"] == "positive"].sample(
    n=100, random_state=42
)

negative = df[df["Sentiment"] == "negative"].sample(
    n=100, random_state=42
)

# 7. Combine both classes and shuffle the records
evaluation_data = pd.concat(
    [positive, negative],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

# 8. Combine title and description into one text input
evaluation_data["text"] = (
    evaluation_data["Title"].fillna("").astype(str)
    + ". "
    + evaluation_data["Description"].fillna("").astype(str)
).str.strip()

# Decode HTML entities and normalize whitespace
evaluation_data["text"] = evaluation_data["text"].apply(
    lambda value: " ".join(html.unescape(value).split())
)

# 9. Prepare the final evaluation CSV
final_data = evaluation_data[
    ["text", "Sentiment", "URL", "Title", "Description"]
].rename(columns={"Sentiment": "label"})

# 10. Save the evaluation dataset
final_data.to_csv(output_file, index=False)

# 11. Display verification information
print("\nEvaluation dataset created successfully!")
print("Saved to:", output_file)
print("Total records:", len(final_data))

print("\nFinal sentiment distribution:")
print(final_data["label"].value_counts())

print("\nFirst evaluation example:")
print(final_data.iloc[0]["text"])
