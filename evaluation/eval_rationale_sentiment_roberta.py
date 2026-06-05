import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from scipy.special import softmax
from tqdm import tqdm
import os

CACHE_DIR = "/vol/tensusers8/aparikh/dpoftllm/cache_dir" 
os.environ["HF_HOME"] = CACHE_DIR
# 1. Load Data
df = pd.read_csv('/vol/tensusers8/aparikh/dpoftllm/Multi-Granular/eval/processed_results.csv')

# Ensure we have the target labels to compare against
# We need to parse the 'target' column to get the Ground Truth (Accuracy Label)
def get_target_accuracy(text):
    if not isinstance(text, str): return None
    for line in text.split('\n'):
        if line.startswith('Accuracy:'):
            return line.split(':', 1)[1].strip()
    return None

df['gt_label'] = df['target'].apply(get_target_accuracy)

# 2. Define Mappings
# Map Ground Truth Labels to Sentiment
gt_to_sentiment = {
    'Excellent': 'Positive', 'Good': 'Positive',
    'Average': 'Neutral', 'Fair': 'Neutral',
    'Bad': 'Negative', 'Poor': 'Negative', 'Worst': 'Negative'
}

df['gt_sentiment'] = df['gt_label'].map(gt_to_sentiment)

# 3. Load Model (CardiffNLP Twitter-RoBERTa)
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print(f"Loading model: {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# 4. Run Inference on 'explanation' column
# Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
def predict_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return None
    
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_input)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    top_label_id = ranking[0]
    
    mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return mapping[top_label_id]

print("Running sentiment analysis on explanations...")
tqdm.pandas()
df['roberta_sentiment'] = df['explanation'].progress_apply(predict_sentiment)

# 5. Calculate Metrics
# Filter out empty/invalid rows
valid_df = df.dropna(subset=['gt_sentiment', 'roberta_sentiment'])

total = len(valid_df)
matches = (valid_df['gt_sentiment'] == valid_df['roberta_sentiment']).sum()
match_rate = matches / total

print("\n" + "="*30)
print("RESULTS: RoBERTa vs Ground Truth")
print("="*30)
print(f"Total Samples: {total}")
print(f"Matches:       {matches}")
print(f"Match Rate:    {match_rate:.2%}")

# 6. Confusion Matrix
print("\nConfusion Matrix:")
confusion = pd.crosstab(valid_df['gt_sentiment'], valid_df['roberta_sentiment'], 
                        rownames=['Ground Truth'], colnames=['RoBERTa Prediction'])
print(confusion)

# 7. Per-Class Accuracy
print("\nPer-Class Recall:")
for sentiment in ['Negative', 'Neutral', 'Positive']:
    if sentiment in confusion.index:
        correct = confusion.loc[sentiment, sentiment]
        total_class = confusion.loc[sentiment].sum()
        print(f"{sentiment}: {correct/total_class:.2%} ({correct}/{total_class})")

# Save detailed results
df.to_csv('results_with_sentiment.csv', index=False)
print("\nSaved detailed results to 'results_with_sentiment.csv'")