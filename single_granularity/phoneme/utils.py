import re
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_squared_error
from scipy.stats import pearsonr

# --- Constants & Mappings ---
LABELS = ["Worst", "Bad", "Average", "Good", "Excellent"]
LABEL2ID = {
    "Worst": 0,
    "Bad": 1, "Poor": 1,
    "Average": 2, "Fair": 2,
    "Good": 3,
    "Excellent": 4
}
ID2LABEL = {v: k for k, v in LABEL2ID.items() if k in LABELS}

# --- Prompt Builder ---
def build_comprehensive_prompt(transcript, target_phonemes):
    """
    Constructs the input prompt for the Audio-LLM.
    """
    return (
        f"Transcript: {transcript}\n"
        f"Target Phonemes: {target_phonemes}\n\n"
        "Assess the audio quality based on the transcript and phonemes. "
        "Provide a comprehensive assessment including:\n"
        "1. Sentence-level Accuracy, Fluency, and Prosody ratings (Excellent, Good, Average, Bad, Poor).\n"
        "2. Word-level accuracy for each word.\n"
        "3. Phoneme-level accuracy for each phoneme.\n"
        "4. A final Verdict explaining the assessment."
    )

# --- Output Parser ---
def parse_comprehensive_output(text):
    """
    Parses the model's generated text into a structured dictionary.
    Handles Sentence metrics, Words, Phonemes, and the Verdict.
    """
    data = {}
    text = str(text)
    
    # 1. Extract Sentence Level Metrics
    # Looks for "Accuracy: Good", "Fluency: Average", etc.
    for key in ['Accuracy', 'Fluency', 'Prosody']:
        match = re.search(f"{key}:\\s*([A-Za-z]+)", text, re.IGNORECASE)
        if match: 
            data[key] = match.group(1).strip().capitalize()
        else:
            data[key] = None

    # 2. Extract Sequence Level Metrics (Words & Phonemes)
    # Looks for "Words: WORD/Label WORD/Label"
    for key in ['Words', 'Phonemes']:
        match = re.search(f"{key}:\\s*(.*)", text)
        if match:
            seq_str = match.group(1).strip()
            labels = []
            # Split by whitespace, then look for the slash
            for item in seq_str.split():
                if '/' in item:
                    try:
                        # Extract the label part (e.g., "Excellent" from "CAT/Excellent")
                        label = item.rsplit('/', 1)[1].strip()
                        # Clean punctuation if present
                        label = re.sub(r'[^\w]', '', label)
                        labels.append(label)
                    except IndexError:
                        pass
            data[key] = labels
        else:
            data[key] = []
    
    # 3. Extract Verdict
    # Looks for "Verdict:", "Overall,", "Explanation:", or just text at the end
    match_verdict = re.search(r"(Overall,|Verdict:|Explanation:)(.*)", text, re.DOTALL | re.IGNORECASE)
    if match_verdict:
        data['Verdict'] = match_verdict.group(2).strip()
    else:
        # Fallback: Assume everything after the Phonemes line is the verdict
        lines = text.split('\n')
        found_phonemes = False
        verdict_lines = []
        for line in lines:
            if line.strip().startswith("Phonemes:"):
                found_phonemes = True
                continue
            if found_phonemes:
                verdict_lines.append(line)
        
        data['Verdict'] = " ".join(verdict_lines).strip()
                
    return data

# --- Metrics Calculator ---
def calculate_metrics(y_true, y_pred):
    """
    Calculates Accuracy, Pearson Correlation (PCC), Quadratic Weighted Kappa (QWK), and MSE.
    """
    if not y_true or not y_pred:
        return {'acc': 0.0, 'pcc': 0.0, 'qwk': 0.0, 'mse': 0.0}

    # Ensure inputs are numpy arrays of integers
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)
    
    # Handle length mismatch by truncating to the shorter length
    # (This happens if model generates fewer/more phonemes than target)
    min_len = min(len(y_t), len(y_p))
    y_t = y_t[:min_len]
    y_p = y_p[:min_len]

    if len(y_t) == 0:
        return {'acc': 0.0, 'pcc': 0.0, 'qwk': 0.0, 'mse': 0.0}

    acc = accuracy_score(y_t, y_p)
    mse = mean_squared_error(y_t, y_p)

    # PCC requires variance in input
    if len(y_t) > 1 and np.std(y_p) > 0 and np.std(y_t) > 0:
        pcc, _ = pearsonr(y_t, y_p)
    else:
        pcc = 0.0

    qwk = cohen_kappa_score(y_t, y_p, weights="quadratic")
    
    return {'acc': acc, 'pcc': pcc, 'qwk': qwk, 'mse': mse}