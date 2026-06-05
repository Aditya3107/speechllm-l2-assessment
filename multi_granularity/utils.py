import re
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score

LABELS = ["Worst", "Bad", "Average", "Good", "Excellent"]

LABEL2ID = {
    # Standard Rubric
    "Worst": 0, 
    "Bad": 1, "Poor": 1,      
    "Average": 2, "Fair": 2, 
    "Good": 3, 
    "Excellent": 4,
    
    # Word Stress Binary Rubric (Mapped to 0/1 for binary metrics)
    "Incorrect": 0,
    "Correct": 1
}

def build_comprehensive_prompt(transcript, phoneme_seq):
    return (
        "Task: Assess the speech comprehensively based on the following rubrics.\n\n"
        "1. Sentence Accuracy:\n"
        "- Excellent: The pronunciation of the whole sentence is correct.\n"
        "- Good: Most words are correct, but has heavy accents.\n"
        "- Average: No more than 30% of words are wrongly pronounced.\n"
        "- Bad: More than 30% of words are wrong.\n"
        "- Worst: Hard to distinguish or mostly missed.\n\n"
        "2. Fluency:\n"
        "- Excellent: Native-like, no unnecessary pauses.\n"
        "- Good: Smooth, but with some pauses or hesitations.\n"
        "- Average: Noticeable pauses breaking the flow.\n"
        "- Bad: Disjointed speech, very slow.\n"
        "- Worst: Utterance is barely coherent.\n\n"
        "3. Prosody:\n"
        "- Excellent: Correct intonation, stress, and rhythm.\n"
        "- Good: Mostly correct, minor monotone or stress errors.\n"
        "- Average: Flat intonation or incorrect stress patterns.\n"
        "- Bad: Robotic or completely wrong rhythm.\n"
        "- Worst: Unnatural, hard to follow.\n\n"
        "4. Word Accuracy:\n"
        "- Rate each word inline (Excellent/Good/Average/Bad/Worst) based on phone errors.\n\n"
        "5. Word Stress: \n"
        "- Correct: The stress position is correct, or the word is a mono-syllable word \n"
        "- Incorrect:  The stress position is incorrect \n\n"
        "6. Phoneme Accuracy:\n"
        "- Rate each phoneme (Excellent/Good/Average/Bad/Worst).\n"
        "  (Excellent=1.9-2.0, Good=1.5-1.8, Average=1.0-1.4, Bad=0.5-0.9, Worst=0.0-0.4)\n\n"
        f"Transcript: \"{transcript}\"\n"
        f"Target Phonemes: {phoneme_seq}\n\n"
        "Output Format:\n"
        "Accuracy: Label\n"
        "Fluency: Label\n"
        "Prosody: Label\n"
        "Words: WORD/Label WORD/Label ...\n"
        "Word Stress: WORD/Label WORD/Label ...\n"
        "Phonemes: p/Label p/Label ...\n"
        "Verdict:"
    )

def parse_comprehensive_output(text):
    data = {}
    text = str(text)
    
    # 1. Parse Sentence Metrics
    for key in ['Accuracy', 'Fluency', 'Prosody']:
        match = re.search(f"{key}:\\s*([A-Za-z]+)", text)
        if match: data[key] = match.group(1).strip()
            
    # 2. Parse Sequence Metrics (Added 'Word Stress' here!)
    for key in ['Words', 'Word Stress', 'Phonemes']:
        match = re.search(f"{key}:\\s*(.*)", text)
        if match:
            seq_str = match.group(1).strip()
            labels = []
            for item in seq_str.split():
                if '/' in item:
                    try:
                        # Take the part after the last slash
                        labels.append(item.rsplit('/', 1)[1].strip())
                    except IndexError:
                        pass
            data[key] = labels
            
    return data

def calculate_metrics(y_true, y_pred):
    if not y_true or not y_pred: 
        return {'acc': 0.0, 'pcc': 0.0, 'qwk': 0.0, 'mse': 0.0}
        
    y_t, y_p = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    
    acc = accuracy_score(y_t, y_p)
    mse = mean_squared_error(y_t, y_p)
    
    # Pearson requires variance; handle constant input (e.g., if everything is "Correct")
    if len(y_t) > 1 and np.std(y_p) > 0 and np.std(y_t) > 0:
        pcc, _ = pearsonr(y_t, y_p)
    else:
        pcc = 0.0
        
    qwk = cohen_kappa_score(y_t, y_p, weights="quadratic")
    
    return {'acc': acc, 'pcc': pcc, 'qwk': qwk, 'mse': mse}