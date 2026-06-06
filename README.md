# A Finetuned SpeechLLM for Joint Multi-Granular L2 Assessment and Natural-Language Rationales

**Aditya Kamlesh Parikh, Cristian Tejedor Garcia, Catia Cucchiarini, Helmer Strik**  
Centre for Language Studies, Radboud University, Nijmegen, The Netherlands  
*Interspeech 2026*

---

## Overview

We propose a rubric-guided SpeechLLM for multi-aspect, multi-granular L2 pronunciation assessment trained with a hybrid objective combining Supervised Fine-Tuning (SFT) and Bounded Direct Preference Optimization (BDPO). The model jointly predicts ordinal proficiency labels at three granularities and generates a natural-language rationale in a single response:

- **Sentence-level**: Accuracy, Fluency, Prosody
- **Word-level**: Accuracy (inline per word)
- **Phoneme-level**: Accuracy (inline per phoneme)
- **Rationale**: Free-text justification grounded in the predicted labels

The backbone is [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct), fine-tuned with LoRA (r=64) under 4-bit quantization. We evaluate on the [SpeechOcean762](https://www.openslr.org/111/) dataset.

---

## Repository Structure

```
├── single_granularity/          # Single-granularity baselines (BDPO-S)
│   ├── sentence/                # Sentence-level models (Accuracy / Fluency / Prosody)
│   │   ├── train_accuracy.py
│   │   ├── train_fluency.py
│   │   ├── train_prosody.py
│   │   ├── train_resume.py      # Resume from checkpoint
│   │   ├── run.sh
│   │   ├── run_resume.sh
│   │   └── data/
│   ├── word/                    # Word-level accuracy model
│   │   ├── train.py
│   │   ├── run.sh
│   │   └── data/
│   └── phoneme/                 # Phoneme-level accuracy model
│       ├── train.py
│       ├── train_resume.py
│       ├── utils.py
│       ├── eval.py              # Multi-GPU batched evaluation
│       ├── run.sh
│       ├── run_eval.sh
│       └── data/
│
├── multi_granularity/           # Joint multi-granular model (BDPO-M) — main contribution
│   ├── train.py                 # Joint training: Sentence + Word + Phoneme + Rationale
│   ├── train_resume.py          # Resume from checkpoint
│   ├── utils.py                 # Prompt builder, output parser, metrics
│   ├── eval.py                  # Multi-GPU batched evaluation
│   ├── run.sh
│   ├── run_resume.sh
│   └── data/
│       ├── train_multitask_comprehensive_v2.csv
│       └── test_multitask_comprehensive_v2.csv
│
├── evaluation/                  # Rationale reliability analysis (post-hoc)
│   ├── eval_rationale_sentiment_llm.py      # LLM-based sentiment → Table 4
│   ├── eval_rationale_sentiment_roberta.py  # RoBERTa-based sentiment → Table 4
│   └── eval_rationale_mentions.py           # Word/phoneme mention extraction → Table 5
│
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

> **Note:** Requires a CUDA-capable GPU with at least 48GB VRAM for training (used NVIDIA RTX A6000). Inference can run on smaller GPUs with 4-bit quantization enabled.

---

## Data

This work uses the **SpeechOcean762** dataset ([openslr.org/111](https://www.openslr.org/111/)), which contains 5000 English read-speech utterances (2500 train / 2500 test) with sentence-, word-, and phoneme-level human annotations.

The `data/` folders in each subdirectory contain pre-formatted DPO training CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `audio_path` | Absolute path to the `.wav` file |
| `transcript` | Orthographic transcript |
| `target_phonemes` | Target phoneme sequence (multi-granular and phoneme models) |
| `chosen` | Ground-truth structured label response |
| `rejected` | Synthesized perturbed label response (Section 2.3.1 of paper) |

> Update all `audio_path` entries and `--out_dir` arguments to match your local file system before running.

---

## Training

All training scripts use the hybrid loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BDPO}} + \lambda \cdot \mathcal{L}_{\text{SFT}}, \quad \lambda = 1.0$$

with BDPO hyperparameters β = 0.1, α = 0.5. Effective batch size = 16 (batch\_size=1, grad\_accum=16).

---

### Single-Granularity Models (BDPO-S)

Each model is fine-tuned independently for a single dimension.


---

### Multi-Granular Model (BDPO-M)

Trains a single model jointly on all three granularities. Training runs on one GPU; evaluation is dispatched asynchronously to additional GPUs after each epoch.

---

#### Model Output Format

Each utterance produces a single structured response:

```
Accuracy: Good
Fluency: Excellent
Prosody: Excellent
Words: WE/Excellent CALL/Excellent IT/Excellent BEAR/Average
Phonemes: w/Excellent i/Excellent k/Excellent ɔ/Good l/Good ...
Overall, the speech is mostly accurate. Minor phoneme-level errors are
present in the word BEAR — the /ɛ/ and /ɹ/ phonemes are slightly
mispronounced.
```

The free-text rationale is generated by the base model's instruction-following capability prompted by the `Verdict:` cue, it is not explicitly supervised during training.

---

## Rationale Reliability Analysis

Rationale evaluation scripts operate on the prediction CSV produced by `eval.py` (e.g., `runs/mg_bdpo/preds_epoch_11.csv`). Update the `input_file` path inside each script before running.

### Sentiment Consistency

Classifies each rationale as Positive/Neutral/Negative and compares against the polarity of the model's predicted labels (Internal) and ground-truth labels (External).

**LLM-based** (Qwen2.5-7B-Instruct, multi-GPU parallel):

```bash
cd evaluation
python eval_rationale_sentiment_llm.py
```

**RoBERTa-based** (cardiffnlp/twitter-roberta-base-sentiment, CPU/single GPU):

```bash
python eval_rationale_sentiment_roberta.py
```

### Mention-Based Faithfulness — Table 5

Extracts specific words and phonemes explicitly mentioned in the rationale and measures agreement with Internal (Pred) and External (GT) labels.

```bash
python eval_rationale_mentions.py
```

---

## Citation

Soon on arxiv.

---

## Acknowledgements

This publication is part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research programme NGF AiNed Fellowship Grants, financed by the Dutch Research Council (NWO).
