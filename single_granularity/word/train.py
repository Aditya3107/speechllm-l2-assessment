#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Word-Level Pronunciation Scoring with Bounded-DPO (BDPO) + SFT.
Target Format: "WE/Excellent CALL/Bad IT/Good"
Features:
- BDPO Loss (Fixes Rejection Dominance)
- SFT Loss (Fixes Mode Collapse)
- Rubric-Based Prompting
- Robust Evaluation & Logging
"""

import os
import argparse
import csv
import re
import random
from dataclasses import dataclass  # <--- Added missing import
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import librosa
from scipy.stats import pearsonr

# HuggingFace & PEFT
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

try:
    from sklearn.metrics import cohen_kappa_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ==========================================
# 1. CONFIGURATION & UTILS
# ==========================================

LABELS = ["Worst", "Bad", "Average", "Good", "Excellent"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

def build_prompt(transcript: str) -> str:
    """
    Generates the rubric-based prompt for word-level inline scoring.
    """
    return (
        "Task: Rate the pronunciation of each word inline based on the following rubric.\n\n"
        "Rubric:\n"
        "- Excellent: The pronunciation of the whole word is correct.\n"
        "- Good: Most phones are correct, but has heavy accents.\n"
        "- Average: No more than 30% of phones are wrongly pronounced.\n"
        "- Bad: More than 30% of phones are wrong, or mispronounced as another word.\n"
        "- Worst: Hard to distinguish or the word is missed.\n\n"
        "Format: WORD/Label WORD/Label...\n"
        f"Transcript: \"{transcript}\"\n"
        "Verdict:"
    )

def parse_inline_labels(text):
    """
    Robust parser for 'WORD/Label' format.
    Handles extra spaces, punctuation, and mixed case.
    """
    labels = []
    # Replace newlines with spaces to treat as single sequence
    text = str(text).replace("\n", " ")
    
    # Regex matches: Any non-whitespace chars + forward slash + Alphabetic Label
    matches = re.findall(r"(\S+)/([A-Za-z]+)", text)
    
    for word, label in matches:
        # Canonize label
        clean_label = label.strip().capitalize()
        
        # Fix common model typos/synonyms
        if clean_label == "Fair": clean_label = "Average"
        if clean_label == "Poor": clean_label = "Bad"
        
        if clean_label in LABEL2ID:
            labels.append(clean_label)
            
    return labels

# ==========================================
# 2. DATASET & COLLATOR
# ==========================================

class WordLevelDPODataset(Dataset):
    def __init__(self, path, is_train=True):
        df = pd.read_csv(path)
        self.rows = []
        self.weights = [] 

        for _, r in df.iterrows():
            uttid = str(r.get("id", f"row_{_}"))
            audio = str(r["audio_path"])
            text = str(r["transcript"])

            if is_train:
                # Training Mode: DPO Pairs
                # Expects 'chosen' and 'rejected' columns with inline text
                if "chosen" not in r or "rejected" not in r:
                    continue
                
                chosen_str = str(r["chosen"])
                rejected_str = str(r["rejected"])
                
                # Default weight or read from CSV
                dpo_w = float(r.get("dpo_weight", 1.0))

                self.rows.append({
                    "uttid": uttid, "audio_path": audio, "transcript": text,
                    "chosen": chosen_str, "rejected": rejected_str, "dpo_weight": dpo_w
                })
                self.weights.append(1.0) 

            else:
                # Test Mode: Ground Truth
                # We try to find the ground truth in 'chosen', 'target', or 'label' columns
                target_str = str(r.get("target", r.get("chosen", "")))
                
                self.rows.append({
                    "uttid": uttid, "audio_path": audio, "transcript": text, "target": target_str
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

@dataclass
class DPOCollator:
    processor: AutoProcessor
    
    def __call__(self, batch):
        sr = self.processor.feature_extractor.sampling_rate
        audios, prompts = [], []
        chosen_texts, rejected_texts = [], []
        dpo_weights = []

        for item in batch:
            wav, _ = librosa.load(item["audio_path"], sr=sr, mono=True)
            audios.append(wav)
            
            prompt_raw = build_prompt(item["transcript"])
            msgs = [{"role": "user", "content": [{"type": "audio", "audio_url": "ptr"}, {"type": "text", "text": prompt_raw}]}]
            prompt_fmt = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            
            prompts.append(prompt_fmt)
            chosen_texts.append(prompt_fmt + item["chosen"])
            rejected_texts.append(prompt_fmt + item["rejected"])
            dpo_weights.append(item.get("dpo_weight", 1.0))

        # Tokenize Inputs
        enc_prompt = self.processor(text=prompts, audios=audios, sampling_rate=sr, return_tensors="pt", padding=True)
        prompt_lens = enc_prompt.attention_mask.sum(dim=1)

        enc_chosen = self.processor(text=chosen_texts, audios=audios, sampling_rate=sr, return_tensors="pt", padding=True)
        enc_rejected = self.processor(text=rejected_texts, audios=audios, sampling_rate=sr, return_tensors="pt", padding=True)

        # Create Labels (Masking the Prompt)
        def make_labels(input_ids, p_lens):
            labels = input_ids.clone()
            for i, L in enumerate(p_lens):
                labels[i, :L] = -100 
            labels[input_ids == self.processor.tokenizer.pad_token_id] = -100
            return labels

        labels_c = make_labels(enc_chosen.input_ids, prompt_lens)
        labels_r = make_labels(enc_rejected.input_ids, prompt_lens)

        return {
            "chosen": enc_chosen, "rejected": enc_rejected,
            "labels_c": labels_c, "labels_r": labels_r,
            "dpo_weights": torch.tensor(dpo_weights, dtype=torch.float32)
        }

# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================

def get_batch_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = (shift_labels != -100)
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    safe_labels = flat_labels.clone()
    safe_labels[safe_labels == -100] = 0
    selected_logps = torch.gather(log_probs, 1, safe_labels.unsqueeze(1)).squeeze(1)
    selected_logps = selected_logps.view(shift_labels.shape)
    masked_logps = selected_logps * loss_mask
    return masked_logps.sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)

def get_sft_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

def bdpo_loss(policy_c_logps, policy_r_logps, ref_c_logps, ref_r_logps, weights, beta=0.1, alpha=0.5):
    log_ratio_c = policy_c_logps - ref_c_logps
    log_ratio_r = policy_r_logps - ref_r_logps

    alpha_t = torch.tensor(alpha, device=policy_c_logps.device, dtype=policy_c_logps.dtype)
    alpha_t = torch.clamp(alpha_t, min=1e-6, max=1.0 - 1e-6)

    # Log-Sum-Exp for numerical stability: log( (1-a)*exp(log_r) + a )
    term_a = torch.log(1.0 - alpha_t) + log_ratio_r
    term_b = torch.log(alpha_t)
    bounded_log_ratio_r = torch.logaddexp(term_a, term_b)

    logits = beta * log_ratio_c - beta * bounded_log_ratio_r
    losses = -F.logsigmoid(logits) * weights
    
    with torch.no_grad():
        rewards_chosen = beta * log_ratio_c.detach()
        rewards_rejected = beta * log_ratio_r.detach()
        rewards_margin = rewards_chosen - rewards_rejected

    return losses.mean(), rewards_margin.mean()

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sft_weight", type=float, default=1.0)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    log_file = os.path.join(args.out_dir, "training_log.csv")
    with open(log_file, "w") as f:
        csv.writer(f).writerow(["epoch", "loss", "margin"])

    print(f"[INIT] BDPO+SFT | Alpha={args.alpha} | Beta={args.beta} | SFT_W={args.sft_weight}")

    # Load Model
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant_config, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False 
    
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=128, target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Datasets
    train_ds = WordLevelDPODataset(args.train_csv, is_train=True)
    
    collator = DPOCollator(processor)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    model.train()
    
    for epoch in range(args.num_epochs):
        print(f"\n[EPOCH] {epoch+1}/{args.num_epochs}")
        pbar = tqdm(loader, desc=f"Ep {epoch+1}")
        stats = {"loss": 0, "margin": 0, "steps": 0}
        
        for step, batch in enumerate(pbar):
            def to_dev(d): return {k: v.to(device) for k, v in d.items() if torch.is_tensor(v)}
            batch_c = to_dev(batch["chosen"])
            batch_r = to_dev(batch["rejected"])
            labels_c = batch["labels_c"].to(device)
            labels_r = batch["labels_r"].to(device)
            dpo_w = batch["dpo_weights"].to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # 1. Forward
                out_pc = model(**batch_c)
                out_pr = model(**batch_r)
                
                with torch.no_grad():
                    with model.disable_adapter():
                        out_rc = model(**batch_c)
                        out_rr = model(**batch_r)
                
                # 2. Log Probs
                pol_logp_c = get_batch_logps(out_pc.logits, labels_c)
                pol_logp_r = get_batch_logps(out_pr.logits, labels_r)
                ref_logp_c = get_batch_logps(out_rc.logits, labels_c)
                ref_logp_r = get_batch_logps(out_rr.logits, labels_r)

                # 3. BDPO Loss
                loss_dpo, r_m = bdpo_loss(
                    pol_logp_c, pol_logp_r, ref_logp_c, ref_logp_r,
                    weights=dpo_w, beta=args.beta, alpha=args.alpha
                )

                # 4. SFT Loss
                loss_sft = get_sft_loss(out_pc.logits, labels_c)

                # Combine
                loss = (loss_dpo + args.sft_weight * loss_sft) / args.grad_accum
            
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                stats["loss"] += loss.item() * args.grad_accum
                stats["margin"] += r_m.item()
                stats["steps"] += 1
                pbar.set_postfix({"L": f"{loss.item()*args.grad_accum:.3f}", "M": f"{r_m.item():.3f}"})

        # Save Checkpoint
        ckpt_dir = os.path.join(args.out_dir, f"epoch_{epoch+1}")
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        
        avg_loss = stats["loss"] / max(1, stats["steps"])
        avg_margin = stats["margin"] / max(1, stats["steps"])
        print(f"[EPOCH {epoch+1}] loss={avg_loss:.4f}  margin={avg_margin:.4f}")

        with open(log_file, "a") as f:
            csv.writer(f).writerow([epoch+1, f"{avg_loss:.4f}", f"{avg_margin:.4f}"])

    print("\n[DONE] Finished.")

if __name__ == "__main__":
    main()