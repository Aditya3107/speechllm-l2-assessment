import os
import argparse
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import librosa
import pandas as pd
from dataclasses import dataclass
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import build_comprehensive_prompt

# --- DATASET & COLLATOR ---
class ComprehensiveDataset(Dataset):
    def __init__(self, path, is_train=True):
        df = pd.read_csv(path)
        self.rows = []
        for _, r in df.iterrows():
            if is_train:
                if "chosen" not in r or "rejected" not in r: continue
                self.rows.append({
                    "audio_path": r["audio_path"],
                    "transcript": r["transcript"],
                    "target_phonemes": str(r["target_phonemes"]),
                    "chosen": str(r["chosen"]),
                    "rejected": str(r["rejected"]),
                    "dpo_weight": float(r.get("dpo_weight", 1.0))
                })
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

@dataclass
class ComprehensiveCollator:
    processor: AutoProcessor
    def __call__(self, batch):
        sr = self.processor.feature_extractor.sampling_rate
        audios, prompts, chosen_texts, rejected_texts, dpo_weights = [], [], [], [], []
        
        for item in batch:
            try:
                wav, _ = librosa.load(item["audio_path"], sr=sr, mono=True)
            except Exception:
                wav = torch.zeros(sr) # Fallback for corrupted audio
                
            audios.append(wav)
            p = build_comprehensive_prompt(item["transcript"], item["target_phonemes"])
            
            msgs = [{"role": "user", "content": [{"type": "audio", "audio_url": "ptr"}, {"type": "text", "text": p}]}]
            p_fmt = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            
            prompts.append(p_fmt)
            chosen_texts.append(p_fmt + item["chosen"])
            rejected_texts.append(p_fmt + item["rejected"])
            dpo_weights.append(item["dpo_weight"])

        enc_chosen = self.processor(text=chosen_texts, audios=audios, sampling_rate=sr, return_tensors="pt", padding=True)
        enc_rejected = self.processor(text=rejected_texts, audios=audios, sampling_rate=sr, return_tensors="pt", padding=True)
        
        # Calculate prompt lengths to mask them out in loss
        enc_prompt = self.processor(text=prompts, audios=audios, sampling_rate=sr, return_tensors="pt", padding=True)
        prompt_lens = enc_prompt.attention_mask.sum(dim=1)

        def make_labels(enc, plens):
            labels = enc.input_ids.clone()
            for i, l in enumerate(plens): labels[i, :l] = -100
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return labels

        return {
            "chosen": enc_chosen, "rejected": enc_rejected,
            "labels_c": make_labels(enc_chosen, prompt_lens),
            "labels_r": make_labels(enc_rejected, prompt_lens),
            "dpo_weights": torch.tensor(dpo_weights, dtype=torch.float32)
        }

# --- LOSS FUNCTIONS ---
def get_batch_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = (shift_labels != -100)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    selected_logps = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
    return (selected_logps * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)

def bdpo_loss(p_c_logps, p_r_logps, r_c_logps, r_r_logps, weights, beta=0.1, alpha=0.5):
    log_ratio_c = p_c_logps - r_c_logps
    log_ratio_r = p_r_logps - r_r_logps
    alpha_t = torch.tensor(alpha, device=p_c_logps.device)
    bounded_log_ratio_r = torch.logaddexp(torch.log(1.0 - alpha_t) + log_ratio_r, torch.log(alpha_t))
    losses = -F.logsigmoid(beta * (log_ratio_c - bounded_log_ratio_r)) * weights
    return losses.mean(), (beta * (log_ratio_c - log_ratio_r)).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sft_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # --- IMPORTANT: Set to 0. This maps to the first VISIBLE gpu (which is physical GPU 4)
    device = torch.device("cuda:0") 
    
    # Initialize Log File
    log_file = os.path.join(args.out_dir, "training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            csv.writer(f).writerow(["epoch", "loss", "margin"])

    print(f"[INIT] Loading Model on {device} (Physical GPU 4)...")
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": device}, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=128, target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = ComprehensiveDataset(args.train_csv, is_train=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ComprehensiveCollator(processor), drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        model.train()
        stats = {"loss": 0, "margin": 0, "steps": 0}
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            def to_d(d): return {k: v.to(device) for k, v in d.items() if torch.is_tensor(v)}
            b_c, b_r = to_d(batch["chosen"]), to_d(batch["rejected"])
            l_c, l_r = batch["labels_c"].to(device), batch["labels_r"].to(device)
            dpo_w = batch["dpo_weights"].to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Policy
                logits_pc = model(**b_c).logits
                logits_pr = model(**b_r).logits
                
                # Reference (Frozen)
                with torch.no_grad():
                    with model.disable_adapter():
                        logits_rc = model(**b_c).logits
                        logits_rr = model(**b_r).logits
                
                p_lp_c = get_batch_logps(logits_pc, l_c)
                p_lp_r = get_batch_logps(logits_pr, l_r)
                r_lp_c = get_batch_logps(logits_rc, l_c)
                r_lp_r = get_batch_logps(logits_rr, l_r)
                
                loss_dpo, margin = bdpo_loss(p_lp_c, p_lp_r, r_lp_c, r_lp_r, dpo_w, args.beta, args.alpha)
                loss_sft = F.cross_entropy(logits_pc[..., :-1, :].reshape(-1, logits_pc.size(-1)), l_c[..., 1:].reshape(-1), ignore_index=-100)
                
                loss = (loss_dpo + args.sft_weight * loss_sft) / args.grad_accum
            
            loss.backward()
            
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
                stats["loss"] += loss.item() * args.grad_accum
                stats["margin"] += margin.item()
                stats["steps"] += 1
                pbar.set_postfix({"L": f"{loss.item()*args.grad_accum:.3f}", "M": f"{margin.item():.3f}"})

        # Save Checkpoint
        ckpt_dir = os.path.join(args.out_dir, f"epoch_{epoch+1}")
        model.save_pretrained(ckpt_dir); processor.save_pretrained(ckpt_dir)
        
        avg_l = stats["loss"]/max(1, stats["steps"])
        avg_m = stats["margin"]/max(1, stats["steps"])
        print(f"[EPOCH {epoch+1}] loss={avg_l:.4f}  margin={avg_m:.4f}")

        with open(log_file, "a") as f:
            csv.writer(f).writerow([epoch+1, f"{avg_l:.4f}", f"{avg_m:.4f}"])

if __name__ == "__main__":
    main()