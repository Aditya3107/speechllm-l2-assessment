import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm
import time

# --- Dataset & Collator (Same as optimized version) ---
class EvalDataset(Dataset):
    def __init__(self, df, processor):
        self.rows = df.to_dict('records')
        self.processor = processor
        self.sr = processor.feature_extractor.sampling_rate

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            wav, _ = librosa.load(row["audio_path"], sr=self.sr, mono=True)
        except Exception as e:
            print(f"Error loading {row['audio_path']}: {e}")
            wav = np.zeros(self.sr)
        
        return {
            "audio": wav,
            "transcript": row["transcript"],
            "target_phonemes": str(row["target_phonemes"]),
            "audio_path": row["audio_path"],
            "target": row.get("target", row.get("chosen", ""))
        }

class EvalCollator:
    def __init__(self, processor):
        self.processor = processor
        from utils import build_comprehensive_prompt
        self.prompt_fn = build_comprehensive_prompt

    def __call__(self, batch):
        audios = [item["audio"] for item in batch]
        texts = []
        for item in batch:
            p = self.prompt_fn(item["transcript"], item["target_phonemes"])
            msgs = [{"role": "user", "content": [{"type": "audio", "audio_url": "ptr"}, {"type": "text", "text": p}]}]
            text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            texts.append(text)
            
        batch_inputs = self.processor(
            text=texts, audios=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, 
            return_tensors="pt", padding=True
        )
        return batch_inputs, batch

# --- Worker Function ---
def run_shard(rank, gpu_id, shard_df, model_path, out_file, batch_size):
    device = f"cuda:{rank}" # Relative index for this process
    print(f"[Worker] Rank {rank} (Physical GPU?) starting on {len(shard_df)} samples...")
    
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        processor.tokenizer.padding_side = 'left'
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, quantization_config=bnb_config, device_map={"": device}, torch_dtype=torch.bfloat16
        )
        model.eval()
        
        dataset = EvalDataset(shard_df, processor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=EvalCollator(processor), num_workers=4)
        
        results = []
        for inputs, original_items in tqdm(loader, desc=f"GPU {rank}"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=450, do_sample=False)
            
            input_len = inputs["input_ids"].shape[1]
            decoded = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            
            for i, text in enumerate(decoded):
                results.append({
                    "audio_path": original_items[i]["audio_path"],
                    "target": original_items[i]["target"],
                    "pred": text.strip()
                })
        
        pd.DataFrame(results).to_csv(out_file, index=False)
        
    except Exception as e:
        print(f"[CRITICAL] Worker {rank} crashed: {e}")

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Dynamic GPU Detection
    num_gpus = torch.cuda.device_count()
    print(f"[SYSTEM] Detected {num_gpus} available GPUs for evaluation.")
    
    df = pd.read_csv(args.test_csv)
    shards = np.array_split(df, num_gpus)
    
    mp.set_start_method('spawn', force=True)
    processes = []
    tmp_paths = [os.path.join(args.out_dir, f"tmp_ep{args.epoch}_r{i}.csv") for i in range(num_gpus)]
    
    for i in range(num_gpus):
        if os.path.exists(tmp_paths[i]): os.remove(tmp_paths[i])
        p = mp.Process(target=run_shard, args=(i, i, shards[i], args.model_path, tmp_paths[i], args.batch_size))
        p.start()
        processes.append(p)
        
    for p in processes: p.join()

    # Aggregate
    valid_dfs = [pd.read_csv(p) for p in tmp_paths if os.path.exists(p)]
    if not valid_dfs: return

    all_res = pd.concat(valid_dfs)
    all_res.to_csv(os.path.join(args.out_dir, f"preds_epoch_{args.epoch}.csv"), index=False)
    
    # Simple cleanup
    for p in tmp_paths:
        if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    main()