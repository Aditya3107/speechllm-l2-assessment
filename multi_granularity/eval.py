import os,csv
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

# Define a Dataset class for efficient loading
class EvalDataset(Dataset):
    def __init__(self, df, processor):
        self.rows = df.to_dict('records')
        self.processor = processor
        self.sr = processor.feature_extractor.sampling_rate

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            # Load audio
            wav, _ = librosa.load(row["audio_path"], sr=self.sr, mono=True)
        except Exception as e:
            # Fallback for corrupted audio
            print(f"Error loading {row['audio_path']}: {e}")
            wav = np.zeros(self.sr) # 1 sec silence
        
        return {
            "audio": wav,
            "transcript": row["transcript"],
            "target_phonemes": str(row["target_phonemes"]),
            "audio_path": row["audio_path"],
            "target": row.get("target", row.get("chosen", ""))
        }

# Collator to handle batching and padding
class EvalCollator:
    def __init__(self, processor):
        self.processor = processor
        # Import prompt builder here to avoid pickling issues with mp
        from utils import build_comprehensive_prompt
        self.prompt_fn = build_comprehensive_prompt

    def __call__(self, batch):
        audios = [item["audio"] for item in batch]
        texts = []
        
        # Prepare prompts
        for item in batch:
            p = self.prompt_fn(item["transcript"], item["target_phonemes"])
            msgs = [{"role": "user", "content": [{"type": "audio", "audio_url": "ptr"}, {"type": "text", "text": p}]}]
            text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            texts.append(text)
            
        # Processor handles padding of both audio features and text tokens
        batch_inputs = self.processor(
            text=texts, 
            audios=audios, 
            sampling_rate=self.processor.feature_extractor.sampling_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        return batch_inputs, batch

def run_shard(rank, gpu_id, shard_df, model_path, out_file, batch_size):
    device = f"cuda:{gpu_id}"
    print(f"[Worker] GPU index {gpu_id} (Physical {gpu_id+4}) starting on {len(shard_df)} samples with Batch Size {batch_size}...")
    
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        # IMPORTANT: Set padding side to left for batched generation
        processor.tokenizer.padding_side = 'left'
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, 
            quantization_config=bnb_config,
            device_map={"": device}, 
            torch_dtype=torch.bfloat16
        )
        model.eval()
        
        # Create DataLoader
        dataset = EvalDataset(shard_df, processor)
        collator = EvalCollator(processor)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collator, 
            num_workers=4,        # Preload data on CPU
            pin_memory=True
        )
        
        results = []
        
        for inputs, original_items in tqdm(loader, desc=f"GPU {gpu_id}"):
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Batch Generation
                    outputs = model.generate(**inputs, max_new_tokens=450, do_sample=False)
                
                # Decode batch
                input_len = inputs["input_ids"].shape[1]
                generated_ids = outputs[:, input_len:]
                decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Store results
                for i, text in enumerate(decoded_texts):
                    results.append({
                        "audio_path": original_items[i]["audio_path"],
                        "target": original_items[i]["target"],
                        "pred": text.strip()
                    })
                    
            except Exception as e:
                print(f"[Error] GPU {gpu_id} batch failed: {e}")
                continue
                
        pd.DataFrame(results).to_csv(out_file, index=False)
        print(f"[Worker] GPU {gpu_id} finished.")
        
    except Exception as e:
        print(f"[CRITICAL] GPU {gpu_id} worker crashed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--batch_size", type=int, default=8) # Default to 8
    args = parser.parse_args()

    # Load and Split Data
    df = pd.read_csv(args.test_csv)
    
    # Indices relative to CUDA_VISIBLE_DEVICES=4,5,6,7
    # 0 is training (busy), so use 1, 2, 3
    gpu_ids = [1, 2, 3] 
    
    shards = np.array_split(df, len(gpu_ids))
    
    mp.set_start_method('spawn', force=True)
    processes = []
    tmp_paths = [os.path.join(args.out_dir, f"tmp_ep{args.epoch}_g{g}.csv") for g in gpu_ids]
    
    # Launch Workers
    for i, gpu in enumerate(gpu_ids):
        if os.path.exists(tmp_paths[i]): os.remove(tmp_paths[i])
        p = mp.Process(target=run_shard, args=(i, gpu, shards[i], args.model_path, tmp_paths[i], args.batch_size))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    # Aggregate Results
    # (Same aggregation logic as before...)
    from utils import parse_comprehensive_output, calculate_metrics, LABEL2ID
    
    valid_dfs = [pd.read_csv(p) for p in tmp_paths if os.path.exists(p)]
    if not valid_dfs:
        print("No results generated!")
        return

    all_res = pd.concat(valid_dfs)
    all_res.to_csv(os.path.join(args.out_dir, f"preds_epoch_{args.epoch}.csv"), index=False)
    
    # Calculate Metrics
    collectors = {k: {'true': [], 'pred': []} for k in ['Accuracy', 'Fluency', 'Prosody', 'Words', 'Phonemes']}
    
    for _, row in all_res.iterrows():
        t_data = parse_comprehensive_output(row['target'])
        p_data = parse_comprehensive_output(row['pred'])
        
        for k in ['Accuracy', 'Fluency', 'Prosody']:
            if k in t_data and k in p_data:
                tv = LABEL2ID.get(t_data[k])
                pv = LABEL2ID.get(p_data[k])
                if tv is not None and pv is not None:
                    collectors[k]['true'].append(tv)
                    collectors[k]['pred'].append(pv)
        
        for k in ['Words', 'Phonemes']:
            if k in t_data and k in p_data:
                t_labels = [LABEL2ID.get(x) for x in t_data[k]]
                p_labels = [LABEL2ID.get(x) for x in p_data[k]]
                min_len = min(len(t_labels), len(p_labels))
                for i in range(min_len):
                    tv, pv = t_labels[i], p_labels[i]
                    if tv is not None and pv is not None:
                        collectors[k]['true'].append(tv)
                        collectors[k]['pred'].append(pv)

    # Log to CSV
    row_data = [args.epoch, "N/A", "N/A"]
    print(f"\n--- Epoch {args.epoch} Results ---")
    for task in ['Accuracy', 'Fluency', 'Prosody', 'Words', 'Phonemes']:
        m = calculate_metrics(collectors[task]['true'], collectors[task]['pred'])
        row_data.extend([f"{m['acc']:.4f}", f"{m['pcc']:.4f}", f"{m['qwk']:.4f}", f"{m['mse']:.4f}"])
        print(f"{task:<10} | Acc: {m['acc']:.3f} | PCC: {m['pcc']:.3f}")

    with open(args.log_file, "a") as f:
        csv.writer(f).writerow(row_data)
        
    for p in tmp_paths:
        if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    main()