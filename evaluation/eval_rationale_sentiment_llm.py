import pandas as pd
import math
import multiprocessing as mp
import re
import json
import os

# -----------------------------------------------------------------
# 1. Structured JSON Prompt for Sentiment Analysis
# -----------------------------------------------------------------
def get_sentiment_prompt(explanation):
    return f"""Analyze the speech evaluation feedback below and determine its overall sentiment.

CRITICAL RULES FOR EXTRACTION:
1. You must classify the explanation into exactly ONE of three categories: Positive, Negative, or Neutral.
2. Use the following strict rubric to determine the sentiment:
   - Positive: The feedback generally praises the speech, describing it as "excellent", "good", "accurate", "smooth", or having very few errors.
   - Neutral: The feedback describes the speech as "average", "fair", or acceptable. It might evenly balance positive remarks with noticeable issues.
   - Negative: The feedback focuses heavily on errors, describing the speech as "bad", "poor", "worst", "incorrect", "unintelligible", or having a "heavy accent".
3. Output ONLY valid JSON in this exact format. Do not use markdown blocks (```json). Just output the raw JSON string:

{{
  "sentiment": "Positive"
}}

Feedback: "{explanation}"
"""

# Helper to safely parse the LLM's JSON output
def safe_parse_json(generated_text):
    # Strip markdown formatting if the model accidentally includes it
    clean_text = generated_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_text)
        # Ensure it only returns the sentiment string or a fallback
        return data.get("sentiment", "Unknown")
    except json.JSONDecodeError:
        # Fallback regex if JSON parsing completely fails
        if "Positive" in clean_text: return "Positive"
        if "Negative" in clean_text: return "Negative"
        if "Neutral" in clean_text: return "Neutral"
        return "Parse Error"

# -----------------------------------------------------------------
# 2. The Worker Function
# -----------------------------------------------------------------
def process_chunk(args):
    chunk_df, gpu_id, chunk_id, model_id, cache_dir = args
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    from tqdm import tqdm

    print(f"[Worker {chunk_id}] Booting up on GPU {gpu_id}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map={"":"cuda:0"} 
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    out_df = chunk_df.copy()
    out_df['explanation_sentiment'] = ""
    
    output_filename = f'llm_sentiment_chunk_{chunk_id}_GPU{gpu_id}.csv'

    for i, (index, row) in enumerate(tqdm(chunk_df.iterrows(), total=len(chunk_df), position=chunk_id, desc=f"GPU {gpu_id}")):
        explanation = row['explanation']
        
        # Skip empty explanations
        if not isinstance(explanation, str) or len(explanation.strip()) < 5:
            out_df.at[index, 'explanation_sentiment'] = "Unknown"
            continue

        prompt_text = get_sentiment_prompt(explanation)
        
        messages = [
            {"role": "system", "content": "You are a precise sentiment analyzer. Classify the text into exactly one category and output ONLY raw JSON."},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            # Note: We assign the output directly to formatted_prompt
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            outputs = pipe(
                formatted_prompt,
                max_new_tokens=20, # We only need a few tokens for {"sentiment": "X"}
                temperature=0.1,   # Low temp for deterministic output
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            
            gen_text = outputs[0]["generated_text"].strip()
            out_df.at[index, 'explanation_sentiment'] = safe_parse_json(gen_text)
            
        except Exception as e:
            out_df.at[index, 'explanation_sentiment'] = "Error"
            
        # Save checkpoints
        if (i + 1) % 50 == 0 or (i + 1) == len(chunk_df):
            out_df.to_csv(output_filename, index=False)

    return output_filename

# -----------------------------------------------------------------
# 3. Execution Setup
# -----------------------------------------------------------------
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    AVAILABLE_GPUS = [0, 1, 3, 4, 6, 2, 7, 5] 
    model_id = "Qwen/Qwen2.5-7B-Instruct" 
    custom_cache_path = "/vol/tensusers8/aparikh/dpoftllm/cache_dir/"
    
    input_file = '/vol/tensusers8/aparikh/dpoftllm/Multi-Granular/runs/comprehensive_bdpo_v2/preds_epoch_11.csv' 

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Extract the explanation part
    def extract_exp(text):
        parts = re.split(r'(?mi)^Phonemes:.*?\n', str(text) + '\n')
        return parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        
    if 'explanation' not in df.columns:
        df['explanation'] = df['pred'].apply(extract_exp)
        
    df_to_process = df.dropna(subset=['explanation']).copy()

    num_gpus = len(AVAILABLE_GPUS)
    chunk_size = math.ceil(len(df_to_process) / num_gpus)
    chunks = [df_to_process.iloc[i:i + chunk_size] for i in range(0, len(df_to_process), chunk_size)]

    print(f"Spinning up {num_gpus} parallel workers for SENTIMENT EXTRACTION...")

    worker_args = []
    for chunk_id, chunk_df in enumerate(chunks):
        gpu_id = AVAILABLE_GPUS[chunk_id]
        worker_args.append((chunk_df, gpu_id, chunk_id, model_id, custom_cache_path))

    with mp.Pool(processes=num_gpus) as pool:
        chunk_files_produced = pool.map(process_chunk, worker_args)

    print("\nMerging chunk files...")
    all_dfs = [pd.read_csv(f) for f in chunk_files_produced]
    final_merged_df = pd.concat(all_dfs).sort_index()
    
    final_merged_df.to_csv('llm_explanation_sentiment_final.csv', index=False)
    print("Done! Saved to 'llm_explanation_sentiment_final.csv'")