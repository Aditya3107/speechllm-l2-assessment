import pandas as pd
import math
import multiprocessing as mp
import re
import json

# -----------------------------------------------------------------
# 1. Structured JSON Prompt with Strict Rubric & Visual Clues
# -----------------------------------------------------------------
def get_json_prompt(explanation):
    return f"""Analyze the speech evaluation feedback below to extract specific words and phonemes, along with their implied scores.

CRITICAL RULES FOR EXTRACTION:
1. HOW TO IDENTIFY WORDS: Target vocabulary words are usually enclosed in quotes (e.g., "USUAL", 'ELEPHANT') or written in ALL CAPS.
2. HOW TO IDENTIFY PHONEMES: Phonemes are usually enclosed in forward slashes (e.g., /m/, /æ/) or quotes (e.g., 't', "d").
3. DO NOT extract meta-language or grading terms (e.g., ignore words like "accuracy", "fluency", "prosody", "word", "pronounced", "errors", "spelling", "phone", "sound", "stress", "intonation").
4. Clean the output: When writing the JSON, remove the quotes or slashes from the word/phoneme (e.g., output "m" instead of "/m/").

SCORING RUBRIC:
Assign an implied score (Excellent, Good, Average, Bad, Worst) to each extracted word/phoneme using these strict guidelines:
- Excellent: The item is described as "excellent", "very accurate", "correctly pronounced/spelled", or having "no errors".
- Good: The item is described as "good", "mostly correct", having "minor errors", or a "slight accent".
- Average: The item is described as "average", "fair", "acceptable", having "some errors", or spoken with a "heavy accent".
- Bad: The item is described as "poor", "bad", "misspelled", "incorrect", having "several errors", "extra sound", or "missing sound".
- Worst: The item is described explicitly as the "worst", "completely unintelligible", or "completely unnatural".

Output ONLY valid JSON in this exact format. Do not use markdown blocks (```json). Just output the raw JSON string:
{{
  "words": [
    {{"word": "USUAL", "implied_score": "Bad"}},
    {{"word": "EVEN", "implied_score": "Good"}}
  ],
  "phonemes": [
    {{"phoneme": "m", "implied_score": "Average"}},
    {{"phoneme": "æ", "implied_score": "Bad"}}
  ]
}}
If no specific vocabulary words or phonetic characters are explicitly named, return {{"words": [], "phonemes": []}}.

Feedback: "{explanation}"
"""

# Helper to safely parse the LLM's JSON output
def safe_parse_json(generated_text):
    clean_text = generated_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_text)
        return json.dumps(data) 
    except json.JSONDecodeError:
        return json.dumps({"words": [], "phonemes": [], "error": "JSON Parse Error"})

# -----------------------------------------------------------------
# 2. The Worker Function
# -----------------------------------------------------------------
def process_chunk(args):
    chunk_df, gpu_id, chunk_id, model_id, cache_dir = args
    
    import os
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
    out_df['llm_extracted_json'] = ""
    
    output_filename = f'llm_word_phoneme_chunk_{chunk_id}_GPU{gpu_id}.csv'

    for i, (index, row) in enumerate(tqdm(chunk_df.iterrows(), total=len(chunk_df), position=chunk_id, desc=f"GPU {gpu_id}")):
        explanation = row['explanation']
        prompt_text = get_json_prompt(explanation)
        
        messages = [
            {"role": "system", "content": "You are a precise JSON data extractor following a strict rubric. Output only raw JSON."},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            outputs = pipe(
                formatted_prompt,
                max_new_tokens=150, 
                temperature=0.1, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            
            gen_text = outputs[0]["generated_text"].strip()
            out_df.at[index, 'llm_extracted_json'] = safe_parse_json(gen_text)
            
        except Exception as e:
            out_df.at[index, 'llm_extracted_json'] = json.dumps({"error": str(e)})
            
        if (i + 1) % 10 == 0 or (i + 1) == len(chunk_df):
            out_df.to_csv(output_filename, index=False)

    return output_filename

# -----------------------------------------------------------------
# 3. Execution Setup
# -----------------------------------------------------------------
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    AVAILABLE_GPUS = [0, 1, 3, 4, 6, 2, 7] 
    model_id = "Qwen/Qwen2.5-7B-Instruct" 
    custom_cache_path = "/vol/tensusers8/aparikh/dpoftllm/cache_dir/"
    
    input_file = '/vol/tensusers8/aparikh/dpoftllm/Multi-Granular/runs/comprehensive_bdpo_v2/preds_epoch_11.csv' 

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    def extract_exp(text):
        parts = re.split(r'(?mi)^Phonemes:.*?\n', str(text) + '\n')
        return parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        
    if 'explanation' not in df.columns:
        df['explanation'] = df['pred'].apply(extract_exp)
        
    df_to_process = df.dropna(subset=['explanation']).copy()

    num_gpus = len(AVAILABLE_GPUS)
    chunk_size = math.ceil(len(df_to_process) / num_gpus)
    chunks = [df_to_process.iloc[i:i + chunk_size] for i in range(0, len(df_to_process), chunk_size)]

    print(f"Spinning up {num_gpus} parallel workers for JSON EXTRACTION...")

    worker_args = []
    for chunk_id, chunk_df in enumerate(chunks):
        gpu_id = AVAILABLE_GPUS[chunk_id]
        worker_args.append((chunk_df, gpu_id, chunk_id, model_id, custom_cache_path))

    with mp.Pool(processes=num_gpus) as pool:
        chunk_files_produced = pool.map(process_chunk, worker_args)

    print("\nMerging chunk files...")
    all_dfs = [pd.read_csv(f) for f in chunk_files_produced]
    final_merged_df = pd.concat(all_dfs).sort_index()
    
    final_merged_df.to_csv('llm_word_phoneme_json_v5_visual_clues.csv', index=False)
    print("Done! Saved to 'llm_word_phoneme_json_v5_visual_clues.csv'")