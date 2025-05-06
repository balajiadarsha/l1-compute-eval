import random
import pandas as pd


from config import *
from utils import *
from utils_model import VLLM_LocalServer, morethink_batch
from strong_reject.load_datasets import load_strongreject_small
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from strong_reject.generate import generate_to_dataset
from strong_reject.evaluate import evaluate_dataset

#df = pd.read_parquet("hf://datasets/walledai/StrongREJECT/data/train-00000-of-00001.parquet")
#print(df.head())
dataset = load_strongreject_small()
prompt = dataset['forbidden_prompt']

for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []
    i = 1
    for query in prompt:
        question = query.strip()
        if num_tokens >=-1:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer within \\boxed{}." + (f" Think for {num_tokens} tokens." if num_tokens != -1 else "")
        else:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer within \\boxed{}." + (f" Think for maximum {abs(num_tokens)} tokens.")
    



        all_data.append({
                    "data_source": "strongreject",
                    "prompt": [{
                        "role": "user",
                        "content": question
                    }],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": 'strongreject',
                        "num_tokens": num_tokens
                    },
                    "extra_info": {
                        'split': 'test',
                        'index': i
                    }
                })
        i=i+1
    
    if num_tokens == -1:
        pd.DataFrame(all_data).to_parquet(f'/lus/eagle/projects/argonne_tpc/abalaji/datasets/deepscaler/data/strongreject.parquet')
    else:
        if num_tokens < 0:
            pd.DataFrame(all_data).to_parquet(f'/lus/eagle/projects/argonne_tpc/abalaji/datasets/deepscaler/data9_{num_tokens}/strongreject.parquet')
        else:
            pd.DataFrame(all_data).to_parquet(f'/lus/eagle/projects/argonne_tpc/abalaji/datasets/deepscaler/data_{num_tokens}/strongreject.parquet')
    
