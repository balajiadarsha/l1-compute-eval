from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.third_party.vllm import LLM, vllm_version
from verl.utils.model import update_model_config
from vllm import SamplingParams
from llama_cpp import Llama
import torch
from datasets import load_dataset, Dataset

import numpy as np
import os
import csv
import pandas as pd
from pprint import pprint
import argparse

#evaluator import
from strong_reject.evaluate import evaluate_dataset
from utils.utils import Safe_SubString 

#DEBUG FLAGS:
TEST=0
LLAMA=0
EVALUATOR='LLAMA'

fieldnames=['prompt', 'response']
def write_dict_to_csv(data, filename, fieldnames):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile)
        writer.writeheader()
        writer.writerows(data)


# Create the parser
parser = argparse.ArgumentParser(description='Quantization and reasoning length')

# Add arguments
parser.add_argument('-q', '--quant', type=int, default=8,  help='quantization of the model weights')
parser.add_argument('-r', '--reason', type=int, default=512, help='reasoning length')
parser.add_argument('-d', '--dataset', type=str, default='gpqa', help='dataset')
parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')

# Parse the arguments
args = parser.parse_args()

# Use the arguments
if args.verbose:
    print("Verbose mode is on")

if args.quant is not None:
    print(f"Qunatization: {args.quant} bits")

if args.reason is not None:
    print(f"Reasoning length: {args.reason}")

model_name = "mradermacher/L1-Qwen-1.5B-Exact-GGUF"
filename = "L1-Qwen-1.5B-Exact.f16.gguf"

if args.quant==16:
    filename= "L1-Qwen-1.5B-Exact.f16.gguf"
elif args.quant==8:
    filename = "L1-Qwen-1.5B-Exact.Q8_0.gguf"
elif args.quant==4:
    filename = "L1-Qwen-1.5B-Exact.Q4_K_S.gguf"

if TEST:
    input_text = "What are we having for dinner?"
    input_ids = tokenizer(input_text, return_tensors="pt", device_map='auto')
    output = model.generate(**input_ids, max_new_tokens=10)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    exit()

if TEST==0:    
    config = AutoConfig.from_pretrained(model_name)
    actor_model_config = AutoConfig.from_pretrained(model_name, gguf_file=filename)
    model = AutoModelForCausalLM.from_pretrained(model_name, gguf_file=filename, torch_dtype=torch.float16, model_type="llama", device_map = 'auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name, gguf_file=filename, device_map='auto')
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

# Load Dataset
#tokenizer.pad_token = tokenizer.eos_token
dataset = pd.read_parquet('/lus/eagle/projects/argonne_tpc/abalaji/datasets/deepscaler/data_'+str(args.reason)+'/'+args.dataset+'.parquet')

# output path to store the response
output_path = '/lus/eagle/projects/argonne_tpc/abalaji/reasoning/l1/quant/response_'+str(args.reason)+'/'
output_dir = os.path.dirname(output_path)
makedirs(output_dir, exist_ok=True)

#config parameters for inferencei
config_batch_size = 2048
print('Config size')
print(config_batch_size) 
total_samples = len(dataset)
batch_size = 2048
num_batch = (total_samples // batch_size) + 1
chat_lst = dataset['prompt'].tolist()
print(chat_lst)
print(len(chat_lst))
chat_lst = [chat.tolist() for chat in chat_lst]
n_samples=16
max_length=args.reason*3



n_samples = 16
answers = []

evaluator_strongreject = Safe_SubString()

#list of output responses. 
output_lst = []
score_lst = []
passes=0

output_dict = {'prompt': [], 'response': []}

for batch_idx in range(num_batch):
    #print(f'[{batch_idx+1}/{num_batch}] Start to process.')
    print(len(chat_lst))
    batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
    print(batch_idx * config_batch_size, (batch_idx + 1) * config_batch_size)
    # Repeat the batch n_samples times
    repeated_chat_lst = batch_chat_lst
    #for chat in batch_chat_lst:
    #    repeated_chat_lst.extend([chat] * n_samples)

    #prompt = p
    
    #input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    inputs = tokenizer.apply_chat_template(repeated_chat_lst,
                                         add_generation_prompt=True,
                                         padding=True,
                                         truncation=True,
                                         max_length=512,
                                         return_tensors='pt',
                                         return_dict=True,
                                         device_map='auto',
                                         tokenize=True)
   
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    position_ids = compute_position_id_with_mask(attention_mask)
    inputs = inputs.to("cuda")
    #input_ids = inputs['input_ids'].to('cuda')
    #input_len = len(input_ids)
    # Generate all samples at once
    output = model.generate(**inputs, max_new_tokens=512)
    print(output.batch['input_ids'])
    print(len(output.batch['input_ids']))
    output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -args.reason:], skip_special_tokens=False)
    
    # Remove padding
    pad_token = tokenizer.pad_token
    output_text_unpad = []
    for text in output_text:
        output_text_unpad.append(text.replace(pad_token, ''))

    #print('Prompt: %s \n' %(prompt))
    print('Response: %s \n\n\n' %(output_text))

    exit()

    output_lst.append(output_text_unpad)

    if EVALUATOR=='String':
        score = evaluator_strongreject.eval_single(output_text)
    
        if score==0:
            passes+=1
    
        question = prompt
   
    #DEBUG: break after n queries
    #if i == 30:
    #    break
    
# Reshape output_lst from (total_samples,) to (n_data, n_samples)
total_samples = 60
n_data = total_samples // 1
#token_lengths = [len(tokenizer.encode(response)) for response in output_lst]
#output_lst = np.array(output_lst).reshape(n_data, 1).tolist()
#token_lengths = np.array(token_lengths).reshape(n_data, 1).tolist()

# Add to the data frame
dataset['responses'] = output_lst
#dataset['token_lengths'] = token_lengths

if EVALUATOR=='String':
    pass_at_1 = passes/total_samples
    print(pass_at_1)

dataset.to_parquet(output_path+'response_'+str(args.quant)+str(args.reason)+'.parquet')

#write_dict_to_csv(output_dict, fieldnames)


if LLAMA:
    dataset = Dataset.from_dict({"forbidden_prompt": 2*[prompt], "response":[output_text]})  
    EVALUATOR = ["llama_guard"]
    eval_dataset = evaluate_dataset(dataset, EVALUATOR)
    print(eval_dataset["score"])
    eval_dataset.to_parquet("eval_results.parquet")
