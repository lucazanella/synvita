import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import os
import csv
import json
import re
import torch
import argparse
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data import DataLoader
from lmms.mPLUG_Owl.mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from lmms.mPLUG_Owl.mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from peft import LoraConfig, get_peft_model
from data_utils.xgpt3_dataset import MultiModalDataset
from utils import batchify
from transformers.trainer_utils import enable_full_determinism

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type = str, required = True, help = 'data directory')
parser.add_argument('--input_csv', type = str, required = True, help = 'input json file')
parser.add_argument('--output_csv', type = str, help = 'output csv with scores')
parser.add_argument('--pretrained_ckpt', type = str, required = True, help = 'pretrained ckpt')
parser.add_argument('--trained_ckpt', type = str, help = 'trained ckpt')
parser.add_argument('--lora_r', type = int, default = 32)
parser.add_argument('--lora_alpha', type = int, default = 32)
parser.add_argument('--use_lora', action = 'store_true', help = 'lora model')
parser.add_argument('--all-params', action = 'store_true', help = 'use all params of the model')
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--use-extracted-features', action='store_true', default=False, help='Whether to use pre-extracted features')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--update-state-dict', action = 'store_true', help = 'update state dict')

args = parser.parse_args()
softmax = nn.Softmax(dim=2)

def update_state_dict(state_dict):
    new_state_dict = defaultdict()

    pattern = re.compile(r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj).weight')

    for key, value in state_dict.items():
        if pattern.match(key):
            key = key.split('.')
            key.insert(-1, 'base_layer')
            key = '.'.join(key)
        new_state_dict[key] = value

    return new_state_dict

def get_entail(logits, input_ids, tokenizer):
    logits = softmax(logits)
    token_id_yes = tokenizer.encode('Yes', add_special_tokens = False)[0]
    token_id_no  = tokenizer.encode('No', add_special_tokens = False)[0]
    entailment = []
    for j in range(len(logits)):
        for i in range(len(input_ids[j])):
            if input_ids[j][i] == tokenizer.pad_token_id: # pad token if the answer is not present
                i = i - 1
                break
            elif i == len(input_ids[j]) - 1:
                break
        score = logits[j][i][token_id_yes] / (logits[j][i][token_id_yes] + logits[j][i][token_id_no])
        entailment.append(score)
    entailment = torch.stack(entailment)
    return entailment

def get_scores(model, tokenizer, dataloader, use_extracted_features):

    with torch.no_grad():
        for index, inputs in tqdm(enumerate(dataloader)):
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    if v.dtype == torch.float:
                        inputs[k] = v.bfloat16()
                    inputs[k] = inputs[k].to(model.device)
            outputs = model(pixel_values = inputs['pixel_values'], video_pixel_values = inputs['video_pixel_values'], labels = None, \
                                num_images = inputs['num_images'], num_videos = inputs['num_videos'], input_ids = inputs['input_ids'], non_padding_mask = inputs['non_padding_mask'], \
                                non_media_mask = inputs['non_media_mask'], prompt_mask = inputs['prompt_mask'], \
                                use_extracted_features = use_extracted_features)
            logits = outputs['logits']
            entail_scores = get_entail(logits, inputs['input_ids'], tokenizer)
            for m in range(len(entail_scores)):
                with open(args.output_csv, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([inputs['videopaths'][m], inputs['captions'][m], entail_scores[m].item()])
            print(f"Batch {index} Done")

def main():
    enable_full_determinism(args.seed)
    
    pretrained_ckpt = args.pretrained_ckpt
    use_extracted_features = args.use_extracted_features

    # Processors
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_ckpt)
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    valid_data = MultiModalDataset(args.data_dir, args.data_dir, args.input_csv, tokenizer, processor, max_length = 256, loss_objective = 'sequential', use_extracted_features = use_extracted_features)
    dataloader = DataLoader(valid_data, batch_size=args.batch_size, pin_memory=True, collate_fn=batchify)
    
    # Instantiate model
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
        device_map={'':0}
    )

    if args.use_lora:
        for name, param in model.named_parameters():
            param.requires_grad = False
        if args.all_params:
            peft_config = LoraConfig(
                target_modules=r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj)', 
                inference_mode=True, 
                r=args.lora_r, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=0.05
            )
        else:
            peft_config = LoraConfig(
                target_modules=r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj)', 
                inference_mode=True, 
                r=args.lora_r, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=0.05
            )
            
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        with open(args.trained_ckpt, 'rb') as f:
            ckpt = torch.load(f, map_location = torch.device(f"cuda:0"))
        if args.update_state_dict:
            ckpt = update_state_dict(ckpt)
        model.load_state_dict(ckpt)
        model = model.to(torch.bfloat16)
        print('Model Loaded')

    model.eval()

    get_scores(model, tokenizer, dataloader, use_extracted_features)

if __name__  == "__main__":
    main()
