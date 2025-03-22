import rootutils

# find absolute root path (searches for directory containing .project-root file)
# search starts from current file and recursively goes over parent directories
# returns pathlib object
path = rootutils.find_root(search_from=__file__, indicator=".project-root")

# set root directory
rootutils.set_root(
    path=path, # path to the root directory
    project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
    dotenv=True, # load environment variables from .env if exists in root directory
    pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
    cwd=True, # change current working directory to the root directory (helps with filepaths)
)

import argparse
import csv
import os
from pathlib import Path
import random
import time
import numpy as np
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, required = True, help = 'model name or path')
    parser.add_argument('--model_base', type = str, help = 'model name or path')
    parser.add_argument('--data_dir', type = str, required = True, help = 'data directory')
    parser.add_argument('--input_csv', type = str, required = True, help = 'input json file')
    parser.add_argument('--output_csv', type = str, help = 'output csv with scores')
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--cache_dir', type = str, default='cache_dir')
    return parser.parse_args()


def extract_description(conversation):
    pattern = r'Does this video entail the description: "(.*)"\?'
    match = re.search(pattern, conversation)
    if match:
        return match.group(1)
    else:
        raise ValueError("Description not found")
    
    
def get_entail_batch(scores, tokenizer):
    softmax = nn.Softmax(dim=1)
    probs = softmax(scores)
    token_id_yes = tokenizer.encode('Yes', add_special_tokens = False)[0]
    token_id_no  = tokenizer.encode('No', add_special_tokens = False)[0]
    entailment = []
    for i in range(len(scores)):
        score = probs[i][token_id_yes] / (probs[i][token_id_yes] + probs[i][token_id_no])
        entailment.append(score.item())
    return entailment


def get_entail(scores, tokenizer):
    softmax = nn.Softmax(dim=1)
    probs = softmax(scores)
    token_id_yes = tokenizer.encode('Yes', add_special_tokens = False)[0]
    token_id_no  = tokenizer.encode('No', add_special_tokens = False)[0]
    entailment = probs[0][token_id_yes] / (probs[0][token_id_yes] + probs[0][token_id_no])
    entailment = entailment.item()
    return entailment


def roll_padding_to_front(padded_input_ids, padding_value):
    padding_lengths = (padded_input_ids == padding_value).long().sum(dim=1)
    rolled_input_ids = torch.stack([torch.roll(input_id, shifts=padding_length.item()) for input_id, padding_length in zip(padded_input_ids, padding_lengths)])
    return rolled_input_ids


def get_scores_batch(model, tokenizer, video_processor, conv_mode, args):

    with open(args.input_csv, 'r') as in_f:
        reader = csv.DictReader(in_f, delimiter=',')
        rows = list(reader)

    with open(args.output_csv, 'w') as out_f:
        writer = csv.writer(out_f)

        batch_size = args.batch_size
        num_batches = len(rows) // batch_size + (len(rows) % batch_size > 0)

        for i in tqdm(range(0, len(rows), batch_size), desc="Processing batches", total=num_batches):
            batch = rows[i:i + batch_size]

            batch_video_path = [os.path.join(args.data_dir, row['videopath']) for row in batch]
            captions = [extract_description(row['caption']) for row in batch]
            inputs = [f'Does this video entail the description: "{caption}"?' for caption in captions]
            video_tensor = video_processor(batch_video_path, return_tensors='pt')['pixel_values']

            if type(video_tensor) is list:
                # tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
                tensor = [video.to(model.device, dtype=torch.bfloat16) for video in video_tensor]
            else:
                # tensor = video_tensor.to(model.device, dtype=torch.float16)
                tensor = video_tensor.to(model.device, dtype=torch.bfloat16)

            convs = []
            input_ids = []
            stopping_criterias = []
            for inp in inputs:
                conv = conv_templates[conv_mode].copy()
                roles = conv.roles
                inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                cur_input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                convs.append(conv)
                input_ids.append(cur_input_ids.squeeze(0))

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, cur_input_ids)
                stopping_criterias.append(stopping_criteria)

            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=tokenizer.pad_token_id).cuda()
            padded_input_ids = padded_input_ids[:, :tokenizer.model_max_length]
            padded_input_ids = roll_padding_to_front(padded_input_ids, tokenizer.pad_token_id)
            setattr(model.config, 'tokenizer_padding_side', "left")

            # stop_str = convs[0].sep if convs[0].sep_style != SeparatorStyle.TWO else convs[0].sep2
            # keywords = [stop_str]
            # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, padded_input_ids)
            
            # with torch.inference_mode():
            #     generation_output = model.generate(
            #         padded_input_ids,
            #         images=tensor,
            #         non_padding_mask=non_padding_mask,
            #         do_sample=False,
            #         temperature=0.0,
            #         max_new_tokens=1024,
            #         use_cache=True,
            #         stopping_criteria=stopping_criterias,
            #         return_dict_in_generate=True,
            #         output_scores=True)
            with torch.inference_mode():
                generation_output = model.generate(
                    padded_input_ids,
                    attention_mask=padded_input_ids.ne(tokenizer.pad_token_id),
                    images=tensor,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=stopping_criterias,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            output_ids, scores = generation_output[:2]
            # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            outputs = tokenizer.batch_decode(
                output_ids[:, padded_input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            outputs = [output.strip() for output in outputs]
            # print(outputs)

            entail_scores = get_entail_batch(scores[0], tokenizer)
            for i, row in enumerate(batch):
                writer.writerow([batch_video_path[i], captions[i], entail_scores[i]])


def get_scores(model, tokenizer, video_processor, conv_mode, args):
    with open(args.input_csv, 'r') as in_f:
        reader = csv.DictReader(in_f, delimiter=',')
        with open(args.output_csv, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(['videopath', 'caption', 'entailment'])
            for row in tqdm(reader, desc="Processing rows"):
                inp = '''Does this video entail the description: "{caption}"?'''
                video_path = row['videopath']
                caption = row['caption']
                caption = extract_description(caption)
                inp = inp.format(caption = caption)

                video_path = os.path.join(args.data_dir, video_path)
                video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
                if type(video_tensor) is list:
                    # tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
                    tensor = [video.to(model.device, dtype=torch.bfloat16) for video in video_tensor]
                else:
                    # tensor = video_tensor.to(model.device, dtype=torch.float16)
                    tensor = video_tensor.to(model.device, dtype=torch.bfloat16)

                conv = conv_templates[conv_mode].copy()
                roles = conv.roles
                print(f"{roles[1]}: {inp}")
                inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    generation_output = model.generate(
                        input_ids,
                        images=tensor,
                        do_sample=False,
                        temperature=0.0,
                        max_new_tokens=256,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        return_dict_in_generate=True,
                        output_scores=True)

                output_ids, scores = generation_output[:2]
                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                print(outputs)
                entail_score = get_entail(scores[0], tokenizer)
                # print(entail_score)
                
                writer.writerow([video_path, caption, entail_score])


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    disable_torch_init()
    model_path = args.model_path
    model_base = args.model_base
    if model_base is None:
        model_name = get_model_name_from_path('LanguageBind/Video-LLaVA-7B')
    else:
        model_name = get_model_name_from_path(model_path)
    cache_dir = args.cache_dir
    device = 'cuda'
    load_4bit, load_8bit = False, False
    tokenizer, model, processor, _ = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    model = model.to(dtype=torch.bfloat16)
    video_processor = processor['video']
    conv_mode = "llava_v1"

    if args.batch_size > 1:
        get_scores_batch(model, tokenizer, video_processor, conv_mode, args)
    else:
        get_scores(model, tokenizer, video_processor, conv_mode, args)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
