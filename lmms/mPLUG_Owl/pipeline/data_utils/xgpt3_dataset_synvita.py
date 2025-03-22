import json
import logging
import os
from pathlib import Path
import random
import re
import time
import traceback
import warnings
from io import BytesIO
import pandas as pd
import h5py
import numpy as np
import torch
import pylcs
from icecream import ic
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset

from src.domain import Domain
from src.semantic_mask import generate_shared_semantic_mask

from utils import get_args

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class MultiModalDataset(Dataset):
    """MultiModal dataset"""

    def __init__(self, data_dir, features_dir, input_file, tokenizer, processor,
                 max_length=2048,
                 media_tokens=['<image>', '<|video|>'], loss_objective = 'sequential', use_extracted_features=False):
        
        args = get_args()

        self.loss_objective = loss_objective
        if 'sequential' in self.loss_objective:
            self.dataset = pd.read_csv(input_file) 
            self.dataset = self.dataset.dropna()
        else:
            raise NotImplementedError('dataset loader not implemented for other loss objectives')
        
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.dataset = pd.read_csv(input_file)
        self.dataset_real = self.dataset[self.dataset['domain'] == 'real'].reset_index(drop=True)
        self.dataset_synthetic = self.dataset[self.dataset['domain'] == 'synthetic'].reset_index(drop=True)
        self.dataset_synthetic['video_id'] = self.dataset_synthetic['videopath'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        self.dataset_real = self.dataset_real.replace({np.nan: None})
        self.dataset_synthetic = self.dataset_synthetic.replace({np.nan: None})
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processor = processor
        self.media_tokens = {k: -int(i+1) for i, k in enumerate(media_tokens)}
        self.media_lengths = {'<image>': 1+64,'<|video|>': 1+64}
        print("num_media_token: ", self.media_lengths)
        print(len(self.dataset))
        self.bucket = {}
        self.use_extracted_features = use_extracted_features

        real_paired_df = self.dataset_real.copy(deep=True)
        real_paired_df["original_index"] = self.dataset_real.index
        real_paired_df = real_paired_df[real_paired_df['syn_video'].notnull()]       
        self.real_indices_paired = real_paired_df["original_index"].tolist()
        self.real_indices_non_paired = list(set(self.dataset_real.index) - set(self.real_indices_paired))

    def __len__(self):
        return len(self.dataset_real)

    def __getitem__(self, index):
        data = self.dataset_real.iloc[index]
        videopath   = data['videopath']
        caption     = data['caption']
        label       = data['label']
        video_id    = data['syn_video']

        data_list = [data]
        
        neg_caption = None

        if index in self.real_indices_paired:
            # label == 0 --> (V^r, t^s) and (V^s, t^s)
            # label == 1 --> (V^r, t^r) and (V^s, t^r)
            syn_label = 1 if label == 0 else 0

            data_synth = self.dataset_synthetic[
                (self.dataset_synthetic['video_id'] == video_id) &
                (self.dataset_synthetic['label'] == syn_label)
            ].reset_index(drop=True)

            data_list += data_synth.to_dict('records')

            neg_caption = self.dataset_synthetic[
                (self.dataset_synthetic['video_id'] == video_id) &
                (self.dataset_synthetic['label'] == label)
            ].iloc[0]['caption']

        items = []

        for data in data_list:
            videopath = data['videopath']
            caption = data['caption']
            label = data['label']
            alignment = data['alignment'] if data['alignment'] is not None else 1.0
            alignment_diff = data['alignment_diff'] if data['alignment_diff'] is not None else 1.0
            domain = Domain[data['domain']]

            if domain == Domain.synthetic and label == 0:
                alignment = alignment_diff + alignment

            if self.use_extracted_features and domain == Domain.real:
                videopath = os.path.join(self.features_dir, videopath)
                video_input = torch.load(Path(videopath).with_suffix('.pt'))
            else:
                videopath = os.path.join(self.data_dir, videopath)
                video_input = self.processor(videos=[videopath], num_frames=32, return_tensors='pt') # video_pixel_values

            text_input = self._extract_text_token_from_conversation(caption, self.max_length, index)
            item  = {'video': video_input, 'text': text_input, 'videopath': videopath, 'caption': caption, 'domain': domain, 'alignment': alignment, 'alignment_diff': alignment_diff}
            items.append(item)

            if neg_caption is not None:
                text_input = self._extract_text_token_from_conversation(caption, self.max_length, index, neg_caption)
                item  = {'video': video_input, 'text': text_input, 'videopath': videopath, 'caption': caption, 'domain': domain, 'alignment': alignment, 'alignment_diff': alignment_diff}
                items.append(item) 

        return items

    def _extract_text_token_from_conversation(self, data, max_length, index, neg_data=None, remove_not_mask=False):      
        # output enc_chunk
        enc_chunk = []

        if self.tokenizer.bos_token_id > 0:
            prompt_chunk = [self.tokenizer.bos_token_id]
        else:
            prompt_chunk = []

        # conversation = data["completion"]
        conversation = data

        # For Text only data
        if all([media_token not in conversation for media_token in self.media_tokens.keys()]):
            pattern = '|'.join(map(re.escape, ['AI: ', '\nHuman: ']))
            chunk_strs = re.split(f'({pattern})', conversation)
            prompt_length = -1
            stop_flag = False
            for idx, chunk_str in enumerate(chunk_strs):
                if idx == 0:
                    enc_chunk = prompt_chunk + \
                        self.tokenizer(chunk_str, add_special_tokens=False)[
                            'input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_strs[idx-1] == 'AI: ':
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False)['input_ids']
                        if enc_length + len(curr_chunk) >= max_length:
                            curr_chunk = curr_chunk[:max_length-enc_length]
                            stop_flag = True
                        curr_chunk += [self.tokenizer.eos_token_id]
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [1] * len(curr_chunk)
                    else:
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False)['input_ids']
                        if enc_length + len(curr_chunk) >= max_length + 1:
                            curr_chunk = curr_chunk[:max_length+1-enc_length]
                            stop_flag = True
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)
                    if stop_flag:
                        break

        # For Image-Text Data
        else:
            enc_length = 0
            prompt_length = -2
            pattern = '|'.join(
                map(re.escape, list(self.media_tokens.keys()) + ['AI: ', '\nHuman: ']))
            chunk_strs = re.split(f'({pattern})', conversation)
            chunk_strs = [x for x in chunk_strs if len(x) > 0]
            for idx, chunk_str in enumerate(chunk_strs):
                if enc_length >= max_length + 1:
                    break

                if idx == 0:
                    enc_chunk = prompt_chunk + \
                        self.tokenizer(chunk_str, add_special_tokens=False)[
                            'input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_str in self.media_tokens:
                        # [CLS] + 256 + [EOS]
                        if enc_length + self.media_lengths[chunk_str] > max_length + 1:
                            break
                        else:
                            enc_chunk += [self.media_tokens[chunk_str]
                                          ] * self.media_lengths[chunk_str]
                            enc_length += self.media_lengths[chunk_str]
                            label_chunk += [0] * self.media_lengths[chunk_str]
                    else:

                        if chunk_strs[idx-1] == 'AI: ':
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False)['input_ids']
                            if enc_length + len(curr_chunk) >= max_length:
                                curr_chunk = curr_chunk[:max_length-enc_length]
                            curr_chunk += [self.tokenizer.eos_token_id]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [1] * len(curr_chunk)
                        else:
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False)['input_ids']
                            if enc_length + len(curr_chunk) >= max_length + 1:
                                curr_chunk = curr_chunk[:max_length +
                                                        1-enc_length]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [0] * len(curr_chunk)

        if enc_length < max_length + 1:
            padding_chunk = [self.tokenizer.pad_token_id] * \
                (max_length + 1 - enc_length)
            padding_length = len(padding_chunk)
            label_chunk += [0] * (max_length + 1 - enc_length)
            enc_chunk = enc_chunk + padding_chunk
        else:
            padding_length = 0

        assert enc_length + padding_length == max_length + \
            1, (index, prompt_length, enc_length,
                padding_length, max_length + 1)
        assert len(label_chunk) == max_length + \
            1, (len(label_chunk), max_length + 1)
        non_padding_mask = [1 if i < enc_length -
                            1 else 0 for i in range(max_length)]

        enc_chunk = torch.tensor(enc_chunk).long()
        non_padding_mask = torch.tensor(non_padding_mask).long()
        prompt_mask = torch.tensor(label_chunk)[1:].long()
        prompt_length = torch.tensor([prompt_length]).long()

        # Create loss mask
        if all([media_token not in conversation for media_token in self.media_tokens.keys()]):
            non_media_mask = torch.ones_like(non_padding_mask).long()
        else:
            tmp_enc_chunk = enc_chunk.clone()
            tmp_enc_chunk[tmp_enc_chunk >= 0] = 1
            tmp_enc_chunk[tmp_enc_chunk < 0] = 0
            non_media_mask = torch.tensor(tmp_enc_chunk).long()
            non_media_mask = non_media_mask[1:].long()
    
        shared_semantic = False
        shared_semantic_mask = torch.ones_like(enc_chunk).long()

        if neg_data is not None:
            shared_semantic = True
            shared_semantic_mask = generate_shared_semantic_mask(data, neg_data, enc_chunk, self.tokenizer)

        return {'input_ids': enc_chunk, "prompt_length": prompt_length, 'seq_length': enc_length,
                "non_padding_mask": non_padding_mask, 'non_media_mask': non_media_mask, 'prompt_mask': prompt_mask,
                'shared_semantic_mask': shared_semantic_mask, 'shared_semantic': shared_semantic}