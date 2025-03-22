import copy
import json
import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pylcs
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
# from videollava.train.train import preprocess, preprocess_multimodal
from lmms.Video_LLaVA.videollava.train.train import preprocess, preprocess_multimodal

from src.domain import Domain
from src.semantic_mask import generate_shared_semantic_mask

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_jsonl(filename):
    with open(filename, encoding="utf-8") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class MultiModalDataset(Dataset):
    """MultiModal dataset."""

    def __init__(
        self,
        video_folder,
        features_folder,
        input_file,
        tokenizer,
        processor,
        data_args,
        max_length=2048,
        use_extracted_features=False,
    ):
        self.video_folder = video_folder
        self.features_folder = features_folder
        self.data_args = data_args
        self.dataset = pd.read_csv(input_file)
        self.dataset_real = self.dataset[self.dataset['domain'] == 'real'].reset_index(drop=True)
        self.dataset_synthetic = self.dataset[self.dataset['domain'] == 'synthetic'].reset_index(drop=True)
        self.dataset_synthetic['video_id'] = self.dataset_synthetic['videopath'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        self.dataset_real = self.dataset_real.replace({np.nan: None})
        self.dataset_synthetic = self.dataset_synthetic.replace({np.nan: None})
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processor = processor
        print(len(self.dataset))
        self.bucket = {}
        self.use_extracted_features = use_extracted_features

        real_paired_df = self.dataset_real.copy(deep=True)
        real_paired_df["original_index"] = self.dataset_real.index
        real_paired_df = real_paired_df[real_paired_df["syn_video"].notnull()]
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
                videopath = os.path.join(self.features_folder, videopath)
                video_input = torch.load(Path(videopath).with_suffix(".pt"), map_location="cpu")
            else:
                videopath = os.path.join(self.video_folder, videopath)
                video_input = self.processor([videopath], return_tensors="pt")["pixel_values"][0]
            
            videopath = os.path.join(self.video_folder, data['videopath'])
            source = self._convert_text_to_json(caption, videopath, index)
            text_input = self._extract_text_token_from_conversation([source])
            item  = {
                'video': video_input, 
                'text': text_input, 
                'videopath': videopath, 
                'caption': caption, 
                'domain': domain, 
                'alignment': alignment, 
                'alignment_diff': alignment_diff
            }
            items.append(item)

            if neg_caption is not None:
                source = self._convert_text_to_json(caption, videopath, index)
                text_input = self._extract_text_token_from_conversation([source], caption, neg_caption)
                item  = {
                    'video': video_input, 
                    'text': text_input, 
                    'videopath': videopath, 
                    'caption': caption, 
                    'domain': domain, 
                    'alignment': alignment, 
                    'alignment_diff': alignment_diff
                }
                items.append(item)

        return items
        
    def _convert_text_to_json(self, text, videopath, index):
        ignored_part, main_content = text.split("<|video|>\n", 1)
        parts = main_content.split("\n")
        human_part = parts[0].replace("Human: ", "").strip()
        human_part = "<video>\n" + human_part
        ai_part = parts[1].replace("AI: ", "").strip()

        # Create the desired dictionary format
        data = {
            "id": index,
            "video": videopath,
            "conversations": [
                {
                    "from": "human",
                    "value": human_part
                },
                {
                    "from": "gpt",
                    "value": ai_part
                }
            ]
        }

        return data

    def _extract_text_token_from_conversation(self, sources, data=None, neg_data=None):
        sources = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]), self.data_args
        )
        data_dict = preprocess(sources, self.tokenizer, has_image=True)

        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        shared_semantic = False
        shared_semantic_mask = input_ids.new_ones(*input_ids.shape).long()

        if neg_data is not None:
            shared_semantic = True
            shared_semantic_mask = generate_shared_semantic_mask(data, neg_data, input_ids, self.tokenizer)

        return {
            "input_ids": input_ids, 
            "labels": labels,
            "shared_semantic_mask": shared_semantic_mask,
            "shared_semantic": shared_semantic
        }
