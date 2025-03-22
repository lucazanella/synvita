import copy
import json
import logging
import os
import warnings
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from videollava.train.train import preprocess, preprocess_multimodal

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
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processor = processor
        print(len(self.dataset))
        self.bucket = {}
        self.use_extracted_features = use_extracted_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        videopath = data["videopath"]
        caption = data["caption"]

        if self.use_extracted_features:
            videopath = os.path.join(self.features_folder, videopath)
            video_input = torch.load(Path(videopath).with_suffix(".pt"), map_location="cpu")
        else:
            videopath = os.path.join(self.video_folder, videopath)
            video_input = self.processor([videopath], return_tensors="pt")["pixel_values"][0]

        videopath = os.path.join(self.video_folder, data["videopath"])
        source = self._convert_text_to_json(caption, videopath, index)
        text_input = self._extract_text_token_from_conversation([source])
        item = {
            "video": video_input,
            "text": text_input,
            "videopath": videopath,
            "caption": caption,
        }
        return item

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
                {"from": "human", "value": human_part},
                {"from": "gpt", "value": ai_part},
            ],
        }

        return data

    def _extract_text_token_from_conversation(self, sources):
        sources = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]), self.data_args
        )
        data_dict = preprocess(sources, self.tokenizer, has_image=True)

        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        return {"input_ids": input_ids, "labels": labels}
