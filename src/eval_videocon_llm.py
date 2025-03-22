import argparse
import os
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="data directory")
    parser.add_argument(
        "--input_csv_1", type=str, required=True, help="input csv file 1: ground truth"
    )
    parser.add_argument(
        "--input_csv_2", type=str, required=True, help="input csv file 2: prediction"
    )
    parser.add_argument("--output_file", type=str, required=True, help="output file")
    parser.add_argument(
        "--extract_caption",
        action="store_true",
        help="Indicates that input_csv_2 contains a caption. Extract the caption from the prompt in input_csv_1.",
    )
    return parser.parse_args()


def extract_description(conversation):
    pattern = r'Does this video entail the description: "(.*)"\?'
    match = re.search(pattern, conversation)
    if match:
        return match.group(1)
    else:
        raise ValueError("Description not found")


def main(data_dir, input_csv_1, input_csv_2, output_file):
    df1 = pd.read_csv(input_csv_1)
    df2 = pd.read_csv(input_csv_2, names=["videopath", "caption", "entailment"])
    df1 = df1.drop_duplicates()
    df2 = df2.drop_duplicates()
    if args.extract_caption:
        df1["caption"] = df1["caption"].apply(extract_description)
    print(len(df1), len(df2))

    # df1['videopath'] = df1['videopath'].apply(lambda x: os.path.join(data_dir, x))
    df2["videopath"] = df2["videopath"].apply(lambda x: x.split(data_dir)[1])
    df = pd.merge(df1, df2, on=["videopath", "caption"], how="inner")
    df = df.drop_duplicates()
    print(len(df))

    auc_roc_score = roc_auc_score(df["label"], df["entailment"])
    print(f"AUC-ROC: {100 * auc_roc_score}")
    with open(output_file, "w") as f:
        f.write(f"AUC-ROC: {(100 * auc_roc_score):.2f}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.input_csv_1, args.input_csv_2, args.output_file)
