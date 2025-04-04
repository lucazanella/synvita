import argparse
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="data directory")
parser.add_argument("--input_file_1", type=str, required=True, help="input file file")
parser.add_argument("--input_file_2", type=str, required=True, help="input file file")
parser.add_argument("--vid_per_caption", type=int, default=588, help="captions per video")
parser.add_argument("--output_file", type=str, required=True, help="output file")
parser.add_argument(
    "--extract_caption",
    action="store_true",
    help="Indicates that input_file_2 contains a caption. Extract the caption from the prompt in input_file_1.",
)
args = parser.parse_args()


def extract_description(conversation):
    pattern = r'Does this video entail the description: "(.*)"\?'
    match = re.search(pattern, conversation)
    if match:
        return match.group(1)
    else:
        raise ValueError("Description not found")


def create_labels(labels):
    res = []
    for i in range(len(labels)):
        tmp = []
        for j in range(len(labels[i])):
            if labels[i][j] == 1:
                tmp.append(j)
        res.append(tmp)
    res = np.array(res)
    return np.squeeze(res)


def main():
    df1 = pd.read_csv(args.input_file_1)  # videopath, caption, match
    if args.extract_caption:
        df1["caption"] = df1["caption"].apply(extract_description)
    print(len(df1))
    print(df1.head())

    # df2 = pd.read_csv(args.input_file_2, names = ['videopath', 'caption', 'entailment', 'pos', 'neg'])
    df2 = pd.read_csv(args.input_file_2, names=["videopath", "caption", "entailment"])
    print(len(df2))
    print(df2.head())

    df2["videopath"] = df2["videopath"].apply(lambda x: x.split(args.data_dir)[1])
    df = pd.merge(df1, df2, on=["videopath", "caption"], how="inner")
    print(len(df))
    df = df.sort_values(by=["videopath", "caption"])

    predictions = []
    labels = []
    captions = []

    for _, cap_df in df.groupby(["caption"]):
        if len(cap_df) == args.vid_per_caption:
            cap_df = cap_df.sort_values(by=["videopath", "caption"])
            predictions.append(cap_df["entailment"].tolist())
            labels.append(cap_df["match"].tolist())
            captions.append(cap_df["caption"].tolist())

    predictions = np.array(predictions)
    labels = np.array(labels)
    print(len(labels))
    print(sum(labels[0]))

    mAP = []
    r1 = []
    r5 = []
    r10 = []
    for j in tqdm(range(len(predictions))):
        pred = predictions[j]
        # pred = np.random.rand(len(predictions[j]))
        ranks = np.argsort(-pred)
        hit_precision = []
        hits = 0
        for i in range(len(ranks)):
            if labels[j][ranks[i]] == 1:
                hits += 1
                hit_precision.append(hits / (i + 1))
        average_precision = sum(hit_precision) / len(hit_precision)
        mAP.append(average_precision)

        count = 0
        for i in range(len(ranks[:10])):
            if labels[j][ranks[i]] == 1:
                count = 1
            if i == 0:
                r1.append(count)
            if i == 4:
                r5.append(count)
            if i == 9:
                r10.append(count)

    mean_ap = sum(mAP) / len(mAP)
    r1 = sum(r1) / len(r1)
    r5 = sum(r5) / len(r5)
    r10 = sum(r10) / len(r10)

    print(f"mAP: {100 * mean_ap} | R@1: {100 * r1} | R@5: {100 * r5} | R@10: {100 * r10}")

    with open(args.output_file, "w") as f:
        f.write(f"mAP: {100 * mean_ap} | R@1: {100 * r1} | R@5: {100 * r5} | R@10: {100 * r10}\n")


if __name__ == "__main__":
    main()
