#!/usr/bin/env python3
"""
Prepare wildguard_RL.csv for GRPO (online RL) training.
Creates dataset with prompt (conversational) + ground_truth for reward computation.
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/wildguard_RL.csv",
        help="Path to wildguard_RL.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ft_dataset/SAD",
        help="Output path for GRPO dataset",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.05,
        help="Fraction of data for evaluation (default: 5%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)

    def make_grpo_example(row):
        prompt = [
            {"role": "system", "content": row["system_msg"]},
            {"role": "user", "content": row["user_msg"]},
        ]
        gt = str(row["ground_truth"]).strip()
        if gt not in ("Yes", "No"):
            raise ValueError(f"Invalid ground_truth: {gt}")
        return {"prompt": prompt, "ground_truth": gt}

    print("Converting to GRPO format...")
    grpo_data = [make_grpo_example(row) for _, row in df.iterrows()]
    dataset = Dataset.from_list(grpo_data)

    if args.test_split > 0:
        split = dataset.train_test_split(test_size=args.test_split, seed=args.seed)
        dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})
        print(f"Train: {len(split['train'])}, Test: {len(split['test'])}")
    else:
        dataset_dict = DatasetDict({"train": dataset})
        print(f"Train: {len(dataset)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    print(f"Saved GRPO dataset to {output_path}")


if __name__ == "__main__":
    main()
