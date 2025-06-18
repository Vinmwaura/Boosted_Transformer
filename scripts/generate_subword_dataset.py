import os
import csv
import json
import logging
import pathlib
import argparse

import torch

# Compute matching mask for subword groupings.
def compute_matching_mask_indices(
        window_size,
        vocab_elemental_tensor,
        subwords_groupings_tensor):
    # Boolean mask where new subword pair matches in subword.
    matching_mask = torch.eq(
        subwords_groupings_tensor,
        vocab_elemental_tensor).all(dim=1)  # (num_subword_pairs,)

    # Total count of matched subword groupings.
    total_matched = matching_mask.float().sum().item()

    if total_matched > 1:
        """
        HACK: Ensure certain 1 values are set to 0 to deal with cases of repeating patterns.
        The way unfold operation works with a stride of 1 can cause issues where repeating patterns
        occur. Remove invalid 1's in the matching mask.
        """
        # Get all indices of matching_mask for non-zero values i.e 1.
        matched_indices = torch.nonzero(matching_mask).squeeze(dim=-1)  # (num_matches,)

        # Get pairs of matched indices.
        matched_indices_unfold = matched_indices.unfold(
            dimension=0,
            size=2,
            step=1)  # (num_matches_pairs,2)

        # How close are the 1's together, need to maintain a min distance to be valid.
        diff_matched_indices = matched_indices_unfold[:,1] - matched_indices_unfold[:,0]  # (num_matched_pairs,)

        # Generate a Boolean mask where 1's are at invalid positions.
        invalid_mask = (diff_matched_indices < window_size)  # (num_matches_pairs,)

        # Get invalid indices of pairs where 1's are.
        invalid_matched_indices = matched_indices_unfold[invalid_mask]  # (K,2)

        invalid_matched_indices = invalid_matched_indices.flatten()  # Flatten.
        invalid_matched_indices = invalid_matched_indices.unique()  # Remove duplicates.
        invalid_matched_indices, _ = torch.sort(invalid_matched_indices)  # Ensure indices are sorted.

        # Set invalid indices in matching mask to 0, not to be considered.
        matching_mask[invalid_matched_indices] = 0

    # Get indices where matching mask is 1.
    matching_mask_indices = torch.nonzero(matching_mask)

    return matching_mask_indices

# Merge characters to form subwords.
def merge_subwords_groupings(
        device,
        sorted_vocabs,
        vocab_elementals,
        subwords_indices_tensor):
    stride = 1  # Stride for sliding window.

    len_vocab_elementals = len(vocab_elementals)

    for vocab_index, vocab_elemental_tuple in enumerate(vocab_elementals[:]):
        logging.info("=" * 100)
        logging.info(f"Processing {repr(sorted_vocabs[vocab_index])} : {vocab_index + 1:,} / {len_vocab_elementals:,}")

        vocab_elemental, merged_subword_index = vocab_elemental_tuple

        # Length of Vocabulary item being searched for.
        window_size = len(vocab_elemental)

        # Skip single characters.
        if window_size < 2:
            continue

        # Convert list of indices to tensor.
        vocab_elemental_tensor = torch.tensor(
            [vocab_elemental],
            device=device)  # (1,window_size)

        # Simulate sliding window operation to get all the possible subword groupings.
        subwords_groupings_tensor = subwords_indices_tensor.unfold(
            dimension=0,
            size=window_size,
            step=stride)  # (num_subwords_groupings,window_size)

        # Get indices where subwords start in subwords_list.
        matching_mask_indices = compute_matching_mask_indices(
            window_size=window_size,
            vocab_elemental_tensor=vocab_elemental_tensor,
            subwords_groupings_tensor=subwords_groupings_tensor)

        # Merge subwords grouping together.
        subwords_indices_tensor[matching_mask_indices.squeeze(dim=-1)] = merged_subword_index

        # Remove unwanted elemental characters still remaining after merge operation.
        window_range = torch.arange(
            start=1,
            end=window_size,
            device=device).unsqueeze(dim=0)  # (1, window_size-1)

        # Indices where unwanted elements exist.
        unwanted_indices = matching_mask_indices + window_range
        unwanted_indices = unwanted_indices.flatten()

        # Invert the unwanted indices to get what to keep.
        wanted_indices = torch.ones_like(subwords_indices_tensor, dtype=torch.bool)
        wanted_indices[unwanted_indices] = False

        # Filter the subwords indices to keep the valid items.
        subwords_indices_tensor = subwords_indices_tensor[wanted_indices]

        logging.info("=" * 100)

    return subwords_indices_tensor

# Iterate over each individual text file, aggregate subwords and save json file.
def generate_dataset(
        device,
        out_path,
        vocab_list,
        filepaths_list):
    dataset_paths_list = []

    vocabulary_path = os.path.join(
        out_path,
        "vocabulary.json")
    with open(vocabulary_path, "w") as json_f:
        json.dump({"vocab": vocab_list}, json_f)

    set_vocab = set(vocab_list)

    # Map of Vocabulary subwords to their indices for quick lookup.
    vocab_index_map = {}
    for vocab_index, vocab_subword in enumerate(vocab_list):
        vocab_index_map[vocab_subword] = vocab_index

    # Sort Vocabularies by subword length in descending order.
    sorted_vocab_list = sorted(
        vocab_list,
        key=len,
        reverse=True)

    # Break down each Vocabulary subword into individual elemental subwords e.g ["Kenya"] --> ["K","e","n","y","a"].
    vocab_elementals_indices_list = []
    for sorted_vocab in sorted_vocab_list:
        vocab_elemental_chars = list(sorted_vocab)
        vocab_elemental_indices = [vocab_index_map[vocab_elemental_char] for vocab_elemental_char in vocab_elemental_chars]
        vocab_elementals_indices_list.append((vocab_elemental_indices, vocab_list.index(sorted_vocab)))

    len_filepaths = len(filepaths_list)

    for filepath_index, filepath_list in enumerate(filepaths_list):
        filepath = filepath_list[0]
        filename = os.path.basename(filepath).split(".")[0]  # Assumes filename has file extension e.g .txt

        logging.info("*" * 100)
        logging.info(f"Processing Dataset: {filepath} ({filepath_index + 1:,} / {len_filepaths:,})")

        # Load entire text from file into memory.
        with open(filepath, "r") as txt_f:
            og_text = txt_f.read()
        
        set_og_text = set(og_text)

        # Check if set of dataset is a subset of Vocabulary i.e text has only characters found in Vocabulary.
        is_subset = (set_og_text <= set_vocab)

        # TODO: Avoid this all together by generating a proper Vocabulary.
        if not is_subset:
            logging.info(f"Skipping...")
            logging.info("*" * 100)
            continue

        # Break down text into individual characters.
        subwords_char_list = list(og_text)

        # Get indices of characters from Vocabulary.
        subwords_indices_list = [vocab_index_map[subword_char] for subword_char in subwords_char_list]

        # Convert list of integers into tensors.
        subwords_indices_tensor = torch.tensor(
            subwords_indices_list,
            device=device)  # (N,)

        # Merge elemental indices groupings into subword indices.
        new_subwords_indices_tensor = merge_subwords_groupings(
            device=device,
            sorted_vocabs=sorted_vocab_list,
            vocab_elementals=vocab_elementals_indices_list,
            subwords_indices_tensor=subwords_indices_tensor)

        # Check if generated subword indices is valid when converted back to text.
        new_subwords_indices_list = new_subwords_indices_tensor.tolist()
        new_text = "".join([vocab_list[subword_index] for subword_index in new_subwords_indices_list])

        is_valid = (new_text == og_text)

        logging.info(f"Is valid: {is_valid}")

        # If valid, save as json file.
        if is_valid:
            out_fpath = os.path.join(
                out_path,
                f"{filename}.json")
            dataset_paths_list.append(out_fpath)

            try:
                json_data = {
                    "data": new_subwords_indices_list}

                with open(out_fpath, "w") as json_f:
                    json.dump(json_data, json_f)

                logging.info(f"Successfuly saved: {out_fpath}")
            except Exception as e:
                raise e

        # Needs to be manually split into train and test segments.
        csv_dest_path = os.path.join(out_path, "dataset.csv")

        logging.info(f"Saving CSV Dataset to : {csv_dest_path}")
        with open(csv_dest_path, "w") as f:
            writer = csv.writer(f)
            for dataset_fpath in dataset_paths_list:
                writer.writerow([dataset_fpath])
        logging.info("*" * 100)

def main():
    parser = argparse.ArgumentParser(
        description="Generate subword dataset (*.json) from text files.")

    parser.add_argument(
        "--device",
        help="Which hardware device model will run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--dataset-path",
        help="Filepath to csv file containing list of .txt files.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--vocab-path",
        help="Filepath to Vocabulary json file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-path",
        help="Destination output path for json file.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    device = args["device"]
    dataset_path = args["dataset_path"]
    vocab_path = args["vocab_path"]
    out_path = args["out_path"]
    try:
        os.makedirs(out_path, exist_ok=True)
    except Exception as e:
        raise e

    # Log file path.
    log_path = os.path.join(
        out_path,
        "subword_dataset.log")

    # Logs Info to parent directory.
    logging.basicConfig(
        # filename=log_path,
        format="%(asctime)s %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG,
        force=True)

    # Load Vocabulary json file and csv dataset files.
    with open(vocab_path, "r") as vocab_f, open(dataset_path, "r") as dataset_f:
        vocab_json = json.load(vocab_f)

        csv_reader = csv.reader(dataset_f)
        filepaths_list = list(csv_reader)

    vocab_list = vocab_json["vocab"]

    # Generate subword dataset from text files. Assumes text file can fit into memory.
    generate_dataset(
        device=device,
        out_path=out_path,
        vocab_list=vocab_list,
        filepaths_list=filepaths_list)

if __name__ == "__main__":
    main()
