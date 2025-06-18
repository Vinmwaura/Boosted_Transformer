import os
import json
import pathlib
import logging
import argparse

import torch

# Get Most Frequent character pairs.
def get_most_frequent_pair(
        device,
        subword_pairs):    
    # Returns unique subwords and their counts.
    unique_subword_pairs, unique_subword_pairs_counts = torch.unique(
        subword_pairs,
        dim=0,
        return_counts=True)  # (N,num_groupings)

    most_frequent_count, most_frquent_index = unique_subword_pairs_counts.max(dim=0)
    most_frequent_pair = unique_subword_pairs[most_frquent_index].tolist()

    return most_frequent_count, most_frequent_pair

# Compute matching mask for subword pairs.
def compute_matching_mask(
        device,
        subword_pairs,
        most_frequent_pair):
    # Most frequent pair e.g Vobulary indices of ["=","="].
    most_frequent_pair = torch.tensor(
        [most_frequent_pair],
        device=device)  # (1,2)

    # Boolean mask where new subword pair matches in subword.
    matching_mask = torch.eq(
        subword_pairs,
        most_frequent_pair).all(dim=1).to(torch.int16)  # (num_subword_pairs,)

    # HACK: Ensure any immediately succeeding 1's is set to 0 to deal with cases of repeating patterns.
    # Get all indices of matching_mask for non-zero values i.e 1.
    matched_indices = torch.nonzero(matching_mask).squeeze(dim=1)  # (num_matches,)

    # Get pairs of matched indices.
    matched_indices_unfold = matched_indices.unfold(
        dimension=0,
        size=2,
        step=1)  # (num_matches_pairs,2)

    # Difference between indices, if 1 then they are neighbours.
    diff_matched_indices = matched_indices_unfold[:,1] - matched_indices_unfold[:,0]

    # Generate a mask where 1's are grouped together.
    neighbouring_mask = (diff_matched_indices == 1)

    # Get indices where repetition of 1's occur close together.
    neighbouring_indices = torch.nonzero(neighbouring_mask).squeeze(dim=1)

    # Get repetition pairs where 1's are neighbouring each other.
    neighbouring_matched_indices = matched_indices_unfold[neighbouring_indices]

    # Convert repetition pairs tensor to list.
    neighbouring_matched_indices_list = neighbouring_matched_indices.tolist()

    # Iterate over entire neighbouring matched indices pair and replacing the next one to 0.
    for neighbouring_matched_indices_pair in neighbouring_matched_indices_list:
        prev_index, next_index = neighbouring_matched_indices_pair
        if (matching_mask[prev_index].item() == 1 and matching_mask[next_index].item() == 1):
            matching_mask[next_index] = 0

    # Pad 0 at the end to ensure mask is same length as input.
    pad_value = torch.zeros([1], device=device)
    matching_mask = torch.cat(
        (matching_mask, pad_value),
        dim=0)

    return matching_mask

# Merge frequent subword pairs.
def merge_pair(
        new_subword_index,
        matching_mask,
        subwords_indices):
    # Compute new subwords indices.
    new_subwords_indices = (
        subwords_indices * (1 - matching_mask)
    ) + (
            matching_mask * new_subword_index
        )

    # Shift mask to the right by one to get the unwanted subwords left over.
    unwanted_mask = torch.roll(
        matching_mask,
        shifts=1)

    # Invert the mask and convert to bool to get mask of indices to keep.
    wanted_mask = 1 - unwanted_mask
    wanted_mask_bool = wanted_mask.bool()

    # Keep useful data.
    new_subwords_indices = new_subwords_indices[wanted_mask_bool]
    return new_subwords_indices

# Save generated Vocabulary.
def save_vocabulary(
        out_path,
        sorted_vocabs,
        original_text,
        subwords_indices_list):
    # Sanity check to ensure at no point deviations occured.
    generated_text = "".join([sorted_vocabs[subword_index] for subword_index in subwords_indices_list])

    is_valid_vocabulary = (generated_text == original_text)

    len_vocab = len(sorted_vocabs)

    logging.info("*" * 100)
    logging.info("Saving data!")
    logging.info(f"Is Vocabulary valid? {is_valid_vocabulary}")
    logging.info(f"Length of unique vocab: {len_vocab:,}")
    logging.info("*" * 100)

    vocabulary = {"vocab": sorted_vocabs}

    dest_path = os.path.join(
        out_path,
        str(len_vocab))
    try:
        os.makedirs(dest_path, exist_ok=True)
    except Exception as e:
        raise e

    vocabulary_path = os.path.join(
        dest_path,
        "vocabulary.json")
    with open(vocabulary_path, "w") as f:
        json.dump(vocabulary, f)

def main():
    parser = argparse.ArgumentParser(
        description="Generate Subword Vocabulary from text file.")

    parser.add_argument(
        "--device",
        help="Which hardware device model will run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--dataset-path",
        help="Filepath to text dataset.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--vocab-size",
        help="Max size of vocabulary.",
        type=int,
        default=2_048)
    parser.add_argument(
        "--out-path",
        help="Destination output path for Vocabulary json.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    device = args["device"]
    vocab_size = args["vocab_size"]
    dataset_path = args["dataset_path"]
    out_path = args["out_path"]
    try:
        os.makedirs(out_path, exist_ok=True)
    except Exception as e:
        raise e

    # Log file path.
    log_path = os.path.join(
        out_path,
        "vocabulary.log")
    
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

    # Dictionary parameters.
    stride = 1
    window_size = 2

    # TODO: Implement functionality to reload saved checkpoints to resume calculations.

    # Load text data from file.
    with open(dataset_path, "r", encoding='utf-8-sig') as file:
        original_text = file.read()

    # Separate text into a list of individual characters(Alphanumerical + Special character).
    subwords_chars = list(original_text)

    # Get unique characters.
    vocabs = list(set(subwords_chars))
    sorted_vocabs = sorted(vocabs)

    # Dictionary of vocabulary indices for quick lookup.
    vocab_index = {}
    for index, vocab_item in enumerate(sorted_vocabs):
        vocab_index[vocab_item] = index

    # Get indices of subwords from sorted vocabulary.
    subwords_indices_list = [vocab_index[subwords_char] for subwords_char in subwords_chars]

    # Iteration parameters.
    count_loop = 0
    max_count_loop = 100_000

    if vocab_size > max_count_loop:
        raise Exception("Vocab size is too large!")

    logging.info("Started generating subword Vocabulary.")
    logging.info(f"Init Vocabulary:\n{sorted_vocabs}")
    logging.info(f"Max Vocabulary Size: {vocab_size:,}.")

    # Convert list of integers into tensors (int16 ranges: -32,768 to 32,767) for memory efficiency.
    subwords_indices_tensor = torch.tensor(
        subwords_indices_list,
        device=device,
        dtype=torch.int16)  # (N,)

    # HACK: Remove large variable to free up memory for other tasks.
    del subwords_indices_list

    while True:
        # Size of Vocabulary.
        len_vocabs = len(sorted_vocabs)

        logging.info("*" * 100)
        logging.info(f"Length of unique vocab: {len_vocabs:,}")
        logging.info(f"Length of sub-words: {subwords_indices_tensor.numel():,}")

        # Save Vocabulary every time length of vocabulary is 2^x where x is a non-negative integer: (x > 0).
        if len_vocabs > 0 and (len_vocabs & (len_vocabs - 1)) == 0:
            subwords_indices_list = subwords_indices_tensor.tolist()

            save_vocabulary(
                out_path=out_path,
                sorted_vocabs=sorted_vocabs,
                original_text=original_text,
                subwords_indices_list=subwords_indices_list)

            # HACK: Remove large variable to free up memory for other tasks.
            del subwords_indices_list

        if len_vocabs >= vocab_size:
            # Convert tensor back to list.
            subwords_indices_list = subwords_indices_tensor.tolist()

            # Get list of unique indices from new subword list.
            vocabs_indices = list(set(subwords_indices_list))
            sorted_vocabs_indices = sorted(vocabs_indices)

            # Recompute Vocabulary to remove subwords not being utilized after merge operation.
            sorted_vocabs = [sorted_vocabs[sorted_vocabs_index] for sorted_vocabs_index in sorted_vocabs_indices]
            len_vocabs = len(sorted_vocabs)  # New length of Vocabulary.

            # Check if length of recomputed Vocabulary still meets the stop condition.
            if len_vocabs >= vocab_size:
                logging.info("Finished generating subword Vocabulary.")
                break
            else:
                logging.info("Not yet finished generating subword Vocabulary, needs more cooking...")

                # Recompute subword index list to reflect Vocabulary changes.
                sorted_vocab_index = {}
                for new_vocab_index, vocab_item_index in enumerate(sorted_vocabs_indices):
                    sorted_vocab_index[vocab_item_index] = new_vocab_index

                # Update indices of subwords to meet new Vocabulary.
                subwords_indices_list = [sorted_vocab_index[old_subword_index] for old_subword_index in subwords_indices_list]

                # Convert list of integers into tensors (int16 ranges: -32,768 to 32,767).
                subwords_indices_tensor = torch.tensor(
                    subwords_indices_list,
                    device=device,
                    dtype=torch.int16)  # (N,)

                # HACK: Remove large variable to free up memory for other tasks.
                del subwords_indices_list

        # Simulate sliding window operation to get all the possible subword pairings.
        subword_index_pairs = subwords_indices_tensor.unfold(
            dimension=0,
            size=window_size,
            step=stride)  # (num_subwords_pairs,window_size)

        # Get most frequent subword pairs and it's total count.
        max_count, most_frequent_pair = get_most_frequent_pair(
            device=device,
            subword_pairs=subword_index_pairs)

        subword_string_list = [sorted_vocabs[i] for i in most_frequent_pair]
        new_subword_string = "".join(subword_string_list)

        logging.info(f"Most frequent subwords pairs: {repr(new_subword_string)} | Count: {max_count:,}")

        # Boolean mask where new subword pair matches.
        matching_mask_tensor = compute_matching_mask(
            device=device,
            subword_pairs=subword_index_pairs,
            most_frequent_pair=most_frequent_pair)

        # Append new subword to Vocabulary, for merging task.
        sorted_vocabs.append(new_subword_string)

        # Get index of new subword at end of Vocabulary.
        new_subword_index = len(sorted_vocabs) - 1

        # Merge most frequent subword pairs to form one subword.
        subwords_indices_tensor = merge_pair(
            new_subword_index=new_subword_index,
            matching_mask=matching_mask_tensor,
            subwords_indices=subwords_indices_tensor)
        subwords_indices_tensor = subwords_indices_tensor.to(torch.int16)

        count_loop += 1

        # HACK: Prevent infinite loops in cases of wierd issues.
        if count_loop >= max_count_loop:
            logging.info("Reached maximum iteration allowed!")
            break

if __name__ == "__main__":
    main()
