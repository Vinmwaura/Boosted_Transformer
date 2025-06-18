import os
import csv
import json
import logging
import pathlib
import argparse

import numpy as np

def sliding_window_computation(data_list, window_size, stride=1):
    data_list = [data_list[i:i + window_size] for i in range(0, len(data_list) - window_size + 1, stride)]
    return data_list

def generate_dataset(
        tr_fpaths,
        tst_fpaths,
        out_path,
        sliding_window,
        stride):
    logging.info("-" * 100)

    # Train Data.
    tr_fpaths_list = []
    tr_dest_path = os.path.join(out_path, "train")
    for tr_fpath in tr_fpaths:
        tr_fpath = tr_fpath[0]
        with open(tr_fpath, "r") as tr_f:
            data_dict = json.load(tr_f)
            tr_all_data = data_dict["data"]

        tr_split_data_list = sliding_window_computation(
            data_list=tr_all_data,
            window_size=sliding_window,
            stride=stride)
        
        logging.info(f"Split training dataset: {tr_fpath} into {len(tr_split_data_list):,} segments.")

        # Get filename of *.json files.
        tr_fpath_list = tr_fpath.split("/") 
        tr_file_name = tr_fpath_list[-1]
        tr_file_name = tr_file_name.split(".")[0]

        # Create folder for storing split dataset from sliding window operation.
        tr_out_path = os.path.join(
            tr_dest_path,
            tr_file_name)
        try:
            os.makedirs(tr_out_path, exist_ok=True)
        except Exception as e:
            raise e

        # Iterate over entire split dataset and save them to memory.
        for tr_index, tr_split_data in enumerate(tr_split_data_list):
            tr_index_fpath = os.path.join(
                tr_out_path,
                f"{tr_index}.npy")

            tr_fpaths_list.append(tr_index_fpath)

            # Convert to numpy array.
            tr_data_numpy = np.array(tr_split_data)

            # Save numpy array to memory.
            np.save(tr_index_fpath, tr_data_numpy)

    logging.info("-" * 100)

    # Test Data.
    tst_fpaths_list = []
    tst_dest_path = os.path.join(out_path, "test")
    for tst_fpath in tst_fpaths:
        tst_fpath = tst_fpath[0]
        with open(tst_fpath, "r") as tst_f:
            data_dict = json.load(tst_f)
            tst_all_data = data_dict["data"]

        tst_split_data_list = sliding_window_computation(
            data_list=tst_all_data,
            window_size=sliding_window,
            stride=stride)
        
        logging.info(f"Split testing dataset: {tst_fpath} into {len(tst_split_data_list):,} segments.")

        # Get filename of *.json files.
        tst_fpath_list = tst_fpath.split("/") 
        tst_file_name = tst_fpath_list[-1]
        tst_file_name = tst_file_name.split(".")[0]

        # Create folder for storing split dataset from sliding window operation.
        tst_out_path = os.path.join(
            tst_dest_path,
            tst_file_name)
        try:
            os.makedirs(tst_out_path, exist_ok=True)
        except Exception as e:
            raise e

        # Iterate over entire split dataset and save them to memory.
        for tst_index, tst_split_data in enumerate(tst_split_data_list):
            tst_index_fpath = os.path.join(
                tst_out_path,
                f"{tst_index}.npy")

            tst_fpaths_list.append(tst_index_fpath)

            # Convert to numpy array.
            tst_data_numpy = np.array(tst_split_data)

            # Save numpy array to memory.
            np.save(tst_index_fpath, tst_data_numpy)

    logging.info("-" * 100)

    # Save file paths to csv files.
    train_csv_fpath = os.path.join(out_path, "train_file_list.csv")
    test_csv_fpath = os.path.join(out_path, "test_file_list.csv")

    with open(train_csv_fpath, "w") as tr_f, open(test_csv_fpath, "w") as tst_f:
        logging.info(f"Saving {len(tr_fpaths_list):,} training filepaths into {train_csv_fpath}.")
        tr_writer = csv.writer(tr_f)
        for tr_fpath in tr_fpaths_list:
            tr_writer.writerow([tr_fpath])

        logging.info(f"Saving {len(tst_fpaths_list):,} testing filepaths into {test_csv_fpath}.")
        tst_writer = csv.writer(tst_f)
        for tst_fpath in tst_fpaths_list:
            tst_writer.writerow([tst_fpath])

    logging.info("-" * 100)

def main():
    parser = argparse.ArgumentParser(
        description="Generate split train/test dataset (*.np) from subword dataset.")

    parser.add_argument(
        "--sliding-window",
        help="Sliding window to split the dataset into.",
        default=256,
        type=int)
    parser.add_argument(
        "--stride",
        help="Stride value for sliding window.",
        default=1,
        type=int)
    parser.add_argument(
        "--tr-dataset-path",
        help="Filepath to csv file containing list of .txt files for train dataset.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--tst-dataset-path",
        help="Filepath to csv file containing list of .txt files for test dataset.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-path",
        help="Destination output path for numpy and csv files.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    sliding_window = args["sliding_window"]
    stride = args["stride"]
    tr_dataset_path = args["tr_dataset_path"]
    tst_dataset_path = args["tst_dataset_path"]
    out_path = args["out_path"]
    try:
        os.makedirs(out_path, exist_ok=True)
    except Exception as e:
        raise e

    # Log file path.
    log_path = os.path.join(
        out_path,
        "training_dataset.log")

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

    # Load csv dataset files.
    with open(tr_dataset_path, "r") as tr_f, open(tst_dataset_path, "r") as tst_f:
        csv_tr_reader = csv.reader(tr_f)
        csv_tst_reader = csv.reader(tst_f)

        tr_filepaths_list = list(csv_tr_reader)
        tst_filepaths_list = list(csv_tst_reader)

    # Generate train/test dataset using sliding window to make it easier to load.
    generate_dataset(
        tr_fpaths=tr_filepaths_list,
        tst_fpaths=tst_filepaths_list,
        out_path=out_path,
        sliding_window=sliding_window,
        stride=stride)

if __name__ == "__main__":
    main()
