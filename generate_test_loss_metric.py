import os
import csv
import glob
import json
import random
import pathlib
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Decoder_Transformer import DecoderTransformer

from dataset_loader.subword_dataset import SubWord_Dataset

from utils.model_utils import load_model

def main():
    project_name = "Test Loss"

    parser = argparse.ArgumentParser(
        description=f"{project_name}")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--tst-dataset-path",
        help="File path to testing csv dataset file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--batch-size",
        help="Batch size of dataset.",
        type=int,
        default=64)
    parser.add_argument(
        "--model-folder-path",
        help="Folder path to model checkpoints.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-dir",
        help="Folder path of output directory.",
        required=True)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    tst_dataset_path = args["tst_dataset_path"]  # Training csv file path (*.csv).
    batch_size = args["batch_size"]  # Batch size of training dataset.
    model_folder_path = args["model_folder_path"]  # Folder path to models saved.
    out_dir = args["out_dir"]  # Destination path for model.

    out_csv_file = os.path.join(
        out_dir,
        "loss.csv")

    model_regex = os.path.join(
        model_folder_path,
        "*.pt")
    model_paths = glob.glob(model_regex)
    len_model_paths = len(model_paths)

    models_list = []

    for index, model_path in enumerate(model_paths):
        print(f"Model: {index + 1:,} / {len_model_paths:,}, model_path: {model_path}")

        file_name = os.path.basename(model_path)

        model_index = file_name.split("_")[0]
        model_index = int(model_index)

        loaded_model_status, loaded_model_dict = load_model(model_path)
        if not loaded_model_status:
            raise Exception("An error occured while loading model checkpoint!")

        saved_models_list = loaded_model_dict["saved_models"]

        if len(models_list) == 0:
            # Model Params (From model checkpoints).
            num_heads = loaded_model_dict["num_heads"]
            num_models = loaded_model_dict["num_models"]
            hidden_dim = loaded_model_dict["hidden_dim"]    
            embedding_dim = loaded_model_dict["embedding_dim"]
            context_window = loaded_model_dict["context_window"]
            activation_type = loaded_model_dict["activation_type"]
            num_decoder_blocks = loaded_model_dict["num_decoder_blocks"]

            vocab = loaded_model_dict["vocab"]
            vocab_size = len(vocab)

            # TODO: Save this in vocab dictionary.
            start_token = vocab_size
            padding_token = vocab_size + 1

            # Datasets.
            tst_dataset = SubWord_Dataset(
                csv_fpath=tst_dataset_path,
                start_token=start_token,
                padding_token=padding_token,
                context_window=context_window)

            # Dataloaders.
            tst_dataloader = torch.utils.data.DataLoader(
                tst_dataset,
                batch_size=batch_size,
                num_workers=5,
                shuffle=True)

            for curr_model_index in range(num_models):
                temp_model = DecoderTransformer(
                    is_base=(curr_model_index == 0),  # First model is Base model, otherwise Secondary model.
                    num_embeddings=vocab_size + 2,  # Includes [START] and [PAD] tokens.
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    out_classes=vocab_size + 1,  # Includes only [PAD] tokens.
                    num_blocks=num_decoder_blocks,
                    activation_type=activation_type)

                curr_model_state_dict = saved_models_list[curr_model_index]["model"]
                temp_model.custom_load_state_dict(curr_model_state_dict)

                temp_model = temp_model.to(device)

                temp_models_dict = {}
                temp_models_dict["model"] = temp_model

                models_list.append(temp_models_dict)
        else:
            # Dataloaders.
            tst_dataloader = torch.utils.data.DataLoader(
                tst_dataset,
                batch_size=batch_size,
                num_workers=5,
                shuffle=True)

            for curr_model_index, curr_model_dict in enumerate(models_list):
                saved_model_state_dict = saved_models_list[curr_model_index]["model"]
                curr_model_dict["model"].custom_load_state_dict(saved_model_state_dict)

        all_test_loss = []
        for index, (tst_in_seq, tst_target_seq) in enumerate(tst_dataloader):
            # print(f"{index:,} / {len(tst_dataloader):,}")
            tst_in_seq = tst_in_seq.to(device)
            tst_target_seq = tst_target_seq.to(device)
            tst_hidden_dec = None

            temp_test_loss = []
            for curr_model_dict in models_list:
                curr_model = curr_model_dict["model"]

                curr_model.eval()

                with torch.no_grad():
                    tst_hidden_dec, tst_out_classifier = curr_model(
                        x=tst_in_seq,
                        x_hidden=tst_hidden_dec)  # (N,Seq,Class)

                    tst_target_seq_flat = tst_target_seq.flatten()  # (N*Seq,)
                    tst_out_classifier_flat = tst_out_classifier.flatten(
                        start_dim=0,
                        end_dim=1)  # (N*Seq,Class)

                    tst_classifier_loss = F.cross_entropy(
                        tst_out_classifier_flat,
                        tst_target_seq_flat,
                        ignore_index=len(vocab))
                    if torch.isnan(tst_classifier_loss):
                        raise Exception("NaN encountered during training.")

                tst_hidden_dec = tst_hidden_dec.clone().detach()

                temp_test_loss.append(tst_classifier_loss.item())

            all_test_loss.append(temp_test_loss)

        all_test_loss_np = np.array(all_test_loss)
        all_test_loss_np_avg = np.mean(all_test_loss_np, axis=0)

        test_avg_loss = all_test_loss_np_avg.tolist()

        row_data = [model_index] + test_avg_loss

        with open(out_csv_file, "a", newline="") as f:
            file_writer = csv.writer(f, delimiter=',')
            file_writer.writerow(row_data)

if __name__ == "__main__":
    main()
