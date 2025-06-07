import os
import csv
import json
import random
import pathlib
import logging
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Decoder_Transformer import DecoderTransformer

from dataset_loader.subword_dataset import SubWord_Dataset

from utils.model_utils import (
    save_model,
    load_model)

# HACK: Force the RNG to be deterministic to help when comparing model's performances.
# Too time consuming to properly train model(s) multiple times to get a proper measure.
seed_value = 69  # Nice!

torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

def checkpoint_model(
        data_dict,
        models_list,
        out_dir):
    model_state_list = []

    for model_dict in models_list:
        temp_state_dict = {
            "model": model_dict["model"].state_dict(),
            "optim": model_dict["optim"].state_dict()}
        model_state_list.append(temp_state_dict)

    global_steps = data_dict["global_steps"]

    # Save model.
    model_dict = {
        **data_dict,
        "saved_models": model_state_list}

    save_status = save_model(
        model_dict=model_dict,
        dest_path=out_dir,
        init_folder=True,
        file_name=f"{global_steps}_model.pt",
        logging=logging.info)
    if save_status is True:
        logging.info("Successfully saved model.")
    else:
        logging.info("Error occured saving model.")

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.001:
        raise argparse.ArgumentTypeError("%r not in range > 0.1"%(x,))
    return x

def main():
    project_name = "Boosted Transformers <Base+Secondary model(s)>"

    parser = argparse.ArgumentParser(
        description=f"{project_name}")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--vocab-dataset-path",
        help="File path to Vocab json dataset file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--tr-dataset-path",
        help="File path to training csv dataset file.",
        required=True,
        type=pathlib.Path)
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
        "--checkpoint-steps",
        help="Steps for checkpointing and/or testing model.",
        type=int,
        default=1_000)
    parser.add_argument(
        "--config-path",
        help="File path to config json file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to model checkpoint.",
        required=False,
        default=None,
        type=pathlib.Path)
    parser.add_argument(
        "--lr-steps",
        help="Global steps in between halving learning rate.",
        default=50_000,
        type=int)
    parser.add_argument(
        "--max-global-steps",
        help="Max Global steps for training.",
        default=250_000,
        type=int)
    parser.add_argument(
        "--load-optim",
        action='store_true',
        help="Load model's optimizer's weights and parameters, if loading model.")
    parser.add_argument(
        "--out-dir",
        help="Folder path of output directory.",
        required=True)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    lr_steps = args["lr_steps"]  # Global steps in between halving learning rate.
    max_global_steps = args["max_global_steps"]
    load_optim = args["load_optim"]  # Reload saved optimizer weights.
    vocab_dataset_path = args["vocab_dataset_path"]  # Vocabulary json file path (*.json).
    tr_dataset_path = args["tr_dataset_path"]  # Training csv file path (*.csv).
    tst_dataset_path = args["tst_dataset_path"]  # Training csv file path (*.csv).
    batch_size = args["batch_size"]  # Batch size of training dataset.
    config_path = args["config_path"]
    model_checkpoint = args["model_checkpoint"]
    checkpoint_steps = args["checkpoint_steps"]  # Steps to checkpoint model.
    out_dir = args["out_dir"]  # Destination path for model.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    # Log file path.
    log_path = os.path.join(
        out_dir,
        f"{project_name}.log")

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

    # Load JSON dataset.
    with open(vocab_dataset_path, "r") as json_f:
        vocab_json_dataset = json.load(json_f)

    # Vocabulary / Vocabulary size of NLP dataset.
    vocab = vocab_json_dataset["vocab"]
    vocab_size = len(vocab)

    # TODO: Save this in vocab dictionary.
    start_token = vocab_size
    padding_token = vocab_size + 1

    # Training params.
    global_steps = 0

    # Load config from json file.
    logging.info("Loading json Config...")

    with open(config_path, 'r') as json_file:
        json_data = json_file.read()
    config_dict = json.loads(json_data)

    # Model Params (From config file).
    model_lr = config_dict["model_lr"]
    lr_gamma_list = config_dict["lr_gamma"]
    num_models = config_dict["num_models"]

    # Model Params (From config file).
    num_heads = config_dict["num_heads"]
    hidden_dim = config_dict["hidden_dim"]
    embedding_dim = config_dict["embedding_dim"]
    context_window = config_dict["context_window"]
    activation_type = config_dict["activation_type"]
    num_decoder_blocks = config_dict["num_decoder_blocks"]

    saved_models_list = None
    if model_checkpoint is not None:
        # Load Transformer Model checkpoints.
        logging.info("Loading Model...")

        loaded_model_status, loaded_model_dict = load_model(model_checkpoint)
        if not loaded_model_status:
            raise Exception("An error occured while loading model checkpoint!")

        saved_models_list = loaded_model_dict["saved_models"]

    models_list = []
    for model_index in range(num_models):
        temp_model = DecoderTransformer(
            is_base=(model_index == 0),  # First model is Base model, otherwise Secondary model.
            num_embeddings=vocab_size + 2,  # Includes [START] and [PAD] tokens.
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            out_classes=vocab_size + 1,  # Includes only [PAD] tokens.
            num_blocks=num_decoder_blocks,
            activation_type=activation_type)

        if saved_models_list is not None:
            try:
                curr_model_state_dict = saved_models_list[model_index]["model"]
                temp_model.custom_load_state_dict(curr_model_state_dict)
            except IndexError:
                logging.info(f"No saved model for model_{model_index}, Skipped!")
            except Exception as e:
                raise e

        temp_model = temp_model.to(device)

        temp_model_optim = torch.optim.Adam(
            temp_model.parameters(),
            lr=model_lr,
            betas=(0.5, 0.999))

        # Load Optimizer params if allowed.
        if saved_models_list is not None and load_optim:
            logging.info("Resuming Training using saved optimizer weights...")
            try:
                curr_model_optim_state_dict = saved_models_list[model_index]["optim"]
                temp_model_optim.custom_load_state_dict(curr_model_optim_state_dict)
            except IndexError:
                logging.info(f"No saved model_optim for model_{model_index}, Skipped!")
            except Exception as e:
                raise e

        # Learning Rate Scheduler.
        lr_gamma = lr_gamma_list[model_index]
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=temp_model_optim,
            step_size=lr_steps,
            gamma=lr_gamma)

        temp_models_dict = {}
        temp_models_dict["model"] = temp_model
        temp_models_dict["optim"] = temp_model_optim
        temp_models_dict["lr_scheduler"] = lr_scheduler

        models_list.append(temp_models_dict)

    # Training and Testing Datasets.
    tr_dataset = SubWord_Dataset(
        csv_fpath=tr_dataset_path,
        start_token=start_token,
        padding_token=padding_token,
        context_window=context_window)
    tst_dataset = SubWord_Dataset(
        csv_fpath=tst_dataset_path,
        start_token=start_token,
        padding_token=padding_token,
        context_window=context_window)

    # Training and Testing Dataloaders.
    tr_dataloader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True)
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True)
    tst_iterator = iter(tst_dataloader)

    # Model Params size.
    models_params_size = []
    for model_dict in models_list:
        model_params_size = sum(param.numel() for param in model_dict["model"].parameters())
        models_params_size.append(model_params_size)

    # https://pytorch.org/docs/stable/amp.html
    scaler = torch.cuda.amp.GradScaler()

    logging.info(f"{project_name}")
    logging.info(f"Output Directory: {out_dir}")
    logging.info("#" * 100)
    logging.info("Dataset Parameters.")
    logging.info(f"Vocab Size: {vocab_size:,}")
    logging.info(f"Train Batch Size: {batch_size:,}")
    logging.info(f"Context window: {context_window:,}")
    logging.info(f"Total Train Dataset: {len(tr_dataset):,}")
    logging.info(f"Total Test Dataset: {len(tst_dataset):,}")
    logging.info("#" * 100)
    logging.info("Model Parameters.")
    logging.info(f"Number of Models: {num_models:,}")
    for model_index, model_param_size in enumerate(models_params_size):
        logging.info(f"Model_{model_index:,} Param size: {model_param_size:,}")
    logging.info(f"Number of heads: {num_heads:,}")
    logging.info(f"Number of Decoder Blocks: {num_decoder_blocks:,}")
    logging.info(f"Embedding Dimension: {embedding_dim:,}")
    logging.info(f"Hidden Dimension: {hidden_dim:,}")
    logging.info(f"Activation Type: {activation_type}")
    for model_index, model_dict in enumerate(models_list):
        logging.info(f"Model_{model_index:,} Learning Rate: {model_dict["lr_scheduler"].optimizer.param_groups[0]['lr']:.3E} | Gamma Value: {lr_gamma_list[model_index]:,}")
    logging.info("#" * 100)
    logging.info("Training Parameters.")
    logging.info(f"Step: {global_steps:,}")
    logging.info(f"Max Global step: {max_global_steps:,}")
    logging.info(f"Learning rate decay step size: {lr_steps:,}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps:,}")
    logging.info("#" * 100)

    model_data_dict = {
        "vocab": vocab,
        "num_heads": num_heads,
        "num_models": num_models,
        "hidden_dim": hidden_dim,
        "global_steps": global_steps,
        "embedding_dim": embedding_dim,
        "context_window": context_window,
        "activation_type": activation_type,
        "num_decoder_blocks": num_decoder_blocks}

    # Training starts here.
    stop_training = False
    while not stop_training:
        for index, (in_seq, target_seq) in enumerate(tr_dataloader):
            global_steps = global_steps + 1

            # Training Data.
            in_seq = in_seq.to(device)  # (N,Seq)
            target_seq = target_seq.to(device)  # (N,Seq)
            hidden_dec = None

            all_tr_losses = []
            all_tr_correct = []
            for model_dict in models_list:
                curr_model = model_dict["model"]
                curr_model_optim = model_dict["optim"]
                curr_lr_scheduler = model_dict["lr_scheduler"]

                curr_model.train(mode=True)

                # Train Classifier.
                # Runs the forward pass under ``autocast``.
                with torch.autocast(device_type=device, dtype=torch.float16):
                    hidden_dec, out_classifier = curr_model(
                        x=in_seq,
                        x_hidden=hidden_dec)  # (N,Seq,Class)

                    target_seq_flat = target_seq.flatten()  # (N*Seq,)
                    out_classifier_flat = out_classifier.flatten(
                        start_dim=0,
                        end_dim=1)  # (N*Seq,Class)

                    tr_loss = F.cross_entropy(
                        out_classifier_flat,
                        target_seq_flat,
                        ignore_index=len(vocab))

                    if torch.isnan(tr_loss):
                        raise Exception("NaN encountered during training.")

                # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
                scaler.scale(tr_loss).backward()

                scaler.step(curr_model_optim)

                # Updates the scale for next iteration.
                scaler.update()

                curr_model_optim.zero_grad()

                curr_lr_scheduler.step()

                all_tr_losses.append(tr_loss.item())

                hidden_dec = hidden_dec.clone().detach()

            # Test Classifier.
            all_tst_losses = []
            try:
                tst_in_seq, tst_target_seq = next(tst_iterator)
            except StopIteration:
                tst_iterator = iter(tst_dataloader)
                tst_in_seq, tst_target_seq = next(tst_iterator)

            tst_in_seq = tst_in_seq.to(device)
            tst_target_seq = tst_target_seq.to(device)
            tst_hidden_dec = None

            for model_dict in models_list:
                curr_model = model_dict["model"]

                curr_model.eval()

                with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
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

                all_tst_losses.append(tst_classifier_loss.item())

                tst_hidden_dec = tst_hidden_dec.clone().detach()

            curr_lr_list = []
            for model_dict in models_list:
                curr_lr_list.append(model_dict["lr_scheduler"].optimizer.param_groups[0]["lr"])

            log_message = "Cum. Steps: {:,} | Steps: {:,} / {:,} | Train Classifier Loss: {} | Test Classifier Loss: {} | LR: {}".format(
                global_steps,
                index + 1,
                len(tr_dataloader),
                [f"{tr_loss:,.5f}" for tr_loss in all_tr_losses],
                [f"{tst_loss:,.5f}" for tst_loss in all_tst_losses],
                [f"{curr_lr:.3E}" for curr_lr in curr_lr_list])

            logging.info(log_message)

            # Checkpoint and test model.
            if global_steps % checkpoint_steps == 0 or global_steps == 1:
                model_data_dict["global_steps"] = global_steps

                checkpoint_model(
                    data_dict=model_data_dict,
                    out_dir=out_dir,
                    models_list=models_list)

            # Stop training when stopping criteria is met.
            if global_steps >= max_global_steps:
                stop_training = True
                break

        # Final checkpoint model.
        model_data_dict["global_steps"] = global_steps
        checkpoint_model(
            data_dict=model_data_dict,
            out_dir=out_dir,
            models_list=models_list)

if __name__ == "__main__":
    main()
