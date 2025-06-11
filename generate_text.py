import pathlib
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Decoder_Transformer import DecoderTransformer

from utils.model_utils import load_model

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.1:
        raise argparse.ArgumentTypeError("%r not in range > 0.1"%(x,))
    return x

def main():
    project_name = "Generate Text"

    parser = argparse.ArgumentParser(
        description=f"{project_name}")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--temperature",
        help="Temperature parameter for softmax sampling.",
        type=restricted_float,
        default=1.0)
    parser.add_argument(
        "--iterations",
        help="Number of iterations needed for dynamic routing.",
        type=int,
        default=3)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to model checkpoint.",
        required=False,
        default=None,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    temperature = args["temperature"]
    model_checkpoint = args["model_checkpoint"]

    loaded_model_status, loaded_model_dict = load_model(model_checkpoint)
    if not loaded_model_status:
        raise Exception("An error occured while loading model checkpoint!")

    saved_models_list = loaded_model_dict["saved_models"]

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

    models_list = []
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

    generated_token_list = [int(start_token)]

    init_coefficient = torch.zeros((num_models,1), device=device)  # (num_models,1)

    while len(generated_token_list) < context_window:
        generated_tokens_tensor = torch.tensor(
            [generated_token_list],
            device=device)

        with torch.no_grad():
            prev_enc = None
            combined_output_list = []
            for model_index in range(num_models):
                curr_model = models_list[model_index]["model"]
                curr_model.eval()

                prev_enc, out_classifier = curr_model(
                    x=generated_tokens_tensor,
                    x_hidden=prev_enc)  # (1,Seq,Class)

                # Get last predicted token from output.
                combined_output_list.append(out_classifier[0][-1])  # (Class,)

                prev_enc = prev_enc.clone()

            combined_output_tensor = torch.stack(combined_output_list, dim=0)  # (num_models,Class)

            # Average multiple results to arrive at a concensus.
            mean_output_tensor = torch.mean(combined_output_tensor, dim=0)  # (Class,)

            # Multinomial sampling from classes.
            probs = F.softmax(mean_output_tensor / temperature, dim=0)  # (Class,)

            # Pick most likely token for next generation for each Token Sequence (Seq).
            next_token = torch.multinomial(probs, 1)  # (1,)

            # Save last token for next prediction.
            generated_token_list.append(next_token.item())

    # Remove invalid tokens if any like padding token, not in vocab list.
    cleaned_pred_tokens = [clean_token for clean_token in generated_token_list if clean_token < vocab_size]
    pred_token_list = [vocab[c] for c in cleaned_pred_tokens]
    pred_txt = "".join(pred_token_list)

    print("*" * 100, "\n\n", pred_txt, "\n\n", "*" * 100)

if __name__ == "__main__":
    main()
