import os
import json
import argparse
from pathlib import Path
import soundfile as sf
from scipy.signal import resample
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from safetensors import safe_open
from model import XeusForCTC


def ctc_greedy_decoder(logits, vocab):
    pad_id = vocab["[PAD]"]

    # Apply softmax to logits to get probabilities
    probs = F.softmax(logits, dim=-1)

    # Inverse the vocab dictionary to map indices back to characters
    index_to_char = {index: char for char, index in vocab.items()}

    # Get the most probable token indices
    pred_indices = torch.argmax(probs, dim=-1)

    decoded_sequences = []
    for indices in pred_indices:
        decoded = []
        prev_index = None
        for index in indices:
            index = index.item()
            if index != prev_index:
                if index != pad_id:
                    decoded.append(index_to_char.get(index, "[UNK]"))
                prev_index = index
        decoded_sequences.append("".join(decoded).replace("|", " "))

    return decoded_sequences


def load_model(config, ckpt_path):
    # Load model
    model = XeusForCTC(config)

    # Load checkpoint
    with safe_open(f"{ckpt_path}/model.safetensors", framework="pt") as f:
        state_dict = {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.eval()

    return model


def perform_inference(model, wavs, vocab):
    # Tokenize input text
    wav_lengths = torch.LongTensor([len(wav) for wav in [wavs]])
    wavs = pad_sequence(torch.Tensor([wavs]), batch_first=True)

    with torch.inference_mode():
        _, logits, _ = model(wavs, None, wav_lengths)

    # Get prediction
    prediction = ctc_greedy_decoder(logits, vocab)

    return prediction


def load_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def read_and_resample_wav(wav_path, target_sr=16_000):
    # Load the audio file
    y, sr = sf.read(wav_path)

    # Resample the audio to the target sampling rate
    if sr != target_sr:
        num_samples = int(len(y) * target_sr / sr)
        y = resample(y, num_samples)
        sr = target_sr

    return y, sr


# Example usage:
def main(args):
    # Check if checkpoint exists
    ckpt_path = Path(args.ckpt_path)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint path '{ckpt_path}' does not exist.")
        return

    # Get the parent directory of the checkpoint path
    parent_dir = ckpt_path.parent

    # Load vocab.json from the parent directory
    vocab_path = parent_dir / "vocab.json"
    if not vocab_path.is_file():
        print(f"vocab.json not found in '{parent_dir}'.")
        return

    vocab_dict = load_vocab(vocab_path)

    dummy_config = {
        "vocab_size": len(vocab_dict),
        "pad_token_id": vocab_dict["[PAD]"],
        "pretrained_model_path": "./XEUS/model/xeus_checkpoint.pth",
        "final_dropout": 0.1,
        "hidden_size": 1024,
    }

    config = Config(dummy_config)

    # Load the model
    model = load_model(config, args.ckpt_path)
    audio, _ = read_and_resample_wav(args.audio, target_sr=16_000)

    prediction = perform_inference(model, audio, vocab_dict)

    print(prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for XUES model")

    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to the audio fle"
    )

    main(parser.parse_args())
