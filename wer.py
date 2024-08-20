import os
import argparse
import shutil
from pathlib import Path
import torch
import evaluate
from datasets import load_dataset, Audio
from tqdm import tqdm
from inference import load_model, load_vocab, perform_inference, Config

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False

    return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            "Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


def get_text_column_names(column_names):
    if "text" in column_names:
        return "text"
    elif "sentence" in column_names:
        return "sentence"
    elif "normalized_text" in column_names:
        return "normalized_text"
    elif "transcript" in column_names:
        return "transcript"
    elif "transcription" in column_names:
        return "transcription"


def data(dataset):
    for item in dataset:
        yield {**item["audio"], "reference": get_text(item)}


def main(args):
    ckpt_path_parent = str(Path(args.ckpt_path).parent)
    if not os.path.exists(f"{args.ckpt_path}/vocab.json"):
        shutil.copy2(f"{ckpt_path_parent}/vocab.json", f"{args.ckpt_path}/vocab.json")
    else:
        print(f"Loading vocab.json from {args.ckpt_path}")

    vocab_dict = load_vocab(f"{args.ckpt_path}/vocab.json")

    dummy_config = {
        "vocab_size": len(vocab_dict),
        "pad_token_id": vocab_dict["[PAD]"],
        "pretrained_model_path": "./XEUS/model/xeus_checkpoint.pth",
        "final_dropout": 0.1,
        "hidden_size": 1024,
    }
    config = Config(dummy_config)
    model = load_model(config, args.ckpt_path)
    dataset = load_dataset(
        args.dataset,
        args.name,
        split=args.split,
        use_auth_token=True,
    )
    text_column_name = get_text_column_names(dataset.column_names)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.filter(
        is_target_text_in_range, input_columns=[text_column_name], num_proc=2
    )
    predictions = []
    references = []
    with torch.inference_mode():
        for item in tqdm(data(dataset), total=len(dataset), desc="Decode Progress"):
            prediction = perform_inference(model, item["array"], vocab_dict)

            predictions.append(prediction[0])
            references.append(item["reference"])

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)
    cer = cer_metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)

    print("---")
    print("WER : ", wer)
    print("CER : ", cer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Folder with the pytorch_model.bin file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset from huggingface to evaluate the model on. Example: mozilla-foundation/common_voice_11_0",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Config of the dataset. Eg. 'hi' for the Hindi split of Common Voice",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="test",
        help="Split of the dataset. Eg. 'test'",
    )

    main(parser.parse_args())
