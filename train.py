import torch
import os
from os.path import basename
from pathlib import Path
import datetime
from datasets import load_dataset, Audio, Dataset, concatenate_datasets
import shutil
from functools import partial
import re
from accelerate import Accelerator
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Union
from dataclasses import dataclass, field
from transformers import HfArgumentParser, AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
from model import XeusForCTC
from sconf import Config




MAX_DURATION_IN_SECONDS = 40.0
MIN_DURATION_IN_SECONDS = 1.0

def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS and input_length > MIN_DURATION_IN_SECONDS

def clean_up_data(batch, config):
    # Precompile the regex pattern if 'remove_special_characters' is enabled
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«\]\[\_]'

    try:

        sentence = batch["sentence"]

        # Remove leading and trailing whitespace
        sentence = sentence.strip()

        # Replace newlines and carriage returns with a space (or you could use '')
        sentence = sentence.replace('\n', ' ')

        if config.preprocessing.text.remove_special_characters:
            sentence = re.sub(chars_to_remove_regex, '', sentence)
        if config.preprocessing.text.lowercase:
            sentence = sentence.lower()
        if config.preprocessing.text.remove_latin_characters:
            sentence = re.sub(r'[a-z]+', '', sentence)

        # Update the batch with the cleaned sentence
        batch["sentence"] = sentence

    except Exception as e:
        print(f"An error occurred in preprocessing: {e}")

    return batch


def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def save_vocab_file(source, dest):
    if not os.path.exists(source):
        raise FileNotFoundError(f"the source file does not exist at path {source}")

    shutil.copy2(source, dest)
    print(f"file copied at path {dest}")


def normalize_dataset(dataset, config):

    cleanup_fn = partial(clean_up_data, config=config)
    # Apply the preprocessing function to the dataset
    return dataset.map(cleanup_fn, num_proc=config.num_workers)  # Ensure batched=True if the function expects batched inputs


def create_custom_dataset(input_directory, split):
    audio_dict = []
    sentence_dict = []

    # Walk through the input directory
    for root, dirs, files in os.walk(f"{input_directory}/{split}"):
        for file in files:
            # Check if the current file is a WAV file
            if file.endswith('.wav'):
                # Get the base name of the file (without extension)
                base_name = os.path.splitext(file)[0]
                # Construct the path for the corresponding text file
                text_file_path = os.path.join(root, f"{base_name}.txt")

                # Check if the text file exists
                if os.path.exists(text_file_path):
                    # Read and print the content of the text file
                    with open(text_file_path, 'r', encoding='utf-8') as text_file:
                        content = text_file.read()
                        audio_dict.append(os.path.join(root, f"{base_name}.wav"))
                        sentence_dict.append(content)
                else:
                    print(f"Text file for {file} not found.")


    return Dataset.from_dict({
        "audio": audio_dict,
        "sentence": sentence_dict
    }).cast_column("audio", Audio())


def create_vocabulary(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}





def prepare_dataset(dynamic_datasets, num_workers, config):
    combined_dataset = []

    for dataset_config in dynamic_datasets:
        try:
            dataset = dataset_config.dataset
            split = dataset_config.split
            audio_field = dataset_config.input_fields[0]
            text_field = dataset_config.input_fields[1]
            sampling_rate = config.sampling_rate

            # Load the dataset
            if "name" in dataset_config:
                dataset = load_dataset(dataset, dataset_config.name, split=split)
            else:
                dataset = load_dataset(dataset, split=split)
                # dataset = load_dataset(dataset_name, split=f"{split}[:1000]")


            # Ensure the audio field exists before casting
            if audio_field in dataset.column_names:
                dataset = dataset.cast_column(audio_field, Audio(sampling_rate))
            else:
                raise ValueError(f"The audio field {audio_field} does not exist in the dataset {dataset}.")

            # Rename the text field if necessary
            if text_field in dataset.column_names and text_field != "sentence":
                dataset = dataset.rename_column(text_field, "sentence")
            elif text_field not in dataset.column_names:
                raise ValueError(f"The text field {text_field} does not exist in the dataset {dataset}.")

            # Remove unwanted columns
            required_columns = ["audio", "sentence"]
            columns_to_remove = set(dataset.column_names) - set(required_columns)
            dataset = dataset.remove_columns(columns_to_remove)

            combined_dataset.append(dataset)
        except Exception as e:
            # Instead of printing the error, you can raise it to exit the function
            raise Exception(f"An error occurred while preparing the dataset {dataset}: {e}")

    # Concatenate and shuffle datasets
    ds_to_return = concatenate_datasets(combined_dataset)
    ds_to_return = ds_to_return.shuffle(seed=22)

    ds_to_return = normalize_dataset(ds_to_return, config)
    vocab = ds_to_return.map(create_vocabulary, batched=True, batch_size=-1, keep_in_memory=False,
                                         remove_columns=ds_to_return.column_names, num_proc=num_workers)

    return vocab, ds_to_return


def save_vocab_json(vocab_dict, path):
    save_path = Path(path) / "vocab.json"
    with open(save_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)


def text_to_char_sequence(vocab, text_array):

    sequences = []
    for text in text_array:
        # Trim leading and trailing spaces, and replace spaces between sentences with |
        text = text.strip().replace(" ", "|")
        sequence = [vocab.get(char, vocab.get("[UNK]")) for char in text]
        sequences.append(sequence)
    return sequences







def create_collate_fn(vocab_dict):
    def collate_fn(batch):
        audio = [item["audio"] for item in batch]

        sentence = [item["sentence"] for item in batch]
        labels = text_to_char_sequence(vocab_dict, sentence)

        wavs = [torch.tensor(item["array"], dtype=torch.float32) for item in audio]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs])
        labels_length = torch.LongTensor([len(label) for label in labels])

        labels = [torch.LongTensor(label) for label in labels]

        wavs = pad_sequence(wavs, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)

        return wavs, labels, wav_lengths

    return collate_fn


# Function to load checkpoint
def load_checkpoint(ckpt_path):
    accelerator.load_state(ckpt_path)
    step = str(ckpt_path).split("_")[-1]
    return step



# Function to save checkpoint
def save_checkpoint(step, ckpt_dir):
    accelerator.save_state(f"{ckpt_dir}/step_{step}")
    print(f"Checkpoint saved at step {step} in {ckpt_dir}")

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    config: Optional[str] = field(metadata={"help": "training config file"})
    exp_version: Optional[str] = field(default="", metadata={"help": "experiment version"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = Config(script_args.config)

if not config.get("exp_name", False):
    config.exp_name = basename(script_args.config).split(".")[0]

config.exp_version = (
 datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not script_args.exp_version
    else script_args.exp_version
)

vocab_train, train_set = prepare_dataset(config.train_datasets, config.num_workers, config)
vocab_test, val_set = prepare_dataset(config.eval_datasets, config.num_workers, config)
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

save_path = Path(config.result_path) / config.exp_name / config.exp_version

save_config_file(config, save_path)

# save_vocab_file(f"{script_args.preprocessed_dataset}/vocab.json", save_path / "vocab.json")

save_vocab_json(vocab_dict, save_path)


config.hidden_size = 1024
config.vocab_size = len(vocab_dict)
config.pad_token_id = vocab_dict["[PAD]"]
config.final_dropout = 0.1
config.ckpt_dir = save_path

train_dataloader = DataLoader(
    train_set,
    collate_fn=create_collate_fn(vocab_dict),
    batch_size=config.train_batch_size,
    shuffle=True,
    drop_last=True
)

eval_dataloader = DataLoader(
    val_set,
    collate_fn=create_collate_fn(vocab_dict),
    batch_size=config.eval_batch_size,
    shuffle=False,
    drop_last=True
)

model = XeusForCTC(config, train=True)

optimizer = AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_steps)
from accelerate import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=config.gradient_accumulation_steps)


model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, scheduler
)

# Load from checkpoint if available
start_step = 0
if config.ckpt_path !="" and os.path.exists(config.ckpt_path):
    start_step = load_checkpoint(config.ckpt_path)
    print(f"Resumed from checkpoint at step {start_step}")


progress_bar = tqdm(total=config.total_steps, initial=start_step, desc="Training")

model.train()
global_step = 0
while True:
    for step, batch in enumerate(train_dataloader):
        if global_step < start_step:
            continue

        with accelerator.accumulate(model):
            wavs, labels, wav_lengths = batch
            outputs = model(wavs, labels, wav_lengths)
            loss, logits, _ = outputs
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1

        if global_step % config.logging_steps == 0 and accelerator.is_main_process and global_step != 0:
            print(f"Step {global_step}, Train Loss: {loss}")

        if global_step % config.save_steps == 0 and global_step != 0:
            save_checkpoint(global_step, config.ckpt_dir)

        if global_step >= config.total_steps:
            break

        if global_step % config.eval_step == 0 and global_step != 0:
            model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                if step > config.validation_steps:
                    break
                wavs, labels, wav_lengths = batch
                with torch.no_grad():
                    outputs = model(wavs, labels, wav_lengths)
                loss, _, _ = outputs
                # losses.append(accelerator.gather(loss))
                losses.append(loss.item())

            # eval_loss = torch.mean(torch.cat(losses))
            eval_loss = torch.tensor(losses).mean()
            if accelerator.is_main_process:
                print(f"Step {step + 1}, Eval Loss: {eval_loss}")
            model.train()

    if global_step >= config.total_steps:
        break

