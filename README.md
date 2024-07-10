# XEUS Fine-tuning for ASR [Speech Recognition] 

This repository contains code to fine-tune the espnet XEUS model for Automatic Speech Recognition (ASR).

# Work In Progress ![Work In Progress](https://img.shields.io/badge/status-WIP-yellow)

**Note:** This repository is still in development. Please create an issue if you encounter any bugs.

# Colab Support
 Colab support is coming soon! Stay tuned for updates.

## Prerequisites
Before running the code, ensure you have the following prerequisites installed:

- CUDA 11.8
- torch >= 2.0
- torchaudio

## Installation

1. Clone the repository:

```bash
git clone git@github.com:pashanitw/xeus-finetune.git
cd xeus-finetune
```

2. Install the required Python packages using pip:

```bash
pip install 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'
git lfs install
git clone https://huggingface.co/espnet/XEUS


pip install -r requirements.txt
```

## Training Process Overview


### Step 1: Authenticate with Hugging Face

you can log in to your Hugging Face account by opening a terminal and running the following command:
```bash
huggingface-cli login
````
### Step 2: Configure Your Training


- **Configure Your Training**: Prepare your training configuration by creating or updating a YAML file (`config.yaml`). This file should include your datasets' paths, training parameters, and any model-specific settings. Ensure the configuration aligns with your project needs and the datasets you plan to use.


### Step 3: Initiate Training
To train the model, execute the following command:

```bash
accelerate launch train.py --config configs/hi_hf.yaml
```

### Step 4: Inference
for the inference

```bash
python inference.py --ckpt_path <checkpoint path> --audio audio.wav
```
example
```bash
python inference.py --ckpt_path ./step_2000 --audio audio.wav
```

### Step 5: Calculating the WER
run the following command to calculate word-error-rate metric
```bash
python wer.py --ckpt_path <checkpoint path> --dataset <dataset> --name <subset> --split <split>
```
example
```bash
 python wer.py --ckpt_path ./step_2000 --dataset google/fleurs --name hi_in --split test
```


## Configuration Parameters

The configuration of the model training and evaluation is defined by the following parameters:

### `train_datasets` and `eval_datasets`
Specifies the datasets used for training and evaluating the model, respectively. Each entry in these lists consists of the following fields:

- `dataset`: The path to the dataset.
- `split`: The dataset split to use, e.g., `train`, `test`.
- `input_fields`: The fields to be used as input from the dataset. For training datasets, these are typically `"audio"` and `"sentence"`.

#### Example of `train_datasets` combining multiple datasets:

```yaml
train_datasets:
  - dataset: "mozilla-foundation/common_voice_16_0"
    name: "ka"
    split: "train+validation"
    input_fields:
      - "audio"
      - "sentence"
  - dataset: "mozilla-foundation/common_voice_16_0"
    name: "ka"
    split: "test"
    input_fields:
      - "audio"
      - "sentence"

```

#### Example of `eval_datasets`:
```yaml
eval_datasets:
  - dataset: "google/fleurs"
    name: "ta_in"
    split: "test"
    input_fields: ["audio", "transcription"]
```


Each dataset directory contains the respective splits as indicated.

### preprocessing
Defines the text preprocessing parameters:
- `remove_special_characters`: Whether to remove special characters from the text.
- `lowercase`: Whether to convert all characters to lowercase.
- `remove_punctuation`: Whether to remove punctuation marks from the text.
- `remove_latin_characters`: Whether to remove Latin characters from the text.

### Other Parameters
- `pretrained_model_path`: The path or identifier of the pretrained model to use, e.g., `"./XEUS/model/xeus_checkpoint.pth"`.
- `train_batch_size`: The batch size to use during training.
- `eval_batch_size`: The batch size to use during evaluation.
- `num_workers`: The number of worker threads for loading data.
- `result_path`: The path where results should be saved.
- `exp_name`: The name of the experiment.
- `sampling_rate`: The sampling rate for audio data.
- `warmup_steps`: The number of warmup steps for learning rate scheduling.
- `learning_rate`: The learning rate for training.
- `save_steps`, `eval_step`, `logging_steps`: The frequency of saving checkpoints, evaluating the model, and logging training information, respectively.
- `save_total_limit`: The maximum number of checkpoints to save.
- `gradient_checkpointing`: Enable gradient checkpointing to reduce memory usage.
- `train_epochs`: The number of training epochs.
- `gradient_accumulation_steps`: The number of steps over which gradients are accumulated.
- `resume_from_checkpoint`: Whether to resume training from a checkpoint.
- `ckpt_path`: The directory path for loading checkpoints.

This configuration allows for flexible and detailed setup of model training and evaluation, tailored to specific needs and datasets.