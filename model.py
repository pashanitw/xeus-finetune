import os
from typing import Union, Optional
from argparse import Namespace
import torch
import torch.nn as nn
from espnet2.tasks.ssl import SSLTask
from sconf import Config


class XeusForCTC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ssl_dir = os.path.dirname(config.pretrained_model_path)
        ssl_config = f"{ssl_dir}/config.yaml"
        if not os.path.exists(ssl_config):
            raise FileNotFoundError("XEUS model config file not found")

        config_dict = Config(ssl_config)
        xeus_train_args = Namespace(**config_dict)

        self.xeus_model = SSLTask.build_model(xeus_train_args)

        if config.use_flash_attn:
            for layer in self.xeus_model.decoder.decoders:
                layer.use_flash_attn = True

        if os.path.exists(config.pretrained_model_path):
            self.xeus_model.load_state_dict(
                torch.load(config.pretrained_model_path), strict=True
            )
            print("pretrained XEUS model loaded")

        self.kernel_sizes = [10, 3, 3, 3, 3, 2, 2]
        self.strides = [5, 2, 2, 2, 2, 2, 2]

        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2BertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int]
    ):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (
                torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
            )

        for kernel_size, stride in zip(self.kernel_sizes, self.strides):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def forward(
        self,
        wavs: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        wav_lengths: Optional[torch.Tensor] = None,
    ):

        hidden_states = self.xeus_model.encode(
            wavs, wav_lengths, use_mask=False, use_final_output=False
        )[0][-1]

        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(
                    f"Label values must be <= vocab_size: {self.config.vocab_size}"
                )

            # assuming that padded tokens are filled with -1
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            input_lengths = self._get_feat_extract_output_lengths(wav_lengths).to(
                torch.long
            )

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction="mean",
                    zero_infinity=True,
                )

        return loss, logits, hidden_states
