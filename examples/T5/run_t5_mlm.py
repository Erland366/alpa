# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
import functools
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import wandb
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import HfApi
from loguru import logger
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    BatchEncoding,
    FlaxT5ForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    T5Config,
    is_tensorboard_available,
    set_seed,
)
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right

import alpa
from alpa.model.model_util import DynamicScale, TrainState

load_dotenv()

alpa.init(cluster="ray")
alpa.global_config.force_normalize_embedding_shapes = (
    False  # Tell Alpa not to modify embedding shapes
)
tf.config.experimental.set_visible_devices([], "GPU")

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def instantiate_wandb(run_name: Optional[str] = None, config: Union[Dict, None] = None):
    wandb.init(project="alpafelafax", entity=None, name=run_name, config=config)


def maybe_stop_wandb():
    try:
        if wandb.run:
            wandb.finish()
    except Exception as _:
        pass


def login_wandb(
    environ_name: str = "WANDB_API_KEY", token: Optional[str] = None, **kwargs
):
    if token is None:
        token = os.getenv(environ_name)
        logger.debug(f"Use token from environment variable {environ_name}")
    wandb.login(key=token, **kwargs)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="alpafelafax",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    adafactor: bool = field(
        default=False,
        metadata={"help": "Whether or not to replace AdamW by Adafactor."},
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )
    logging_steps: int = field(
        default=1, metadata={"help": "Log every X updates steps."}
    )
    save_steps: int = field(
        default=99999, metadata={"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(
        default=100, metadata={"help": "Run an evaluation every X steps."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upload the trained model to the model hub after training."
        },
    )
    hub_model_id: str = field(
        default=None,
        metadata={
            "help": "The name of the repository to keep in sync with the local `output_dir`."
        },
    )
    hub_token: str = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    num_micro_batches: int = field(
        default=2, metadata={"help": "Number of micro batches for training."}
    )

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="google/t5-v1_1-base",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="oscar",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default="unshuffled_deduplicated_no",
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input train ref data file for whole word masking in Chinese."
        },
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input validation ref data file for whole word masking in Chinese."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={
            "help": "Ratio of tokens to mask for span masked language modeling loss"
        },
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )
    use_data_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use a sample of the dataset -> Erland/oscar_sampled_1000."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


def compute_input_and_target_lengths(
    inputs_length, noise_density, mean_noise_span_length
):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while (
        _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
        <= inputs_length
    ):
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
        tokens_length
    )

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


@flax.struct.dataclass
class FlaxDataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(
            np.round(
                min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length
            )
        )

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


def generate_batch_splits(
    samples_idx: np.ndarray, batch_size: int, drop_last=True
) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned."""
    num_samples = len(samples_idx)
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx


def write_train_metric_wandb(train_metrics, train_time, step):
    train_metrics = alpa.util.get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        if hasattr(vals, "__iter__"):
            for i, val in enumerate(vals):
                wandb.log({tag: val}, step - len(vals) + i + 1)
        else:
            wandb.log({tag: val}, step)
    wandb.log({"train_time": train_time}, step)


def write_eval_metric_wandb(eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        wandb.log({f"eval_{metric_name}": value}, step)


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = alpa.util.get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    parser_dict = vars(parser)

    login_wandb()
    instantiate_wandb(config=parser_dict)
    logger.info("Logged in to wandb")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=f"train[:{data_args.validation_split_percentage}%]",
    )
    # THIS IS ONLY FOR NOW
    dataset = dataset.train_test_split(test_size=0.1)
    dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        token=model_args.token,
    )

    config = T5Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )

    if training_args.do_train:
        column_names = dataset["train"].column_names
    else:
        column_names = dataset["validation"].column_names

    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_attention_mask=False)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
    )

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # Drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # Customize this part to your needs.
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= expanded_inputs_length:
            total_length = (
                total_length // expanded_inputs_length
            ) * expanded_inputs_length
        result = {
            k: [
                t[i : i + expanded_inputs_length]
                for i in range(0, total_length, expanded_inputs_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    rng = jax.random.PRNGKey(training_args.seed)
    # dropout_rngs = jax.random.split(rng, jax.local_device_count())
    rng, dropout_rng = jax.random.split(rng)

    # Setup train state
    if model_args.dtype == "float16":
        use_master_copy = True
        dynamic_scale = DynamicScale()
        # Fix a bug in huggingface's implementation (https://github.com/huggingface/transformers/pull/18462)
        alpa.global_config.flax_always_use_fp16_embedding = True
    else:
        use_master_copy = dynamic_scale = None

    model = FlaxT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        seed=training_args.seed,
        dtype=getattr(jnp, model_args.dtype),
        token=model_args.token,
    )

    data_collator = FlaxDataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
        input_length=max_seq_length,
        target_length=targets_length,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )

    num_epochs = int(training_args.num_train_epochs)
    # train_batch_size = (
    #     int(training_args.per_device_train_batch_size) * jax.device_count()
    # )
    train_batch_size = (
        int(training_args.per_device_train_batch_size) * alpa.get_global_num_devices()
    )
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    # eval_batch_size = per_device_eval_batch_size * jax.device_count()
    eval_batch_size = per_device_eval_batch_size * alpa.get_global_num_devices()

    num_train_steps = len(tokenized_datasets["train"]) // train_batch_size * num_epochs

    num_of_hosts = jax.process_count()
    current_host_idx = jax.process_index()

    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=training_args.learning_rate,
        transition_steps=training_args.warmup_steps,
    )
    decay_fn = optax.linear_schedule(
        init_value=training_args.learning_rate,
        end_value=0,
        transition_steps=num_train_steps - training_args.warmup_steps,
    )
    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
    )

    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params)
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    optimizer = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    # state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer)
    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dynamic_scale=dynamic_scale,
        use_master_copy=use_master_copy,
    )

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        dropout_rng = jax.random.fold_in(dropout_rng, state.step)

        def loss_fn(params):
            labels = batch.pop("labels")

            logits = state.apply_fn(
                **batch, params=params, dropout_rng=dropout_rng, train=True
            )[0]

            # compute loss
            loss = optax.softmax_cross_entropy(
                logits, onehot(labels, logits.shape[-1])
            ).mean()

            return loss

        # grad_fn = jax.value_and_grad(loss_fn)
        # loss, grad = grad_fn(state.params)

        dynamic_scale = state.dynamic_scale
        if dynamic_scale:
            grad_fn = dynamic_scale.value_and_grad(loss_fn)
            dynamic_scale, is_fin, loss, grad = grad_fn(state.params)
        else:
            grad_fn = alpa.value_and_grad(loss_fn)
            loss, grad = grad_fn(state.params)

        # grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        if dynamic_scale:
            new_state = new_state.replace(
                opt_state=jax.tree_map(
                    functools.partial(jnp.where, is_fin),
                    new_state.opt_state,
                    state.opt_state,
                ),
                params=jax.tree_map(
                    functools.partial(jnp.where, is_fin), new_state.params, state.params
                ),
                master_copy=jax.tree_map(
                    functools.partial(jnp.where, is_fin),
                    new_state.master_copy,
                    state.master_copy,
                ),
                dynamic_scale=dynamic_scale,
            )

        # metrics = jax.lax.pmean(
        #     {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)},
        #     axis_name="batch",
        # )
        metrics = {
            "loss": loss,
            "learning_rate": linear_decay_lr_schedule_fn(state.step),
        }

        return new_state, metrics, dropout_rng

    # Create parallel version of the train step
    method = alpa.Zero2Parallel(num_micro_batches=training_args.num_micro_batches)
    p_train_step = alpa.parallelize(train_step, method=method, donate_argnums=(0,))
    # p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    min_batch_size = alpa.get_global_num_devices() * training_args.num_micro_batches

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")

        logits = model(**batch, params=params, train=False)[0]

        # compute loss
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels)

        # summarize metrics
        metrics = {"loss": loss.mean(), "accuracy": accuracy.mean()}
        # metrics = jax.lax.pmean(metrics, axis_name="batch")

        return metrics

    p_eval_step = alpa.parallelize(eval_step)
    # p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    # state = jax_utils.replicate(state)

    train_time = 0
    epochs = tqdm(range(num_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        num_train_samples = len(tokenized_datasets["train"])
        # Avoid using jax.numpy here in case of TPU training
        train_samples_idx = np.random.permutation(np.arange(num_train_samples))
        train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size)

        # Gather the indexes for creating the batch and do a training step
        for step, batch_idx in enumerate(
            tqdm(train_batch_idx, desc="Training...", position=1)
        ):
            samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples)

            local_host_model_inputs = {
                key: np.split(model_inputs.data[key], num_of_hosts, axis=0)[
                    current_host_idx
                ]
                for key, value in model_inputs.data.items()
            }

            # Model forward
            model_inputs = local_host_model_inputs
            state, train_metric, dropout_rng = p_train_step(
                state, local_host_model_inputs, dropout_rng
            )
            train_metrics.append(train_metric)

            cur_step = epoch * (num_train_samples // train_batch_size) + step

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                # train_metric = jax_utils.unreplicate(train_metric)
                train_metrics = alpa.util.get_metrics(train_metrics)
                train_metrics = jax.tree_util.tree_map(jnp.mean, train_metrics)
                train_time += time.time() - train_start
                # if has_tensorboard and jax.process_index() == 0:
                #     write_train_metric(summary_writer, train_metrics, train_time, cur_step)
                # if jax.process_index() == 0:
                #     write_train_metric_wandb(train_metrics, train_time, cur_step)

                epochs.write(
                    f"Step... ({cur_step} | Loss: {train_metric['loss']}, Learning Rate:"
                    f" {train_metric['learning_rate']})"
                )

                train_metrics = []

            if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                # ======================== Evaluating ==============================
                num_eval_samples = len(tokenized_datasets["validation"])
                # Avoid using jax.numpy here in case of TPU training
                eval_samples_idx = np.arange(num_eval_samples)
                eval_batch_idx = generate_batch_splits(
                    eval_samples_idx, eval_batch_size, drop_last=False
                )

                eval_metrics = []
                for i, batch_idx in enumerate(
                    tqdm(eval_batch_idx, desc="Evaluating ...", position=2)
                ):
                    samples = [
                        tokenized_datasets["validation"][int(idx)] for idx in batch_idx
                    ]
                    model_inputs = data_collator(samples)

                    # metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                    #     state.params,
                    #     model_inputs.data,
                    #     min_device_batch=per_device_eval_batch_size,
                    # )

                    # Model forward
                    metrics = p_eval_step(
                        state.params,
                        model_inputs.data,
                        # min_device_batch=per_device_eval_batch_size,
                    )
                    eval_metrics.append(metrics)

                # get eval metrics
                eval_metrics = alpa.util.get_metrics(eval_metrics)
                eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)

                # normalize eval_metrics
                try:
                    eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
                except OverflowError:
                    eval_metrics["perplexity"] = float("inf")

                # Update progress bar
                epochs.write(
                    f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})"
                )

                # Save metrics
                # if has_tensorboard and jax.process_index() == 0:
                #     write_eval_metric(summary_writer, eval_metrics, cur_step)
                # if jax.process_index() == 0:
                #    write_eval_metric_wandb(eval_metrics, cur_step)

            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    alpa.prefetch(state.params)
                    params = alpa.util.map_to_nparray(state.params)
                    model.save_pretrained(training_args.output_dir, params=params)
                    tokenizer.save_pretrained(training_args.output_dir)
                    # if training_args.push_to_hub:
                    #     api.upload_folder(
                    #         commit_message=f"Saving weights and logs of step {cur_step}",
                    #         folder_path=training_args.output_dir,
                    #         repo_id=repo_id,
                    #         repo_type="model",
                    #         token=training_args.hub_token,
                    #     )

    # Eval after training
    if training_args.do_eval:
        num_eval_samples = len(tokenized_datasets["validation"])
        # Avoid using jax.numpy here in case of TPU training
        eval_samples_idx = np.arange(num_eval_samples)
        eval_batch_idx = generate_batch_splits(
            eval_samples_idx, eval_batch_size, drop_last=False
        )

        eval_metrics = []
        for i, batch_idx in enumerate(
            tqdm(eval_batch_idx, desc="Evaluating ...", position=2)
        ):
            samples = [tokenized_datasets["validation"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples)

            # Model forward
            # metrics = pad_shard_unpad(p_eval_step, static_return=True)(
            #     state.params,
            #     model_inputs.data,
            #     min_device_batch=per_device_eval_batch_size,
            # )
            metrics = p_eval_step(
                state.params,
                model_inputs.data,
            )
            eval_metrics.append(metrics)

        # get eval metrics
        eval_metrics = alpa.util.get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(
            lambda metric: jnp.mean(metric).item(), eval_metrics
        )

        if jax.process_index() == 0:
            eval_metrics = {
                f"eval_{metric_name}": value
                for metric_name, value in eval_metrics.items()
            }
            path = os.path.join(training_args.output_dir, "eval_results.json")
            with open(path, "w") as f:
                json.dump(eval_metrics, f, indent=4, sort_keys=True)

    maybe_stop_wandb()


if __name__ == "__main__":
    main()
