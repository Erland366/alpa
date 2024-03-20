#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import json
import logging
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import datasets
import evaluate
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import struct, traverse_util
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import Repository, create_repo
from tqdm import tqdm
from utils_qa import postprocess_qa_predictions

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    FlaxAutoModelForQuestionAnswering,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    is_tensorboard_available,
)
from transformers.utils import check_min_version, send_example_telemetry
import alpa
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef
from torch.utils.data import Dataset, DataLoader
import torch
from alpa.adaptdl.pollux_agent import pollux_agent
from alpa.adaptdl.api import reallocate_and_update_state

from jax.tree_util import tree_flatten, tree_unflatten
from alpa.adaptdl.gns_util import (extract_values_with_key_p, 
                                   normsqr_groups,
                                   average_groups,
                                   update_avg,
                                   compute_gradient_noise_scale, 
                                   compute_gradsnorms, 
                                   update_variance, 
                                   compute_variance, 
                                   init_dict, 
                                   compute_grad_flat_mean, 
                                   flatten_gradients, 
                                   running_gradient,
                                   run_grads, 
                                   init_running_gradients)
from alpa.adaptdl.dataloader import current_dataloader
from alpa.adaptdl.metrics import update_grad_params, update_progress
from jax._src.config import flags
from alpa.adaptdl.api import update_state_on_bs_change, create_scaled_lr_fn, reallocate_and_update_state
import alpa.adaptdl.dataloader
import alpa.adaptdl.epoch
from alpa.adaptdl.scaling_rules import ScalingRuleBase, LinearScale, SqrtScale
import datetime
import wandb

class QADataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        item = {key: torch.tensor(value) for key, value in item.items()}
        return item

# alpa.init(cluster="ray")
# alpa.init(cluster="ray", num_nodes=1, num_devices_per_node=2, namespace="alpa_default_space_non-adaptive_bert")
alpa.init(cluster="ray", scheduler_address="http://127.0.0.1:8000")
# alpa.init(cluster="ray", num_nodes=1, num_devices_per_node=1)


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.38.0.dev0")

Array = Any
DatasetHF = datasets.arrow_dataset.Dataset
PRNGKey = Any


# region Arguments
@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
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
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    pretrain: bool = field(
        default=True, metadata={"help": "Whether or not to pretrain."}
    )
    count: int = field(default=2, metadata={"help": "The number of stored grads."})
    scale: int = field(default=1, metadata={"help": "Scale"})
    smoothing: float = field(default=0.9, metadata={"help": "Smoothing parameter for PGNS"})

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
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
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
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
                " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


# endregion

@dataclass
class RunningGradientsState:
    store_grads: list
    noise: float
    scale: float
    noise_scale: float
    beta: float
    running_noise: float
    running_scale: float
    n_batch: int


# region Create a train state
def create_train_state(
    model: FlaxAutoModelForQuestionAnswering,
    learning_rate_fn: Callable[[int], float],
    num_labels: int,
    training_args: TrainingArguments,
) -> train_state.TrainState:
    """Create initial training state."""

    class TrainState(train_state.TrainState):
        """Train state with an Optax optimizer.

        The two functions below differ depending on whether the task is classification
        or regression.

        Args:
          logits_fn: Applied to last layer to obtain the logits.
          loss_fn: Function to compute the loss.
        """

        logits_fn: Callable = struct.field(pytree_node=False)
        loss_fn: Callable = struct.field(pytree_node=False)

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
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
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    def cross_entropy_loss(logits, labels):
        start_loss = optax.softmax_cross_entropy(logits[0], onehot(labels[0], num_classes=num_labels))
        end_loss = optax.softmax_cross_entropy(logits[1], onehot(labels[1], num_classes=num_labels))
        xentropy = (start_loss + end_loss) / 2.0
        return jnp.mean(xentropy)

    return TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
        logits_fn=lambda logits: logits,
        loss_fn=cross_entropy_loss,
    )


# endregion


# region Create learning rate function
def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.ndarray]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


# endregion


# region train data iterator
def train_data_collator(rng: PRNGKey, dataset: DatasetHF, batch_size: int):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: np.array(v) for k, v in batch.items()}
        # batch = shard(batch)

        yield batch


# endregion


# region eval data iterator
def eval_data_collator(dataset: DatasetHF, batch_size: int):
    """Returns batches of size `batch_size` from `eval dataset`. Sharding handled by `pad_shard_unpad` in the eval loop."""
    batch_idx = np.arange(len(dataset))

    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch


# endregion


def main():
    # region Argument parsing
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    try:
        from alpa.adaptdl.gradient_noise_scale import gns
    except Exception as e:
        print(f'could not import gns: {e}')

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_qa", model_args, data_args, framework="flax")
    # endregion

    # region Logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # endregion

    # Handle the repository creation
    if training_args.push_to_hub:
        # Retrieve of infer repo_name
        repo_name = training_args.hub_model_id
        if repo_name is None:
            repo_name = Path(training_args.output_dir).absolute().name
        # Create repo and retrieve repo_id
        repo_id = create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id
        # Clone repo locally
        repo = Repository(training_args.output_dir, clone_from=repo_id, token=training_args.hub_token)

    # region Load Data
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            field="data",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.
    # endregion

    # region Load pretrained model and tokenizer
    #
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    # endregion

    # region Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )
    # endregion

    # region Preprocessing the datasets
    # Preprocessing is slightly different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    processed_raw_datasets = {}
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        processed_raw_datasets["train"] = train_dataset
        
        train_dataset_pytorch = QADataset(train_dataset)

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            # tokenized_examples["offset_mapping"][i] = [
            #     (o if sequence_ids[k] == context_index else None)
            #     for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            # ]

        return tokenized_examples

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        processed_raw_datasets["validation"] = eval_dataset

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        predict_dataset = predict_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        processed_raw_datasets["test"] = predict_dataset
    # endregion

    # region Metrics and Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load(
        "squad_v2" if data_args.version_2_with_negative else "squad", cache_dir=model_args.cache_dir
    )

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs of the model.
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    # endregion

    # region Training steps and logging init
    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Define a summary writer
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(training_args.output_dir)
            summary_writer.hparams({**training_args.to_dict(), **vars(model_args), **vars(data_args)})
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    def write_train_metric(summary_writer, train_metrics, train_time, step):
        summary_writer.scalar("train_time", train_time, step)

        train_metrics = get_metrics(train_metrics)
        for key, vals in train_metrics.items():
            tag = f"train_{key}"
            for i, val in enumerate(vals):
                summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    def write_eval_metric(summary_writer, eval_metrics, step):
        for metric_name, value in eval_metrics.items():
            summary_writer.scalar(f"eval_{metric_name}", value, step)

    num_epochs = int(training_args.num_train_epochs)
    rng = jax.random.PRNGKey(training_args.seed)
    # dropout_rngs = jax.random.split(rng, jax.local_device_count())

    train_batch_size = int(training_args.per_device_train_batch_size) * alpa.get_global_num_devices()
    # train_batch_size += 1
    # train_batch_size = 2240
    pollux_agent.total_batch_size = train_batch_size
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * alpa.get_global_num_devices()
    # endregion
    count = training_args.count
    scale = training_args.scale
    theta = training_args.smoothing * scale
    
    pollux_agent.last_state_retrieved_batch_size = train_batch_size
    pollux_agent.dataset_size = len(train_dataset_pytorch)

    pollux_agent.alloc_config_regressor[(1, 1)].coef_ = np.array([0.004806])
    pollux_agent.alloc_config_regressor[(1, 1)].intercept_ = 0.021575327546867917
    pollux_agent.alloc_config_regressor[(2, 1)].coef_ = np.array([0.00226393])
    pollux_agent.alloc_config_regressor[(2, 1)].intercept_ = 0.031203793760921167
    pollux_agent.alloc_config_regressor[(4, 1)].coef_ = np.array([0.00112445])
    pollux_agent.alloc_config_regressor[(4, 1)].intercept_ = 0.030955123211688196
    pollux_agent.fix_regressors()

    train_loader = alpa.adaptdl.dataloader.AdaptiveDataLoader(
        dataset=train_dataset_pytorch,
        batch_size=train_batch_size,
        shuffle=True, # TODO: handle shuffle=True
        # num_workers=data_args.preprocessing_num_workers,
        # persistent_workers=True,
        drop_last=True,
    )

    train_loader.autoscale_batch_size(max_batch_size = 280, 
                                    local_bsz_bounds=(train_batch_size, 70), gradient_accumulation=False)

    # train_loader = DataLoader(train_dataset_pytorch, batch_size=train_batch_size, shuffle=True, drop_last=True)


    if not training_args.pretrain:
        print(f"Finetuning mode")
        # region Load model
        model = FlaxAutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
        )
    else:
        print(f"Pretraining mode")
        model = FlaxAutoModelForQuestionAnswering.from_config(
            config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
        )

    run = wandb.init(
        # Set the project where this run will be logged
        project="BERT_adaptive",
        config={
            "model_name_or_path": model_args.model_name_or_path,
            "model_args": model_args,
            "data_args": data_args,
            "training_args": training_args,
        }
    )

    run.log_code(
                os.path.dirname(os.path.dirname(alpa.__file__)),
                exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("third_party/")
                )


    learning_rate_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # scaling_rule = LinearScale()
    scaling_rule = SqrtScale()
    scaled_learning_rate_fn = create_scaled_lr_fn(original_lr_fn=learning_rate_fn, initial_batch_size=train_batch_size,
                                                            scaling_rule=scaling_rule)

    # state = create_train_state(model, learning_rate_fn, num_labels=max_seq_length, training_args=training_args)
    state = create_train_state(model, scaled_learning_rate_fn, num_labels=max_seq_length, training_args=training_args)
    # endregion

    gns.initialize_gns(state=extract_values_with_key_p(state.params), 
                    init_bsz=train_batch_size, 
                    num_workers=alpa.get_global_num_devices(), 
                    accum_scale=alpa.get_global_num_devices())

    # region Define train step functions
    def train_step(
        state: train_state.TrainState, batch: Dict[str, Array], dropout_rng: PRNGKey, variables: Dict
    ) -> Tuple[train_state.TrainState, float]:
        """Trains model with an optimizer (both in `state`) on `batch`, returning a pair `(new_state, loss)`."""
        # dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        start_positions = batch.pop("start_positions")
        end_positions = batch.pop("end_positions")
        targets = (start_positions, end_positions)

        def loss_fn(params):
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)
            loss = state.loss_fn(logits, targets)
            return loss

        dropout_rng = variables.get('dropout_rng', None)
        prev_grads = variables.get('gns_store_grads', None)
        biased_sqr = variables.get('gns_biased_sqr', None)
        unbias_sqr = variables.get('gns_unbias_sqr', None)
        biased_var = variables.get('gns_biased_var', None)
        unbias_var = variables.get('gns_unbias_var', None)
        count = variables.get('count', None)
        scale = variables.get('scale', None)
        theta = variables.get('theta', None)

        grad_fn = alpa.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grad)

        pinv = jax.tree_util.tree_map(jnp.ones_like, grad)
        gradients = extract_values_with_key_p(grad)
        preconditioners = extract_values_with_key_p(pinv)

        def condition(prev_grads):
            return prev_grads is not None
        
        prev_grads = jax.lax.cond(
            condition(prev_grads),
            lambda x: x,
            lambda x: jax.tree_util.tree_map(jnp.zeros_like, gradients), 
            prev_grads
        )

        grad_sqr, grad_var, biased_sqr, unbias_sqr, biased_var, unbias_var = compute_gradient_noise_scale(prev_grads, gradients,
                                                                                                    preconditioners, 
                                                                                                    biased_sqr, 
                                                                                                    unbias_sqr, 
                                                                                                    biased_var, 
                                                                                                    unbias_var, 
                                                                                                    count, 
                                                                                                    scale, 
                                                                                                    theta)

        metrics = {"loss": loss, 
            "learning_rate": scaled_learning_rate_fn(state.step), 
            "gradients": gradients, 
            "grad_sqr": grad_sqr, 
            "grad_var": grad_var, 
            "biased_sqr": biased_sqr,
            "unbias_sqr": unbias_sqr, 
            "biased_var": biased_var, 
            "unbias_var": unbias_var 
            }

        return new_state, metrics

    # p_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))
    # p_train_step = jax.jit(train_step, donate_argnums=(0,))
    # method = alpa.ShardParallel()
    method = alpa.DataParallel()
    # method = alpa.PipeshardParallel(num_micro_batches=32, stage_option="auto")
    # p_train_step = alpa.parallelize(train_step, method=method, donate_argnums=(0,))
    p_train_step = alpa.parallelize(train_step, method=method)
    # endregion

    # region Define eval step functions
    def eval_step(state, batch):
        logits = state.apply_fn(**batch, params=state.params, train=False)
        return state.logits_fn(logits)

    # p_eval_step = jax.pmap(eval_step, axis_name="batch")
    # p_eval_step = jax.jit(eval_step)
    p_eval_step = alpa.parallelize(eval_step)
    # endregion

    # region Define train and eval loop
    logger.info(f"===== Starting training ({num_epochs} epochs) =====")
    train_time = 0

    # make sure weights are replicated on each device
    # state = replicate(state)

    train_time = 0
    step_per_epoch = len(train_dataset) // train_batch_size
    total_steps = step_per_epoch * num_epochs
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

    dump_debug_info_train_step = dump_debug_info_eval_step = True
    t_throughput = time.time()
    
    # for epoch in epochs:
    for epoch in alpa.adaptdl.epoch.remaining_epochs_until(num_epochs):
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # # train
        # for step, batch in enumerate(
        #     tqdm(
        #         train_data_collator(input_rng, train_dataset, train_batch_size),
        #         total=step_per_epoch,
        #         desc="Training...",
        #         position=1,
        #     ),
        #     1,
        # ):
        for step, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.numpy() for k, v in batch.items()}
            dropout_rng, rng = jax.random.split(rng)

            variables_dict = {'dropout_rng': dropout_rng, 'gns_store_grads': gns.store_grads, 'gns_biased_sqr': gns.biased_sqr, 'gns_unbias_sqr': gns.unbias_sqr, 'gns_biased_var': gns.biased_var, 'gns_unbias_var': gns.unbias_var, 'count': count, 'scale': scale, 'theta': theta}

            variables_dict = dict(variables_dict)

            if pollux_agent.reallocation_approaching:
                p_train_step.get_last_executable().sync()

                materialized_variables_dict = {}
                for k, v in variables_dict.items():
                    if isinstance(v, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                        materialized_variables_dict[k] = v._value
                    elif isinstance(v, list):
                        materialized_list = []
                        for el in v:
                            if isinstance(el, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                                materialized_list.append(el._value)
                            else:
                                materialized_list.append(el)
                        materialized_variables_dict[k] = materialized_list
                    else:
                        materialized_variables_dict[k] = v
                variables_dict = materialized_variables_dict
                if isinstance(pollux_agent.grad_norm_sqr_abstract, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)) \
                     and isinstance(pollux_agent.grad_variance_abstract, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                    pollux_agent.grad_norm_sqr_abstract = pollux_agent.grad_norm_sqr = pollux_agent.grad_norm_sqr_abstract._value.item()
                    pollux_agent.grad_variance_abstract = pollux_agent.grad_variance = pollux_agent.grad_variance_abstract._value.item()

                state = reallocate_and_update_state(state)

                continue # TODO: doing this temporarily to force dataloader batch size change, discards current batch size

            state, train_metric = p_train_step(state, batch, dropout_rng, variables_dict)

            gns.update_state(state, train_metric["grad_sqr"], train_metric["grad_var"], train_metric["biased_sqr"], train_metric["unbias_sqr"], 
                    train_metric["biased_var"], train_metric["unbias_var"], train_metric["gradients"])
            update_grad_params(train_metric["grad_sqr"], train_metric["grad_var"])

            cur_step = epoch * step_per_epoch + step

            if dump_debug_info_train_step:
                dump_debug_info_train_step = False
                executable = p_train_step.get_last_executable()
                executable.sync()
                executable.dump_debug_info("alpa_debug_info")
                epochs.write(f"Initial compilation completed. "
                             f"Time elapsed: {time.time() - train_start:.2f} s")

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                # train_metric = unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                train_metrics = []

                elapsed_throughput = time.time() - t_throughput
                throughput = training_args.logging_steps * train_batch_size / elapsed_throughput
                t_iter = elapsed_throughput / training_args.logging_steps
                t_throughput = time.time()

                epochs.write(
                    f"Step... ({cur_step}/{total_steps} | Training Loss: {train_metric['loss']}, Learning Rate:"
                    f" {train_metric['learning_rate']} | Throughput: {throughput}) | T_iter: {t_iter}"
                )


            # if (
            #     training_args.do_eval
            #     and (cur_step % training_args.eval_steps == 0 or cur_step % step_per_epoch == 0)
            #     and cur_step > 0
            # ):
            #     #updating step
            #     step_val = state.step._value
    
            #     flattened_opt_state = tree_flatten(state.opt_state)
            #     for i, leaf in enumerate(flattened_opt_state[0]):
            #         if isinstance(leaf, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            #             flattened_opt_state[0][i] = leaf._value
            #     unflattened_opt_state = tree_unflatten(flattened_opt_state[1], flattened_opt_state[0])
                
            #     flattened_params = tree_flatten(state.params)
            #     for i, leaf in enumerate(flattened_params[0]):
            #         if isinstance(leaf, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            #             flattened_params[0][i] = leaf._value
            #     unflattened_params = tree_unflatten(flattened_params[1], flattened_params[0])
                
            #     state = state.replace(step=step_val, opt_state=unflattened_opt_state, params=unflattened_params)
            #     #updating step
                
            #     eval_metrics = {}
            #     all_start_logits = []
            #     all_end_logits = []
            #     # evaluate
            #     for batch in tqdm(
            #         eval_data_collator(eval_dataset, eval_batch_size),
            #         total=math.ceil(len(eval_dataset) / eval_batch_size),
            #         desc="Evaluating ...",
            #         position=2,
            #     ):
            #         _ = batch.pop("example_id")
            #         _ = batch.pop("offset_mapping")
            #         # predictions = pad_shard_unpad(p_eval_step)(
            #             # state, batch, min_device_batch=per_device_eval_batch_size
            #         # )
            #         predictions = p_eval_step(state, batch)
            #         start_logits = np.array(predictions[0])
            #         end_logits = np.array(predictions[1])
            #         all_start_logits.append(start_logits)
            #         all_end_logits.append(end_logits)

            #     max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

            #     # concatenate the numpy array
            #     start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
            #     end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

            #     # delete the list of numpy arrays
            #     del all_start_logits
            #     del all_end_logits
            #     outputs_numpy = (start_logits_concat, end_logits_concat)
            #     prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
            #     eval_metrics = compute_metrics(prediction)

            #     logger.info(f"Step... ({cur_step}/{total_steps} | Evaluation metrics: {eval_metrics})")

            #     if has_tensorboard and jax.process_index() == 0:
            #         write_eval_metric(summary_writer, eval_metrics, cur_step)

            # if (cur_step % training_args.save_steps == 0 and cur_step > 0) or (cur_step == total_steps):
            #     # save checkpoint after each epoch and push checkpoint to the hub
            #     if jax.process_index() == 0:
            #         # params = jax.device_get(unreplicate(state.params))
            #         # params = state.params
            #         # Getting proper values from the state to save the model weights
            #         flattened_params = tree_flatten(state.params)
            #         for i, leaf in enumerate(flattened_params[0]):
            #             if isinstance(leaf, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            #                 flattened_params[0][i] = leaf._value
            #         unflattened_params = tree_unflatten(flattened_params[1], flattened_params[0])
            #         params = unflattened_params
            #         model.save_pretrained(training_args.output_dir, params=params)
            #         tokenizer.save_pretrained(training_args.output_dir)
            #         if training_args.push_to_hub:
            #             repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)
        epochs.desc = f"Epoch ... {epoch + 1}/{num_epochs}"
    # endregion

    # Eval after training
    if training_args.do_eval:
        eval_metrics = {}
        all_start_logits = []
        all_end_logits = []

        eval_loader = eval_data_collator(eval_dataset, eval_batch_size)
        for batch in tqdm(
            eval_loader, total=math.ceil(len(eval_dataset) / eval_batch_size), desc="Evaluating ...", position=2
        ):
            _ = batch.pop("example_id")
            _ = batch.pop("offset_mapping")
            # predictions = pad_shard_unpad(p_eval_step)(state, batch, min_device_batch=per_device_eval_batch_size)
            predictions = p_eval_step(state, batch)
            start_logits = np.array(predictions[0])
            end_logits = np.array(predictions[1])
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits
        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
        eval_metrics = compute_metrics(prediction)

        if jax.process_index() == 0:
            eval_metrics = {f"eval_{metric_name}": value for metric_name, value in eval_metrics.items()}
            path = os.path.join(training_args.output_dir, "eval_results.json")
            with open(path, "w") as f:
                json.dump(eval_metrics, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
