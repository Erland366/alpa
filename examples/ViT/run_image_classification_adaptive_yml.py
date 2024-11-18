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
Pre-training/Fine-tuning ViT for image classification .
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=vit
"""

import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Dict
from jax.tree_util import tree_flatten, tree_unflatten

# for dataset and preprocessing
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import alpa
from alpa.model.model_util import TrainState
import jax
import jax.numpy as jnp
import optax
import transformers
from flax.training.common_utils import onehot
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    FlaxAutoModelForImageClassification,
    HfArgumentParser,
    is_tensorboard_available,
    set_seed,
)
from transformers.utils import get_full_repo_name, send_example_telemetry
#from alpa.adaptdl.gradient_noise_scale import GradientNoiseScale
from alpa.adaptdl.gns_util import (extract_values_with_key_p, 
                                   compute_gradient_noise_scale, 
                                   init_distributed_scalar,
                                   init_distributed_zeros_like,
                                   flatten_and_concat)
from alpa.adaptdl.dataloader import current_dataloader
from alpa.adaptdl.metrics import update_grad_params, update_progress
from jax._src.config import flags
#import numpy as np
from alpa.adaptdl.pollux_agent import pollux_agent
from alpa.adaptdl.api import update_state_on_bs_change, create_scaled_lr_fn, reallocate_and_update_state
import alpa.adaptdl.dataloader
import alpa.adaptdl.epoch
from alpa.adaptdl.scaling_rules import ScalingRuleBase, LinearScale, SqrtScale
import datetime
import numpy as np
import wandb
import yaml
from addict import Dict as AddictDict
import argparse


def parse_config_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yml file")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining  # Removing --config from sys.argv
    return args.config

config_path = parse_config_arg()

with open(config_path, 'r') as file:
    yml_config = yaml.safe_load(file)
yml_config = AddictDict(yml_config)

def count_params(model):
    return sum(x.size for x in jax.tree_leaves(model))
    
# alpa.init(cluster="ray")
# alpa.init(cluster="ray", num_nodes=1, num_devices_per_node=2, namespace="alpa_default_space_vit")
# alpa.init(cluster="ray", scheduler_address="http://127.0.0.1:8000")
alpa.init(cluster="ray", copus_enabled=True, scheduler_address=yml_config.scheduler.address if yml_config.scheduler.enabled else None,
          num_nodes=yml_config.cluster_config.num_nodes, num_devices_per_node=yml_config.cluster_config.num_devices_per_node,
          namespace=yml_config.cluster_config.namespace)

logger = logging.getLogger(__name__)

# handler = logging.FileHandler(f"/home/haifatl/Documents/alpa/alpa-adaptdl-feb11/alpa-adaptdl/examples/ViT/logs/gradsqr_gradvar_vit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)



MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default=yml_config.paths.output_dir,
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
    num_micro_batches: int = field(default=1, metadata={"help": "The number of micro batches for gradient accumulation."})
    per_device_train_batch_size: int = field(
        default=yml_config.dataloader.train.init_local_batch_size, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=yml_config.dataloader.eval.local_batch_size, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=yml_config.training.learning_rate, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=yml_config.training.num_train_epochs, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
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
        default=yml_config.training.pretrain, metadata={"help": "Whether or not to pretrain."}
    )
    count: int = field(default=2, metadata={"help": "The number of stored grads."})
    scale: int = field(default=1, metadata={"help": "Scale"})
    smoothing: float = field(default=0.9, metadata={"help": "Smoothing parameter for PGNS"})
    scale_lr: bool = field(
        default=yml_config.training.scale_lr.enabled, metadata={"help": "Whether or not to scale the learning rate with batch size."}
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
        default=yml_config.model_name_or_path,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=yml_config.paths.cache_dir, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    dtype: Optional[str] = field(
        default=yml_config.training.dtype,
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dir: str = field(
        default=yml_config.paths.train_dir,
        metadata={"help": "Path to the root training directory which contains one subdirectory per class."}
    )
    validation_dir: str = field(
        default=yml_config.paths.validation_dir,
        metadata={"help": "Path to the root validation directory which contains one subdirectory per class."},
    )
    image_size: Optional[int] = field(default=224, metadata={"help": " The size (resolution) of each image."})
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=yml_config.dataloader.train.preprocessing_num_workers,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

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

def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = alpa.util.get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn

def get_current_batch_size():
    return pollux_agent.total_batch_size


def main():
    
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

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_classification", model_args, data_args, framework="flax")

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # set seed for random transforms and torch dataloaders
    set_seed(training_args.seed)

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name, token=training_args.hub_token
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

    # Initialize datasets and pre-processing transforms
    # We use torchvision here for faster pre-processing
    # Note that here we are using some default pre-processing, for maximum accuray
    # one should tune this part and carefully select what transformations to use.
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_dataset = torchvision.datasets.ImageFolder(
        data_args.train_dir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(data_args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    eval_dataset = torchvision.datasets.ImageFolder(
        data_args.validation_dir,
        transforms.Compose(
            [
                transforms.Resize(data_args.image_size),
                transforms.CenterCrop(data_args.image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    # Load pretrained model and tokenizer
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            num_labels=len(train_dataset.classes),
            image_size=data_args.image_size,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=len(train_dataset.classes),
            image_size=data_args.image_size,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.model_name_or_path and not training_args.pretrain:
        print(f"Finetuning mode")
        model = FlaxAutoModelForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        print(f"Pretraining mode")
        model = FlaxAutoModelForImageClassification.from_config(
            config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
        )

    run = wandb.init(
        # Set the project where this run will be logged
        project=yml_config.wandb.project,
        config={
            "model_name_or_path": model_args.model_name_or_path,
            "model_args": model_args,
            "data_args": data_args,
            "training_args": training_args,
            "yml_config": yml_config,
        },
        save_code=yml_config.wandb.save_code,
        mode=yml_config.wandb.mode,
        # settings=wandb.Settings(code_dir=os.path.dirname(os.path.dirname(alpa.__file__)))
    )
    run.log_code(
                    os.path.dirname(os.path.dirname(alpa.__file__)),
                    exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("third_party/")
                    )

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * alpa.get_global_num_devices()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * alpa.get_global_num_devices()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs
    count = training_args.count
    scale = training_args.scale
    theta = training_args.smoothing * scale

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples])

        batch = {"pixel_values": pixel_values, "labels": labels}
        batch = {k: v.numpy() for k, v in batch.items()}

        return batch

    pollux_agent.total_batch_size = train_batch_size
    pollux_agent.last_state_retrieved_batch_size = train_batch_size
    pollux_agent.dataset_size = len(train_dataset)

    # Set regression coefficients if specified in the config
    if yml_config.pollux_agent.fix_regressors:
        for item in yml_config.pollux_agent.regression_coefficients:
            key = tuple(item.key)  # Convert list in .yml to tuple
            values = item
            pollux_agent.alloc_config_regressor[key].coef_ = np.array([values.coef])
            pollux_agent.alloc_config_regressor[key].intercept_ = values.intercept
        pollux_agent.fix_regressors()

    # Create data loaders
    if not yml_config.dataloader.train.adaptive_data_loader.enabled:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=yml_config.dataloader.train.shuffle,
            num_workers=data_args.preprocessing_num_workers,
            persistent_workers=yml_config.dataloader.train.persistent_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )
    else:
        train_loader = alpa.adaptdl.dataloader.AdaptiveDataLoader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            shuffle=yml_config.dataloader.train.shuffle, # TODO: handle shuffle=True
            collate_fn=collate_fn,
            num_workers=data_args.preprocessing_num_workers,
            persistent_workers=yml_config.dataloader.train.persistent_workers,
            drop_last=True,
        )
        if yml_config.dataloader.train.adaptive_data_loader.autoscaler.enabled:
            train_loader.autoscale_batch_size(max_batch_size = yml_config.dataloader.train.adaptive_data_loader.autoscaler.max_total_batch_size, 
                                                local_bsz_bounds=(train_batch_size // alpa.get_global_num_devices() if 
                                                                  yml_config.dataloader.train.adaptive_data_loader.autoscaler.local_bsz_bounds.min_is_init_bs
                                                                  else yml_config.dataloader.train.adaptive_data_loader.autoscaler.local_bsz_bounds.min, 
                                                                  yml_config.dataloader.train.adaptive_data_loader.autoscaler.local_bsz_bounds.max), 
                                                gradient_accumulation=False)

    eval_loader = torch.utils.data.DataLoader(
       eval_dataset,
       batch_size=eval_batch_size,
       shuffle=yml_config.dataloader.eval.shuffle,
       num_workers=data_args.preprocessing_num_workers,
       persistent_workers=True,
       drop_last=True,
       collate_fn=collate_fn,
    )

    # eval_loader = alpa.adaptdl.dataloader.AdaptiveDataLoader(
    #     dataset=eval_dataset,
    #     batch_size=eval_batch_size,
    #     shuffle=False, 
    #     drop_last=False,
    #     collate_fn=collate_fn,
    # )

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
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

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    if yml_config.training.scale_lr.type == 'sqrt':
        scaling_rule = SqrtScale()
    else:
        scaling_rule = LinearScale()
    
    # TODO: initial batch size should probably be stored separately to avoid newer batch size being set after a checkpoint-restart
    scaled_linear_decay_lr_schedule_fn = create_scaled_lr_fn(original_lr_fn=linear_decay_lr_schedule_fn, initial_batch_size=train_batch_size,
                                                             scaling_rule=scaling_rule)

    if not training_args.scale_lr:
        scaled_linear_decay_lr_schedule_fn = linear_decay_lr_schedule_fn

    # create adam optimizer
    adamw = optax.adamw(
        #learning_rate=linear_decay_lr_schedule_fn,
        learning_rate=scaled_linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    #adamw = optax.sgd(learning_rate=scaled_linear_decay_lr_schedule_fn)

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw, dynamic_scale=None)
    gns.initialize_gns(state=state, 
                       init_bsz=train_batch_size, 
                       num_workers=alpa.get_global_num_devices(), 
                       accum_scale=alpa.get_global_num_devices(),
                       store_grads=flatten_and_concat(extract_values_with_key_p(state.params)))
    
    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        return loss.mean()

    # Define gradient update step fn
    def train_step(state, 
                   batch, 
                   variables: Dict 
                   #dropout_rng: PRNGKey, prev_grads, biased_sqr, unbias_sqr, biased_var, unbias_var, count, scale, theta
                   ):

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, train=True)[0]
            loss = loss_fn(logits, labels)
            return loss
        
        dropout_rng = variables.get('dropout_rng', None)
        if yml_config.training.gns_enabled:
            prev_grads = variables.get('gns_store_grads', None)
            biased_sqr = variables.get('gns_biased_sqr', None)
            unbias_sqr = variables.get('gns_unbias_sqr', None)
            biased_var = variables.get('gns_biased_var', None)
            unbias_var = variables.get('gns_unbias_var', None)
            count = variables.get('count', None)
            scale = variables.get('scale', None)
            theta = variables.get('theta', None)
        
        grad_fn = alpa.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grad)

        if yml_config.training.gns_enabled:
            pinv = jax.tree_util.tree_map(jnp.ones_like, grad)
            gradients = flatten_and_concat(extract_values_with_key_p(grad))
            preconditioners = flatten_and_concat(extract_values_with_key_p(pinv))
        
            # Basing GNS estimation on the first 10% gradients
            # –––––––––––––––––––––––––––––––––––––––––––––
            first_10_percent = int(round(gradients.shape[0] * 10 / 100))
            gradients = gradients[:first_10_percent]
            preconditioners = preconditioners[:first_10_percent]
            # –––––––––––––––––––––––––––––––––––––––––––––

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
                "learning_rate": scaled_linear_decay_lr_schedule_fn(state.step), 
                "gradients": gradients, 
                "grad_sqr": grad_sqr, 
                "grad_var": grad_var, 
                "biased_sqr": biased_sqr,
                "unbias_sqr": unbias_sqr, 
                "biased_var": biased_var, 
                "unbias_var": unbias_var,
                }
        else:
            metrics = {"loss": loss, 
                "learning_rate": scaled_linear_decay_lr_schedule_fn(state.step),
                }

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)

        # summarize metrics
        accuracy = (jnp.argmax(logits, axis=-1) == labels).mean()
        metrics = {"loss": loss, "accuracy": accuracy}
        return metrics

    # Create parallel version of the train and eval step
    if yml_config.training.parallel_method.method == 'ShardParallel':
        method = alpa.ShardParallel(num_micro_batches=yml_config.training.parallel_method.num_micro_batches if yml_config.training.parallel_method.num_micro_batches != 1 else None)
    elif yml_config.training.parallel_method.method == 'PipeshardParallel':
        stage_option = yml_config.training.parallel_method.parameters.PipeshardParallel.stage_option
        method = alpa.PipeshardParallel(stage_option=stage_option, num_micro_batches=yml_config.training.parallel_method.num_micro_batches)
    elif yml_config.training.parallel_method.method == 'DataParallel':
        method = alpa.DataParallel(num_micro_batches=yml_config.training.parallel_method.num_micro_batches if yml_config.training.parallel_method.num_micro_batches != 1 else None)
    elif yml_config.training.parallel_method.method == '3D':
        method = alpa.get_3d_parallel_method(num_micro_batches=yml_config.training.parallel_method.num_micro_batches,
                                             data_parallel=yml_config.training.parallel_method.parameters._3D.data_parallel,
                                             operator_parallel=yml_config.training.parallel_method.parameters._3D.operator_parallel,
                                             pipeline_parallel=yml_config.training.parallel_method.parameters._3D.pipeline_parallel)
    else:
        method = alpa.DataParallel()

    p_train_step = alpa.parallelize(train_step, method=method, donate_argnums=(0,))
    p_eval_step = alpa.parallelize(eval_step)
    dump_debug_info_train_step = dump_debug_info_eval_step = True

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")
    
    logger.info(f"Number of parameters - {count_params(model.params)}")

    train_time = 0
    last_time = time.time()
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

    if yml_config.training.gns_enabled:
        gns.store_grads = init_distributed_zeros_like(gns.store_grads, percent=10)
        gns.biased_sqr = init_distributed_scalar()
        gns.unbias_sqr = init_distributed_scalar()
        gns.biased_var = init_distributed_scalar()
        gns.unbias_var = init_distributed_scalar()


    for epoch in alpa.adaptdl.epoch.remaining_epochs_until(num_epochs):
        # ======================== Training ================================
        
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_metrics = []
        noise_scale_list = []
        steps_list = []
        losses = []
        epoch_losses = []
        eval_losses = []
        eval_acc = []

        steps_per_epoch = len(train_dataset) // train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for step, batch in enumerate(train_loader):
            #if isinstance(p_train_step.method, alpa.PipeshardParallel) and \
            #    (p_train_step.method == 'auto' or isinstance(p_train_step.method, alpa.AutoLayerOption)):
            #    state = update_state_on_bs_change(state)
            # exec_time = time.perf_counter()
            

            variables_dict = {'dropout_rng': dropout_rng, 'gns_store_grads': gns.store_grads, 'gns_biased_sqr': gns.biased_sqr, 'gns_unbias_sqr': gns.unbias_sqr, 'gns_biased_var': gns.biased_var, 'gns_unbias_var': gns.unbias_var, 'count': count, 'scale': scale, 'theta': theta}
            #state, train_metric = p_train_step(state, batch, dropout_rng, gns.store_grads, 
            #                                    gns.biased_sqr, gns.unbias_sqr, gns.biased_var, gns.unbias_var, count, scale, theta)

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

                if isinstance(gns.store_grads, list):
                    store_grads_materialized = []
                    for el in gns.store_grads:
                        if isinstance(el, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                            store_grads_materialized.append(el._value)
                        else:
                            store_grads_materialized.append(el)
                    gns.store_grads = store_grads_materialized
                if isinstance(gns.biased_sqr, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                    gns.biased_sqr = gns.biased_sqr._value
                if isinstance(gns.unbias_sqr, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                    gns.unbias_sqr = gns.unbias_sqr._value
                if isinstance(gns.biased_var, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                    gns.biased_var = gns.biased_var._value
                if isinstance(gns.unbias_var, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                    gns.unbias_var = gns.unbias_var._value

                state = reallocate_and_update_state(state)

                continue # TODO: doing this temporarily to force dataloader batch size change, discards current batch size
            
            state, train_metric = p_train_step(state, batch, variables_dict)

            # exec_time = time.perf_counter() - exec_time
            # logger.info(f'train_step time: {exec_time}')
            # gns_update_time = time.perf_counter()
            
            # epoch_losses.append(train_metric['loss']._value)
            
                
            # print(f'grad_sqr: {train_metric["grad_sqr"]._value}, grad_var: {train_metric["grad_var"]._value}')
            
            # logger.info(f'epoch: {epoch}, step: {step}')
            # logger.info(f'grad_sqr: {train_metric["grad_sqr"]._value}, grad_var: {train_metric["grad_var"]._value}')

            if yml_config.training.gns_enabled:
                gns.update_state(state, train_metric["grad_sqr"], train_metric["grad_var"], train_metric["biased_sqr"], train_metric["unbias_sqr"], 
                        train_metric["biased_var"], train_metric["unbias_var"], train_metric["gradients"])

                update_grad_params(train_metric["grad_sqr"], train_metric["grad_var"])
            
            #train_metrics.append(train_metric)
        
            cur_step = epoch * (len(train_dataset) // train_batch_size) + step

            if dump_debug_info_train_step:
                dump_debug_info_train_step = False
                executable = p_train_step.get_last_executable()
                executable.sync()
                executable.dump_debug_info("alpa_debug_info")
                epochs.write(f"Initial compilation completed. "
                             f"Time elapsed: {time.time() - train_start:.2f} s")
                             
            train_step_progress_bar.update(1)

        latency = time.time() - last_time
        images_per_second = len(train_dataset) / latency
        train_time += time.time() - train_start
        last_time = time.time()

        logger.info(f"train_loss: {train_metric['loss']}")
        

        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate:"
            f" {train_metric['learning_rate']}), "
            f"Throughput: {images_per_second:.2f} images/s, "   
        )
        
    

        # # ======================== Evaluating ==============================
        if yml_config.evaluation.enabled:
            eval_metrics = []
            eval_steps = max(len(eval_dataset) // eval_batch_size, 1)
            eval_step_progress_bar = tqdm(total=eval_steps, desc="Evaluating...", position=2, leave=False)
            for batch in eval_loader:
                # Model forward
                metrics = p_eval_step(state.params, batch)
                eval_metrics.append(metrics)

                if dump_debug_info_eval_step:
                    dump_debug_info_eval_step = False
                    executable = p_eval_step.get_last_executable()
                    executable.dump_debug_info("alpa_debug_info")

                eval_step_progress_bar.update(1)

            # normalize eval metrics
            eval_metrics = alpa.util.get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

            logger.info(f"eval_loss: {eval_metrics['loss'].item()}")
            logger.info(f"eval_acc: {eval_metrics['accuracy'].item()}")

            # Print metrics and update progress bar
            eval_step_progress_bar.close()
            desc = (
                f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {round(eval_metrics['loss'].item(), 4)} | "
                f"Eval Accuracy: {round(eval_metrics['accuracy'].item(), 4)})"
            )
            epochs.write(desc)
            epochs.desc = desc

            # Save metrics
            if has_tensorboard and jax.process_index() == 0:
                cur_step = epoch * (len(train_dataset) // train_batch_size)
                write_metric(summary_writer, train_metrics, eval_metrics, train_time, cur_step)

            # save checkpoint after each epoch and push checkpoint to the hub
            if jax.process_index() == 0:
                alpa.prefetch(state.params)
                params = alpa.util.map_to_nparray(state.params)
                model.save_pretrained(training_args.output_dir, params=params)
                if training_args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)

    


if __name__ == "__main__":
    main()