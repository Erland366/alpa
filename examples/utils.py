import inspect
import linecache
import re
import argparse
import sys
import addict
import yaml

import jax
import alpa
import os
import logging
import typing
import tensorflow as tf
import transformers
import numpy as np
from datasets import Dataset

from dataclasses import asdict, dataclass, field
from typing import Optional, Callable, Union, Dict, cast, Protocol, Literal, Any
from enum import Enum

import optax
from optax._src import base
import jax.numpy as jnp
from flax import traverse_util
from transformers import (
    is_wandb_available,
    is_tensorboard_available,
    TrainingArguments,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
)

__all__ = [
    "ModelArguments",
    "DataTrainingArguments",
    "TrainingArguments",
    "ImageAugmentationArguments",
    "HAS_WANDB",
    "HAS_TENSORBOARD",
    "DEBUG",
    "write_train_metric",
    "write_eval_metric",
    "write_metric",
    "mb_item",
    "make_batch",
    "create_learning_rate_fn",
    "decay_mask_fn",
    "init_alpa",
    "maybe_stop_wandb",
    "login_wandb",
    "instantiate_wandb",
    "setup_logging",
    "tree_map_params",
    "create_alpa_method",
    "AlpaMethod",
    "count_params",
    "data_loader",
    "init_debug",
    "setup_experiment_logging",
    "create_dynamic_function",
    "monkeypatch_rope_llama",
    "monkeypatch_rope_gemma",
    "parse_args",
    "get_profiling_setup",
    "reset_alpa_state",
    "run_profile"
]

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    yaml_parser = argparse.ArgumentParser(add_help=False)
    yaml_parser.add_argument("--config", type=str, help="Path to the config.yml file")
    yaml_args, remaining = yaml_parser.parse_known_args()
    
    yaml_config = {}
    
    if yaml_args.config:
        with open(yaml_args.config, 'r') as file:
            yaml_config = yaml.safe_load(file)
            yaml_config = addict.Dict(yaml_config) if addict else yaml_config

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    
    if len(remaining) == 1 and remaining[0].endswith(".json"):
        
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(remaining[0])
        )
    else:
        
        sys.argv = [sys.argv[0]] + remaining  
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if yaml_config:
        if 'model' in yaml_config:
            for key, value in yaml_config.model.items():
                if not getattr(model_args, key, None):
                    setattr(model_args, key, value)
                    
        if 'data' in yaml_config:
            for key, value in yaml_config.data.items():
                if not getattr(data_args, key, None):
                    setattr(data_args, key, value)
        
        if 'training' in yaml_config:
            for key, value in yaml_config.training.items():
                if not getattr(training_args, key, None):
                    setattr(training_args, key, value)
    
    return model_args, data_args, training_args

def setup_experiment_logging():
    global HAS_WANDB
    global HAS_TENSORBOARD
    global DEBUG

    HAS_WANDB = is_wandb_available()
    HAS_TENSORBOARD = is_tensorboard_available()
    DEBUG = os.getenv("DEBUG", True) # For now it's flipped

    if DEBUG:
        HAS_WANDB = False
        HAS_TENSORBOARD = False
        os.environ["WANDB_DISABLED"] = "true"

setup_experiment_logging()

class AlpaMethod(str, Enum):
    ZERO2 = "zero2"
    ZERO3 = "zero3"
    PARALLEL_3D = "parallel_3d"
    PIPESHARD = "pipeshard"
    SHARD_PARALLEL = "shard_parallel"
    DATA_PARALLEL = "data_parallel"

def create_alpa_method(method: AlpaMethod, training_args, *args, **kwargs):
    num_micro_batches = training_args.num_micro_batches
    if method == AlpaMethod.ZERO2:
        parallel_method = alpa.Zero2Parallel(
            *args, num_micro_batches=num_micro_batches, **kwargs
        )
    elif method == AlpaMethod.ZERO3:
        parallel_method = alpa.Zero3Parallel(
            *args, num_micro_batches=num_micro_batches, **kwargs
        )
    elif method == AlpaMethod.PARALLEL_3D:
        parallel_method = alpa.get_3d_parallel_method(
            *args,
            num_micro_batches=num_micro_batches,
            data_parallel=training_args.data_parallel,
            operator_parallel=training_args.operator_parallel,
            pipeline_parallel=training_args.pipeline_parallel,
            **kwargs
        )
    elif method == AlpaMethod.PIPESHARD:
        parallel_method = alpa.PipeshardParallel(
            *args,
            num_micro_batches=num_micro_batches,
            pipeline_schedule=training_args.pipeline_schedule,
            stage_option="auto",
            ## Default value for layer_option here
            # layer_option=alpa.AutoLayerOption(layer_num=2),
            # layer_option="manual",
            **kwargs
        )
    elif method == AlpaMethod.SHARD_PARALLEL:
        parallel_method = alpa.ShardParallel(
            *args,
            num_micro_batches=num_micro_batches,
            auto_sharding_option=alpa.AutoShardingOption(allow_mixed_mesh_shape=True, force_batch_dim_to_mesh_dim=0),
            **kwargs
        )
    elif method == AlpaMethod.DATA_PARALLEL:
        parallel_method = alpa.DataParallel(
            *args,
            num_micro_batches=num_micro_batches,
            **kwargs
        )
    else:
        raise ValueError(f"Your {method} is not supported yet")

    print(f"{parallel_method = }")
    return parallel_method

def init_debug():
    if bool(os.getenv("ALPA_DEBUG", False)):
        from alpa import global_config

        os.environ["ALPA_DEBUG_PRINT_AS_STRATEGY"] = "1"
        os.environ["JAX_DISABLE_JIT"] = "True"
        global_config.pipeline_distributed_compile = False

        logger.warning(
            "You activate ALPA_DEBUG environment variable.",
            "Run will be much SLOWER.",
            "You should only ran minimal steps if using this method!"
        )

def create_dynamic_function(source, function_name, original_func=None, filename_prefix="<dynamic>"):
    """
    We assume that we already patch the function, the rest step is to just call 
    `exec(source, globals())`. This is to enable `inspect.getsource` on the patched function.
    """
    frame = inspect.currentframe().f_back
    unique_id = id(frame)
    frame_info = inspect.getframeinfo(frame)
    patch_filepath = os.path.abspath(frame_info.filename)
    patch_line_no = frame_info.lineno

    if original_func:
        original_func_name = original_func.__name__
        filename = f"{filename_prefix}-{original_func_name}-{patch_filepath}-{patch_line_no}-{unique_id}"
    else:
        filename = f"{filename_prefix}-{patch_filepath}-{patch_line_no}-{unique_id}"
    
    filename = re.sub(r"[^\w\-_\.]", "_", filename)

    code = compile(source, filename, "exec")
    globals_ = frame.f_globals
    locals_ = frame.f_locals

    temp_locals = {}
    exec(code, globals_, temp_locals)
    print(temp_locals)
    func = temp_locals[function_name]

    lines = [line + "\n" for line in source.split("\n")]
    linecache.cache[filename] = (
        len(source),
        None,
        lines,
        filename,
    )

    return func

def monkeypatch_rope_llama():
    exec("from transformers.models.llama import modeling_flax_llama", globals())
    source = inspect.getsource(modeling_flax_llama.FlaxLlamaRotaryEmbedding.__call__)
    start = source.find("def")
    source = source.split("\n")
    source = "\n".join([x[start:] for x in source])
    source = source.replace("key = apply", "# key = apply")
    source = source.replace("query = apply", "# query = apply")
    func = create_dynamic_function(source, "__call__")
    modeling_flax_llama.FlaxLlamaRotaryEmbedding.__call__ = func


def init_alpa(cluster: str = "ray", normalize_embedding_shape: bool = True):

    import alpa 

    tf.config.experimental.set_visible_devices([], "GPU")
    alpa.init(cluster=cluster)
    alpa.global_config.force_normalize_embedding_shapes = normalize_embedding_shape

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
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
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )
    save_optimizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to store full train state including optimizer."},
    )
    repo_path_or_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the modelhub repo directory"},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "Token for Huggingface"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether you want to run custom code from the repo of HuggingFace"}
    )
    use_auth_token: Optional[bool] = field(
        default=None,
        metadata={"help" : "Whether you want to use HF API TOKEN for loading dataset, model, etc"}
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
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to local folder containing data files."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # GPT 2 Arguments
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    # ViT Arguments
    train_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory for the training data."}
    )
    validation_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory for the validation data."}
    )
    image_size: Optional[int] = field(
        default=224, metadata={"help": "The size (resolution) of each image."}
    )

    # CLIP Arguments
    text_column_name: Optional[str] = field(
            default='text',
            metadata={"help": "Column containing main text data."},
    )
    augment_images: Optional[bool] = field(
        default=True,
        metadata={ "help": "Augment input training images" }
    )
    augment_captions: Optional[bool] = field(
        default=True,
        metadata={"help": "Augment input training images" }
    )
    captions_per_image: Optional[int] = field(
        default=5,
        metadata={"help": "Number of captions per image to use when creating train dataset."},
    )
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Whether to add max_seq_length to the "}
    )
    use_data_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use data sample or not which consists only 1000 data rows"}
    )

    def __post_init__(self):
        if ((self.dataset_name is None) and
         (self.train_file is None or self.validation_file is None) and
         (self.train_dir is None or self.validation_dir is None)
        ):
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt", "jsonl"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt", "jsonl"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class ImageAugmentationArguments:
    """
    Arguments for image augmentations configuration
    """
    random_horizontal_flip: Optional[float] = field(
        default=0.5,
        metadata={ "help": "Probability of applying random horizontal flip" }
    )
    random_vertical_flip: Optional[float] = field(
        default=0.5,
        metadata={ "help": "Probability of applying random vartical flip" }
    )

@dataclass
class TrainingArguments(TrainingArguments):
    # Alpa Training Strategy
    num_micro_batches: int = field(
        default=1, 
        metadata={"help": "The number of micro batches for gradient accumulation."}
    )
    operator_parallel: int = field(
        default=1, 
        metadata={"help": "The degree of operator model parallelism."}
    )
    pipeline_parallel: int = field(
        default=1, 
        metadata={"help": "The degree of pipeline model parallelism."}
    )
    data_parallel: int = field(
        default=-1,
        metadata={"help": "The degree of data parallelism. By default it will be -1 which means allocate the rest of the machine as data parallel"}
    )
    pipeline_schedule: Literal["1f1b", "gpipe", "inference"] = field(
        default="1f1b",
        metadata={"help": "The pipeline schedules."}
    )
    parallel_strategy: AlpaMethod = field(
        default="pipeshard",
        metadata={"help": "The parallel strategy that you can use for Alpa"}
    )
    use_remat: bool = field(
        default=True, 
        metadata={"help": "Whether or not to use gradient rematerilization/gradient checkpointing."}
    )
    entity: str = field(
        default="wandb",
        metadata={"help": "Entity for wandb"}
    )
    project: Optional[str] = field(
        default=None,
        metadata={"help": "Entity for wandb"}
    )
    manual_sharding: bool = field(
        default=False,
        metadata={"help": "Whether to manually shard the model or not."}
    )
    def __post_init__(self):
        import jax

        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

        if jax.process_index() == 0 and self.parallel_strategy == AlpaMethod.PARALLEL_3D:
            total_devices = len(jax.devices())
            tp = self.operator_parallel
            pp = self.pipeline_parallel
            dp = total_devices // (tp * pp)
            print(f"[DP]: {dp}, [TP]: {tp}, [PP]: {pp}")

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


def write_eval_metric_wandb(eval_metrics, step):
    import wandb

    for metric_name, value in eval_metrics.items():
        wandb.log({f"eval_{metric_name}": value}, step)


def write_train_metric(train_metrics, train_time, step, summary_writer=None):
    import jax
    if jax.process_index() == 0:
        import wandb
        import alpa
        
        if isinstance(train_metrics, list) and all(isinstance(item, dict) for item in train_metrics):
            for i, metrics_dict in enumerate(train_metrics):
                try:
                    metrics_dict = alpa.util.get_metrics(metrics_dict)
                except Exception as _:
                    pass
                
                current_step = step - len(train_metrics) + i + 1
                
                if wandb.run:
                    for key, vals in metrics_dict.items():
                        tag = f"train_{key}"
                        wandb.log({tag: vals}, current_step)
                
                if summary_writer:
                    for key, vals in metrics_dict.items():
                        tag = f"train_{key}"
                        summary_writer.scalar(tag, vals, current_step)
            
            if wandb.run:
                wandb.log({"train_time": train_time}, step)
            
            if summary_writer:
                summary_writer.scalar("train_time", train_time, step)
        else:
            try:
                train_metrics = alpa.util.get_metrics(train_metrics)
            except Exception as _:
                pass
            
            if wandb.run:
                for key, vals in train_metrics.items():
                    tag = f"train_{key}"
                    if hasattr(vals, "__iter__") and not isinstance(vals, (str, bytes)):
                        try:
                            vals_list = list(vals)
                            for i, val in enumerate(vals_list):
                                wandb.log({tag: val}, step - len(vals_list) + i + 1)
                        except TypeError:
                            wandb.log({tag: vals}, step)
                    else:
                        wandb.log({tag: vals}, step)
                wandb.log({"train_time": train_time}, step)
            
            if summary_writer:
                summary_writer.scalar("train_time", train_time, step)
                for key, vals in train_metrics.items():
                    tag = f"train_{key}"
                    if hasattr(vals, "__iter__") and not isinstance(vals, (str, bytes)):
                        try:
                            vals_list = list(vals)
                            for i, val in enumerate(vals_list):
                                summary_writer.scalar(tag, val, step - len(vals_list) + i + 1)
                        except TypeError:
                            summary_writer.scalar(tag, vals, step)
                    else:
                        summary_writer.scalar(tag, vals, step)

def write_eval_metric(eval_metrics, step, summary_writer=None):
    import jax
    if jax.process_index() == 0:
        import wandb
        import alpa
        
        # Handle list of dictionaries case
        if isinstance(eval_metrics, list) and all(isinstance(item, dict) for item in eval_metrics):
            for i, metrics_dict in enumerate(eval_metrics):
                try:
                    metrics_dict = alpa.util.get_metrics(metrics_dict)
                except Exception as _:
                    pass
                
                current_step = step - len(eval_metrics) + i + 1
                
                if wandb.run:
                    for key, vals in metrics_dict.items():
                        tag = f"eval_{key}"
                        wandb.log({tag: vals}, current_step)
                
                if summary_writer:
                    for key, vals in metrics_dict.items():
                        tag = f"eval_{key}"
                        summary_writer.scalar(tag, vals, current_step)
        else:
            # Original case: single dictionary
            try:
                eval_metrics = alpa.util.get_metrics(eval_metrics)
            except Exception as _:
                pass
            
            if wandb.run:
                for key, vals in eval_metrics.items():
                    tag = f"eval_{key}"
                    if hasattr(vals, "__iter__") and not isinstance(vals, (str, bytes)):
                        try:
                            vals_list = list(vals)
                            for i, val in enumerate(vals_list):
                                wandb.log({tag: val}, step - len(vals_list) + i + 1)
                        except TypeError:
                            wandb.log({tag: vals}, step)
                    else:
                        wandb.log({tag: vals}, step)
            
            if summary_writer:
                for key, vals in eval_metrics.items():
                    tag = f"eval_{key}"
                    if hasattr(vals, "__iter__") and not isinstance(vals, (str, bytes)):
                        try:
                            vals_list = list(vals)
                            for i, val in enumerate(vals_list):
                                summary_writer.scalar(tag, val, step - len(vals_list) + i + 1)
                        except TypeError:
                            summary_writer.scalar(tag, vals, step)
                    else:
                        summary_writer.scalar(tag, vals, step)

def write_metric(train_metrics, eval_metrics, train_time, step, summary_writer=None):
    write_train_metric(
        train_metrics=train_metrics, 
        train_time=train_time, 
        step=step, 
        summary_writer=summary_writer
    )

    write_eval_metric(
        eval_metrics=eval_metrics, 
        step=step, 
        summary_writer=summary_writer
    )

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

# utils
def mb_item(x):
    return x.item() if hasattr(x, "item") else x

def make_batch(samples):
    batch = {k : jnp.array(v) for k,v in samples.items()}
    return batch

def decay_mask_fn(params, flatten_layers):
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {
        path: (path[-1] != "bias" and path[-2:] not in flatten_layers)
        for path in flat_params
    }
    return traverse_util.unflatten_dict(flat_mask)

def maybe_stop_wandb():
    import wandb

    try:
        if wandb.run:
            wandb.finish()
    except Exception as _:
        pass

def login_wandb(
    environ_name: str = "WANDB_API_KEY", token: Optional[str] = None, **kwargs
):
    import wandb

    if token is None:
        token = os.getenv(environ_name)
        print(f"Use token from environment variable {environ_name}")
    wandb.login(key=token, **kwargs)

def instantiate_wandb(
    run_name: Optional[str] = None, 
    config: Union[Dict, None] = None,
    project: str = "copus"
):
    import wandb
    wandb.init(project=project, entity=None, name=run_name, config=config)

def setup_logging(name, level: str="DEBUG"):
    import datasets
    import transformers

    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=getattr(logging, level)
    )

    logger.setLevel(logging.INFO)

    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger

@jax.tree_util.register_pytree_node_class
class _ParamsPlaceholder:
    def tree_flatten(self):
        return ((), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux, children
        return cls()

@typing.runtime_checkable
class Initable(Protocol):
    """An object with an init function."""

    def init(self, params):
        """Calling the init for given parameters returns a fresh opt state."""

def tree_map_params(
    initable,
    f,
    state,
    /,
    *rest,
    transform_non_params = None,
    is_leaf = None
):
    """Apply a callable over all params in the given optimizer state.

    This function exists to help construct partition specs over optimizer
    states, in the case that a partition spec is already known for the parameters.

    For example, the following will replace all optimizer state parameter trees
    with copies of the given partition spec instead. The argument
    `transform_non_params` can be used to replace any remaining fields as
    required, in this case, we replace those fields by None.

    >>> params, specs = jnp.array(0.), jnp.array(0.)  # Trees with the same shape
    >>> opt = optax.sgd(1e-3)
    >>> state = opt.init(params)
    >>> opt_specs = optax.tree_map_params(
    ...     opt,
    ...     lambda _, spec: spec,
    ...     state,
    ...     specs,
    ...     transform_non_params=lambda _: None,
    ...     )

    Args:
    initable: A callable taking parameters and returning an optimizer state, or
        an object with an `init` attribute having the same function.
    f: A callable that will be applied for all copies of the parameter tree
        within this optimizer state.
    state: The optimizer state to map over.
    *rest: Additional arguments, having the same shape as the parameter tree,
        that will be passed to f.
    transform_non_params: An optional function that will be called on all
        non-parameter fields within the optimizer state.
    is_leaf: Passed through to `jax.tree.map`. This makes it possible to ignore
        parts of the parameter tree e.g. when the gradient transformations modify
        the shape of the original pytree, such as for ``optax.masked``.

    Returns:
    The result of applying the function f on all trees in the optimizer's state
    that have the same shape as the parameter tree, along with the given
    optional extra arguments.
    """
    placeholder = cast(base.chex.ArrayTree, _ParamsPlaceholder())
     
    if isinstance(initable, Initable):
        initable = cast(Initable, initable)
        state_with_placeholders = initable.init(placeholder)
    else:
        state_with_placeholders = initable(placeholder)

    def map_params(maybe_placeholder_value, value):
        if isinstance(maybe_placeholder_value, _ParamsPlaceholder):
            return jax.tree_util.tree_map(f, value, *rest, is_leaf=is_leaf)
        elif transform_non_params is not None:
            return transform_non_params(value)
        else:
            return value

    return jax.tree_util.tree_map(
        map_params,
        state_with_placeholders,
        state,
        is_leaf=lambda v: isinstance(v, _ParamsPlaceholder)
    )

def count_params(model):
    return sum(x.size for x in jax.tree_leaves(model))

# Mainly for CausalLM model
def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int,
                min_batch_size: int, shuffle: bool = False):
    """
    Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
    Shuffle batches if `shuffle` is `True`.
    """
    if len(dataset) < batch_size:
        assert len(dataset) >= min_batch_size
        batch_size = len(dataset) // min_batch_size * min_batch_size

    data_collator = transformers.DefaultDataCollator("np")
    tf_dataset = dataset.to_tf_dataset(batch_size=batch_size,
                                       columns=dataset.column_names,
                                       collate_fn=data_collator,
                                       shuffle=shuffle,
                                       drop_remainder=True)

    for batch in tf_dataset:
        batch = {k: v._numpy() for k, v in batch.items()}
        yield batch

def reset_alpa_state():
    alpa.adaptdl.epoch._EPOCH_STATE = None # Reset the internal state
    alpa.adaptdl.checkpoint._STATES_TO_NAMES = {} # Reset the checkpoint state
    alpa.adaptdl.checkpoint._NAMES_TO_STATES = {} # Reset the checkpoint state

def monkeypatch_rope_gemma():
    exec("from transformers.models.gemma import modeling_flax_gemma", globals())
    source = inspect.getsource(modeling_flax_gemma.FlaxGemmaRotaryEmbedding.__call__)
    start = source.find("def")
    source = source.split("\n")
    source = "\n".join([x[start:] for x in source])
    source = source.replace("key = apply", "# key = apply")
    source = source.replace("query = apply", "# query = apply")
    func = create_dynamic_function(source, "__call__")
    modeling_flax_gemma.FlaxGemmaRotaryEmbedding.__call__ = func

def get_profiling_setup(
    profiling_enabled: bool, 
    profiling_config: Dict[str, Union[int, bool]],
    yml_config: Dict[str, str], 
    training_args
):
    # Profiling config
    num_micro_batches = training_args.num_micro_batches
    num_devices = alpa.get_global_num_devices()

    batch_sizes_to_run = []
    if profiling_enabled:
        # Disable wandb
        os.environ["WANDB_MODE"] = "offline"

        # Disable preallocation of JAX
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        # batch sizes
        min_batch_size = profiling_config.get("min_batch_size", 1)
        max_batch_size = profiling_config.get("max_batch_size", None)

        if min_batch_size <= 0:
            logger.warning("Minimum batch size should be greater than 0. Setting it to 1.")
            min_batch_size = 1

        current_bs = 1
        while current_bs < min_batch_size:
            current_bs *= 2

        logger.info(f"Starting search for profile batch sizes from {current_bs}")
        logger.info(f"Constraints: Divisible by num_devices ({num_devices}), divisible by num_micro_batches ({num_micro_batches}), max_total_bs ({max_batch_size})")

        while True:
            # 1. Check max batch size limit
            if max_batch_size is not None and current_bs > max_batch_size:
                logger.info(f"Current batch size {current_bs} exceeds max {max_batch_size}. Stopping search.")
                break

            # 2. Check divisibility by number of devices
            if current_bs % num_devices != 0:
                logger.debug(f"Skipping batch size {current_bs}: Not divisible by num_devices ({num_devices})")
                current_bs *= 2
                continue

            # 3. Check divisibility by num_micro_batches
            # The total batch size per step must be divisible by num_micro_batches
            # for gradient accumulation logic.
            if num_micro_batches > 0 and current_bs % num_micro_batches != 0:
                 logger.debug(f"Skipping batch size {current_bs}: Not divisible by num_micro_batches ({num_micro_batches})")
                 current_bs *= 2
                 continue
            elif num_micro_batches <= 0:
                 logger.warning("num_micro_batches is <= 0. Skipping divisibility check.")

            # If all checks pass, add it to the list
            logger.info(f"Found valid profile batch size: {current_bs}")
            batch_sizes_to_run.append(current_bs)

            # Move to the next power of 2
            current_bs *= 2

            # Safety break for extremely large numbers if max_total_bs is None
            if current_bs > 2 ** 20: # Arbitrary large limit (~1 million)
                logger.warning("Reached very large batch size during profiling search without max_total_bs. Stopping.")
                break

        if not batch_sizes_to_run:
            logger.error("No valid batch sizes found for profiling based on the constraints!")
            return # Exit if no batch sizes are viable
    else:
        initial_local_batch_size = yml_config.dataloader.train.init_local_batch_size
        initial_total_batch_size = initial_local_batch_size * num_devices
        batch_sizes_to_run = [initial_total_batch_size]
        logger.info(f"Using initial batch size: {initial_total_batch_size} for profiling.")

    return batch_sizes_to_run

def run_profile(
    p_train_step: Callable,
    state: Any,
    batch: Any,
    variables_dict: Dict[str, Any],
    # CSV
    i_run: int,
    epoch: int,
    current_total_batch_size: int,
    yml_config: Dict[str, str],
):
    print("Running profiling...")
    executable = p_train_step.get_executable(state, batch, variables_dict)
    executable.sync()

    # warmup
    for _ in range(yml_config.profiling.warmup_steps):
        cost_dummy = executable.profile_with_dummy_inputs()

    avg_cost = []
    for _ in range(yml_config.profiling.profile_steps):
        cost_dummy = executable.profile_with_dummy_inputs()
        avg_cost.append(cost_dummy)
    avg_cost = np.mean(np.array(avg_cost))
    mem_gb = executable.get_total_allocation_size() / (1024**3)
    logger.info(f"Average cost: {avg_cost}")
    logger.info(f"Memory usage: {mem_gb} GB")

    if jax.process_index() == 0:
        with open(yml_config.profiling.csv_path, "a") as f:
            if os.stat(yml_config.profiling.csv_path).st_size == 0:
                f.write("run,epoch,batch_size,avg_cost,mem_gb,model_name,strategy,dp,pp,tp\n")
            f.write(f"{i_run},{epoch},{current_total_batch_size},{avg_cost},{mem_gb},"
                    f"{yml_config.model_name_or_path},{yml_config.training.parallel_method.method},"
                    f"{yml_config.training.parallel_method.parameters._3D.data_parallel},{yml_config.training.parallel_method.parameters._3D.operator_parallel},"
                    f"{yml_config.training.parallel_method.parameters._3D.operator_parallel}\n")

    if avg_cost == float("inf"):
        sys.exit(1)

    for _ in range(10):
        import gc
        gc.collect()

logger = setup_logging(__name__)