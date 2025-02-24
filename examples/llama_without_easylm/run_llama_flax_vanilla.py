import sys

sys.path.append(".")

from examples.utils import *  # Assuming this contains necessary utility functions

import json
import math
import os
import time
from dataclasses import dataclass
from functools import partial
import functools
from itertools import chain
from pathlib import Path
from jax.experimental.pjit import PartitionSpec

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Removed alpa import
# import alpa
# from alpa.model.model_util import DynamicScale, TrainState  # Moved TrainState import to flax.training
from flax.training.dynamic_scale import DynamicScale
from flax.training import train_state

# from alpa import ManualShardingOption  # Removed manual sharding
import jax
import jax.numpy as jnp
import optax
import transformers
from flax import traverse_util
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    HfArgumentParser,
    is_tensorboard_available,
    set_seed,
)
from transformers.models.llama.modeling_flax_llama import FlaxLlamaForCausalLMModule
from transformers.testing_utils import CaptureLogger
from transformers.utils import get_full_repo_name, send_example_telemetry
from flax.training import dynamic_scale as dynamic_scale_lib

class TrainState(train_state.TrainState):
  dynamic_scale: dynamic_scale_lib.DynamicScale

IGNORE_TOKEN_ID = -100

logger = setup_logging(__name__)  # Assuming setup_logging is defined in examples.utils

disable_log = True
if disable_log:
    import os
    os.environ["WANDB_DISABLED"] = "true"

# Removed alpa initialization
# init_alpa("ray")


# Removed monkey patching (not necessary without alpa)
# def do_monkey_patch():
#     # TODO: jax 0.3.22 does not support eval shape with static args well. Remove
#     # after rebasing to jax 0.4, use the model's _do_init=False then.
#     def init_dummy(self, *args, **kwargs):
#         avals = jax.eval_shape(partial(self._backup_init, **kwargs), *args)
#         return jax.tree_util.tree_map(lambda x: jnp.full(x.shape, 1e-8, x.dtype),
#                                     avals)
#     if not hasattr(FlaxLlamaForCausalLMModule, "_backup_init"):
#         FlaxLlamaForCausalLMModule._backup_init = FlaxLlamaForCausalLMModule.init
#     FlaxLlamaForCausalLMModule.init = init_dummy


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



# Removed manual sharding function (not needed for pure Flax)
# def llama_manual_sharding(num_layers, state: TrainState):
#     ...


@dataclass
class ModelArguments(ModelArguments):  # Assuming ModelArguments is defined elsewhere
    pass

@dataclass
class DataTrainingArguments(DataTrainingArguments): # Assuming DataTrainingArguments is defined elsewhere

    def __post_init__(self):
        if self.use_data_sample:  # Assuming use_data_sample exists
            self.dataset_name = "Erland/oscar_sampled_1000"
            self.dataset_config_name = "default"
            delattr(self, "use_data_sample")

        super().__post_init__()



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)) # Assuming TrainingArguments is defined elsewhere
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args, framework="flax")

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

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
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

    #  Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if "validation" not in dataset.keys():
            dataset["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            dataset["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            **dataset_args,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if "validation" not in dataset.keys():
            dataset["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            dataset["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # if model_args.config_name:
    #     config = AutoConfig.from_pretrained(
    #         model_args.config_name,
    #         cache_dir=model_args.cache_dir,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #     )
    # elif model_args.model_name_or_path:
    #     config = AutoConfig.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=model_args.cache_dir,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #     )
    # else:
    #     config = CONFIG_MAPPING[model_args.model_type]()
    #     logger.warning("You are instantiating a new config instance from scratch.")

    from transformers import LlamaConfig
    config = LlamaConfig(
        vocab_size=4096,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=8,
    )

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.config_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            #use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
            use_fast=False,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # do_monkey_patch()  # Removed

    if model_args.model_name_or_path:
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            use_auth_token=True if model_args.use_auth_token else None,
            from_pt=False
        )
    else:
        model = FlaxAutoModelForCausalLM.from_config(
            config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    else:
        column_names = dataset["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    logger.info("***** Tokenize dataset *****")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    logger.info("***** Build dataset *****")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))


    # Removed device count and batch size calculations based on alpa
    num_devices = jax.local_device_count()

    train_batch_size = int(training_args.per_device_train_batch_size) * num_devices
    eval_batch_size = int(training_args.per_device_eval_batch_size) * num_devices


    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard:
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

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    # train_batch_size = int(training_args.per_device_train_batch_size) * num_devices # Already calculated
    # eval_batch_size = int(training_args.per_device_eval_batch_size) * num_devices   # Already calculated
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(  # Assuming this is in examples.utils
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxGPT2.
    # For other models, one should correct the layer norm parameter naming
    # accordingly.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in [("input_layernorm", "weight"), ("post_attention_layernorm", "weight")])
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        if training_args.weight_decay == 0.0:
            decay_mask_fn = None
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=linear_decay_lr_schedule_fn,
                b1=training_args.adam_beta1,
                b2=training_args.adam_beta2,
                eps=training_args.adam_epsilon,
                weight_decay=training_args.weight_decay,
                mask=decay_mask_fn
            )
        )
    if training_args.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(optimizer, training_args.gradient_accumulation_steps)
    grad_accum_steps = training_args.gradient_accumulation_steps

    # Setup train state
    if model_args.dtype == "float16":
        use_master_copy = True
        dynamic_scale = DynamicScale()
        # Fix a bug in huggingface's implementation (https://github.com/huggingface/transformers/pull/18462)
        # alpa.global_config.flax_always_use_fp16_embedding = True  # Removed alpa
    else:
        use_master_copy = dynamic_scale = None

    state = TrainState.create(
        apply_fn=model.__call__, 
        params=model.params, 
        tx=optimizer, 
        dynamic_scale=dynamic_scale,
        # use_master_copy=use_master_copy # This is deprecated and should be replaced by param_dtype
        # param_dtype = jnp.float32 if use_master_copy else getattr(jnp, model_args.dtype)
    )

    dump_debug_info_train_step = dump_debug_info_eval_step = True # Not used now, but left in case of future manual debugging

    # Removed Manual partition spec
    # state_manual_sharding = llama_manual_sharding(config.num_hidden_layers, state)
    # ms_option = ManualShardingOption(
    #     ("dp", "mp"), in_axis_resources=(state_manual_sharding, PartitionSpec("dp", None)))
    ignore_ids = (IGNORE_TOKEN_ID, )

    def loss_fn(logits, labels, ignore_indices):
        # Shift logits
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        # Handle the ignore index: compute the valid first
        valid = jnp.full(shift_labels.shape, True)
        for ignore_index in ignore_indices:
            new_valid = jnp.not_equal(shift_labels, ignore_index)
            valid = jnp.logical_and(valid, new_valid)
        valid = jnp.asarray(valid, dtype=jnp.float32)
        valid_len = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
        # OneHot and mask the ignore index. For ignore_index(-100), the whole line
        # in the output would be 0.
        one_hot_labels = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        # Compute the softmax loss
        log_p = jax.nn.log_softmax(shift_logits, axis=-1)
        # (bs, seq_len, vocab) -> (bs, seq_len)
        cross_entropy = jnp.sum(one_hot_labels * log_p, axis=-1)
        loss = -jnp.mean(jnp.sum(cross_entropy * valid, axis=-1) / valid_len)
        return loss

    # Define gradient update step fn
    @jax.jit
    def train_step(state, batch, dropout_rng):

        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels, ignore_ids)
            return loss

        dynamic_scale = state.dynamic_scale
        if dynamic_scale:
            grad_fn = dynamic_scale.value_and_grad(compute_loss)
            dynamic_scale, is_fin, loss, grads = grad_fn(state.params)
            # grads = jax.lax.pmean(grads, axis_name="batch") # Removed pmean for replication across devices
            # loss = jax.lax.pmean(loss, axis_name="batch") # Removed pmean
            # is_fin = jax.lax.pmin(is_fin, axis_name="batch") # Removed pmin
            grads = jax.tree_map(lambda x: jnp.nan_to_num(x), grads)

            new_state = state.apply_gradients(grads=grads)
            
            new_state = new_state.replace(
                opt_state=jax.tree_map(
                    functools.partial(jnp.where, is_fin),
                    new_state.opt_state, state.opt_state),
                params=jax.tree_map(
                    functools.partial(jnp.where, is_fin),
                    new_state.params, state.params),
                # master_copy=jax.tree_map( # Removed master copy, use param_dtype instead
                #     functools.partial(jnp.where, is_fin),
                #     new_state.master_copy, state.master_copy),
                dynamic_scale=dynamic_scale)


        else:
            grad_fn = jax.value_and_grad(compute_loss)
            loss, grads = grad_fn(state.params)
            # grads = jax.lax.pmean(grads, axis_name="batch") # Removed pmean
            # loss = jax.lax.pmean(loss, axis_name="batch")
            grads = jax.tree_map(lambda x: jnp.nan_to_num(x), grads)
            new_state = state.apply_gradients(grads=grads)


        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}

        return new_state, metrics, new_dropout_rng

    @jax.jit
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]  # Set train=False for evaluation
        loss = loss_fn(logits, labels, ignore_ids)
        metrics = {"loss": loss}
        return metrics


    # Removed parallelization using alpa
    # method = create_alpa_method(
    #     AlpaMethod(training_args.parallel_strategy),
    #     training_args,
    # )

    # p_train_step = alpa.parallelize(
    #     train_step,
    #     method=method,
    #     donate_argnums=(0,)
    # )


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Batch size per device (w. accumulation) = {training_args.per_device_train_batch_size}")
    logger.info(f"  Global train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    train_metrics = []
    epochs = tqdm(range(num_epochs), desc="Epoch ... ", position=0)

    step_ct = 0
    last_time = time.time()

    epochs.write("Initial compilation. This might take some minutes...")

    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        train_loader = data_loader(input_rng, train_dataset, train_batch_size,
                                   0, shuffle=True) #removed min_batch_size
        steps_per_epoch = len(train_dataset) // train_batch_size

        steps_trained_progress_bar = tqdm(
            range(steps_per_epoch),
            desc="Training...",
            position=1,
            leave=False
        )
        for step in range(steps_per_epoch):
            cur_step = epoch * (len(train_dataset) // train_batch_size) + step

            batch = next(train_loader)
            # print("Input IDs shape (inside train_step):", batch["input_ids"].shape) # Removed for cleaner output
            batch["position_ids"] = (batch["attention_mask"].cumsum(axis=1) *
                                     batch["attention_mask"]) - 1


            # Replicate batch across devices
            batch = jax.tree_map(lambda x: jnp.array(x), batch)
            #removed shape check because of removed pmap/alpa


            state, train_metric, dropout_rng = train_step(state, batch, dropout_rng)
            train_metrics.append(train_metric)


            if step % grad_accum_steps == 0:
                steps_trained_progress_bar.update(1)

            # Removed debug info dumping (no alpa executables)
            # if dump_debug_info_train_step:
            #     dump_debug_info_train_step = False
            #     executable = p_train_step.get_last_executable()
            #     executable.sync()
            #     executable.dump_debug_info("alpa_debug_info")
            #     epochs.write(f"Initial compilation completed. "
            #                  f"Time elapsed: {time.time() - train_start:.2f} s")

            step_ct += 1
            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Removed syncing for alpa executables
                # executable.sync()
                latency = (time.time() - last_time) / step_ct

                # Removed throughput calculations - these require alpa for accurate FLOP counting
                # throughput_tokens = np.prod(batch["input_ids"].shape) / latency
                # throughput_tflops = alpa.util.compute_gpt_tflops(
                #     batch_size=batch["input_ids"].shape[0],
                #     seq_len=batch["input_ids"].shape[1],
                #     num_layers=config.num_hidden_layers,
                #     hidden_size=config.hidden_size,
                #     vocab_size=config.vocab_size,
                #     num_gpus=alpa.get_global_num_devices(),
                #     latency=latency)
                step_ct = 0


                # Save metrics
                train_time += time.time() - train_start
                if has_tensorboard:
                    # Assuming write_train_metric is defined in examples.utils
                    # write_train_metric(train_metrics, train_time, cur_step, summary_writer)
                    pass


                train_metric = jax.tree_map(np.mean, train_metric)

                epochs.write(
                    f"Step... {cur_step} | "
                    f"Loss: {train_metric['loss'].mean():.4f}, "
                    f"Learning Rate: {train_metric['learning_rate'].mean():.5f}, "
                    # Removed Throughput
                )

                train_metrics = []
                last_time = time.time()

            if training_args.do_eval:
                # Only support steps
                if ( cur_step % (training_args.eval_steps * grad_accum_steps) == 0 and
                    cur_step > 0 and
                    model_args.eval_strategy == "steps"):
                    # ======================== Evaluating ==============================
                    eval_metrics = []
                    eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size,
                                            0) #Removed min_batch_size
                    eval_steps = max(len(eval_dataset) // eval_batch_size, 1)
                    for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
                        # Model forward
                        batch = next(eval_loader)
                        batch["position_ids"] = (batch["attention_mask"].cumsum(axis=1) *
                                                batch["attention_mask"]) - 1

                        # Replicate batch for evaluation
                        batch = jax.tree_map(lambda x: jnp.array(x), batch)

                        metrics = eval_step(state.params, batch)
                        eval_metrics.append(metrics)


                    # normalize eval metrics
                    # eval_metrics = alpa.util.get_metrics(eval_metrics) # Removed alpa utility
                    eval_metrics = jax.tree_util.tree_map(jnp.mean, jax.tree_util.tree_util.tree_flatten(eval_metrics)[0]) #flatten and mean the list of dicts
                    try:
                        eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
                    except OverflowError:
                        eval_metrics["perplexity"] = float("inf")
                                        # Print metrics and update progress bar
                desc = (
                    f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']} | Eval Perplexity:"
                    f" {eval_metrics['perplexity']})"
                )
                epochs.write(desc)

                # Save metrics
                if has_tensorboard:
                    write_eval_metric(summary_writer, eval_metrics, cur_step) # Assuming write_eval_metric is in examples.utils

        if cur_step % training_args.save_steps == 0 and cur_step > 0:
            # save checkpoint after each epoch and push checkpoint to the hub
            epochs.write("\nSave checkpoint...")
            # Removed alpa.prefetch
            # alpa.prefetch(state.params)
            # params = alpa.util.map_to_nparray(state.params)  # Removed alpa utility, use standard Flax serialization
            params = jax.device_get(state.params)

            model.save_pretrained(training_args.output_dir, params=params)
            tokenizer.save_pretrained(training_args.output_dir)
            if training_args.push_to_hub:
                repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)

    # Eval after training
    if training_args.do_eval:
        eval_metrics = []
        eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size,
                                0) #removed min_batch_size
        eval_steps = max(len(eval_dataset) // eval_batch_size, 1)
        for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
            # Model forward
            batch = next(eval_loader)
            batch["position_ids"] = (batch["attention_mask"].cumsum(axis=1) *
                                    batch["attention_mask"]) - 1

            # Replicate batch for eval
            batch = jax.tree_map(lambda x: jnp.array(x), batch)

            metrics = eval_step(state.params, batch)
            eval_metrics.append(metrics)

        # normalize eval metrics
        # eval_metrics = alpa.util.get_metrics(eval_metrics) # Removed alpa utility
        eval_metrics = jax.tree_util.tree_map(jnp.mean, jax.tree_util.tree_util.tree_flatten(eval_metrics)[0]) #flatten and mean

        try:
            eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
        except OverflowError:
            eval_metrics["perplexity"] = float("inf")

        eval_metrics = {f"eval_{metric_name}": value for metric_name, value in eval_metrics.items()}
        path = os.path.join(training_args.output_dir, "eval_results.json")
        with open(path, "w") as f:
            json.dump(eval_metrics, f, indent=4, sort_keys=True)

    # Save the final model
    epochs.write("\nSave the final model...")
    # Removed alpa prefetch and map_to_nparray
    # alpa.prefetch(state.params)
    # params = alpa.util.map_to_nparray(state.params)
    params = jax.device_get(state.params)
    model.save_pretrained(training_args.output_dir, params=params)
    tokenizer.save_pretrained(training_args.output_dir)

    # Removed alpa shutdown
    # alpa.shutdown()

if __name__ == "__main__":
    main()