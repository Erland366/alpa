import sys

sys.path.append(".")

from examples.utils import *

import alpa
import logging
import sys
import time
import functools
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import numpy as np

import datasets
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from typing import Optional

import torch
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import (
    # added for image augmentation
    ToPILImage,
    RandomCrop,
    ColorJitter,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    ToTensor,
    # /added for image augmentation
    CenterCrop,
    ConvertImageDtype,
    Normalize,
    Resize,
)
from torchvision.transforms.functional import InterpolationMode
from alpa.model.model_util import DynamicScale, TrainState

import jax
import jax.profiler
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax.serialization import to_bytes, from_bytes
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    FlaxCLIPModel,
    CLIPProcessor,
    HfArgumentParser,
    is_tensorboard_available,
    IntervalStrategy,
    set_seed,
)

init_alpa(normalize_embedding_shape=False)

logger = setup_logging(__name__)

disable_log = True
if disable_log:
    import os
    os.environ["WANDB_DISABLED"] = "true"


# Modify the Transform class to ensure it's compatible with the model's vision config
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        logger.info(f"Creating image transformer with size: {image_size}, mean: {mean}, std: {std}")
        
        # Ensure image_size is an integer
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]
        
        self.transforms = torch.nn.Sequential(
            Resize(
                [image_size], interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image` or a tensor"""
        with torch.no_grad():
            # Handle different input types
            if not isinstance(x, torch.Tensor):
                # Convert PIL image or numpy array to tensor if needed
                if not isinstance(x, (torch.Tensor, np.ndarray)):
                    x = torch.from_numpy(np.array(x))
                else:
                    x = torch.from_numpy(x)
                    
                # Ensure the tensor is in the right format (CHW)
                if x.ndim == 3 and x.shape[0] not in (1, 3):
                    # Assume HWC format, convert to CHW
                    x = x.permute(2, 0, 1)
                
            # Apply the transforms
            x = self.transforms(x)
            
        return x


def rotate_checkpoints(ckpt_dir: str, save_total_limit: int):
    "Removes older checkpoints so that `save_total_limit` checkpoints are kept"
    # TODO: what to remove is decided using step number only, we might want to improve that
    ckpts = [str(x) for x in Path(ckpt_dir).glob("ckpt-*")]
    # sort checkpoints by step
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
    ckpts_to_delete = ckpts_sorted[:-save_total_limit]
    for ckpt in ckpts_to_delete:
        logger.info(
            f"Deleting older checkpoint [{ckpt}] due to save_total_limit ({save_total_limit})"
        )
        shutil.rmtree(ckpt)


@dataclass
class ModelArguments(ModelArguments):
    processor_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the processor to use to convert the data in features for the model."
        },
    )

    def __post_init__(self):
        if self.tokenizer_name is not None and self.processor_name is None:
            logger.warning(
                "You have provided a `tokenizer_name` but not a `processor_name`. The `tokenizer_name` will be used to determine the processor."
            )
            self.processor_name = self.tokenizer_name
        if self.processor_name is not None and self.tokenizer_name is not None:
            logger.warning(
                "Both `processor_name` and `tokenizer_name` are provided. `processor_name` will be used."
            )
            delattr(self, "tokenizer_name")


@dataclass
class DataTrainingArguments(DataTrainingArguments):
    def __post_init__(self):
        if self.use_data_sample:
            self.dataset_name = "RIW/small-coco"
            self.dataset_config_name = None
            delattr(self, "use_data_sample")

        super().__post_init__()


@dataclass
class TrainingArguments(TrainingArguments):
    project: str = field(default="alpa_clip")


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            ImageAugmentationArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, augmentation_args = (
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        )
    else:
        model_args, data_args, training_args, augmentation_args = (
            parser.parse_args_into_dataclasses()
        )

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

    # login_wandb()
    # instantiate_wandb()
    set_seed(training_args.seed)

    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")


    # Determine the source for the processor
    if model_args.processor_name:
        processor = CLIPProcessor.from_pretrained(
            model_args.processor_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=model_args.token if model_args.token else None,
        )
    elif model_args.model_name_or_path:
        processor = CLIPProcessor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=model_args.token if model_args.token else None,
        )
    elif model_args.config_name:
        processor = CLIPProcessor.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=model_args.token if model_args.token else None,
        )
    else:
        raise ValueError("No processor source provided")

    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    if model_args.model_name_or_path:
        model = FlaxCLIPModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
        )
    elif model_args.config_name:
        model = FlaxCLIPModel.from_pretrained(
            model_args.config_name,
            config=config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
        )
    else:
        model = FlaxCLIPModel(
            config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )


    config = model.config
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    dataset_name_mapping = {
        "image_caption_dataset.py": ("image_path", "caption"),
    }

    dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)
    if data_args.image_column is None:
        image_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.caption_column is None:
        caption_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    image_transformations = Transform(
        config.vision_config.image_size,
        image_processor.image_mean,
        image_processor.image_std,
    )
    image_transformations = torch.jit.script(image_transformations)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = torch.tensor(
            [example["input_ids"] for example in examples], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [example["attention_mask"] for example in examples], dtype=torch.long
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        max_length = config.text_config.max_position_embeddings if hasattr(config, 'text_config') else data_args.max_seq_length
    
        text_inputs = tokenizer(
            captions,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def transform_images(examples):
        if isinstance(examples[image_column][0], Image.Image):
            images = [
                torch.from_numpy(np.array(image.convert("RGB")).transpose(2, 0, 1))
                for image in examples[image_column]
            ]
            # The transpose is needed to get CHW format (channels, height, width) that PyTorch expects
        elif isinstance(examples[image_column][0], str):  # Then it's a path
            images = [
                read_image(image_file, mode=ImageReadMode.RGB)
                for image_file in examples[image_column]
            ]
        else:
            raise ValueError(
                f"{examples[image_column][0]} is not either a `PIL.Image` or a path!"
            )

        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples

    def filter_corrupt_images(examples):
        """remove problematic images"""
        from PIL import Image

        valid_images = []
        for image_file in examples[image_column]:
            try:
                if not isinstance(image_file, Image.Image):
                    Image.open(image_file)
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")

        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset = train_dataset.filter(
            filter_corrupt_images,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        train_dataset.set_transform(transform_images)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        eval_dataset = eval_dataset.filter(
            filter_corrupt_images,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
        eval_dataset = eval_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        eval_dataset.set_transform(transform_images)

    num_devices = alpa.get_global_num_devices()
    train_min_batch_size = (num_devices // training_args.operator_parallel //
                            training_args.pipeline_parallel * training_args.num_micro_batches)
                        
    if training_args.do_eval:
        eval_num_micro_batches = training_args.num_micro_batches
        eval_min_batch_size = (num_devices // training_args.operator_parallel //
                            training_args.pipeline_parallel * eval_num_micro_batches)
        while len(eval_dataset) < eval_min_batch_size:
            eval_num_micro_batches //= 2
            eval_min_batch_size = (num_devices // training_args.operator_parallel //
                                training_args.pipeline_parallel * eval_num_micro_batches)

    HAS_TENSORBOARD = is_tensorboard_available()
    if HAS_TENSORBOARD and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            HAS_TENSORBOARD = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
            summary_writer = None
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * alpa.get_global_num_devices()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * alpa.get_global_num_devices()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    if training_args.do_eval:
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=data_args.preprocessing_num_workers,
            persistent_workers=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    if (
        jax.process_index() == 0 and HAS_WANDB and not DEBUG
        # and ("wandb" in training_args.report_to) # Still error
    ):
        try:
            import wandb

            if training_args.run_name is None:
                run_name = training_args.output_dir.split("/")[-1]
            else:
                run_name = training_args.run_name
            wandb.init(
                name=run_name,
                entity=training_args.entity,
                project=training_args.project,
                sync_tensorboard=True,
            )
            wandb.config.update(training_args)
            wandb.config.update(model_args)
            wandb.config.update(data_args)
        except ImportError as e:
            print(e)
            has_wandb = False

    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    flatten_layers_masked = [("ln_1", "scale"), ("ln_2", "scale"), ("ln_f", "scale")]

    # create optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=functools.partial(decay_mask_fn, flatten_layers=flatten_layers_masked),
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(training_args.max_grad_norm), optimizer
        )
    if training_args.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(
            optimizer, training_args.gradient_accumulation_steps
        )
    grad_accum_steps = training_args.gradient_accumulation_steps

    if model_args.dtype == "float16":
        use_master_copy = True
        dynamic_scale = DynamicScale()
        alpa.global_config.flax_always_use_fp16_embedding = True
    else:
        use_master_copy = dynamic_scale = None

    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dynamic_scale=dynamic_scale,
        use_master_copy=use_master_copy,
    )

    dump_debug_info_train_step = dump_debug_info_eval_step = True

    if training_args.resume_from_checkpoint:
        state = restore_checkpoint(training_args.resume_from_checkpoint, state)
        resume_step = mb_item(state.step)
    else:
        resume_step = 0

    # Check whether this is correct
    # def cross_entropy(logits, axis):
    #     """copied from https://github.com/huggingface/transformers/blob/main/examples/flax/text-retrieval/run_clip.py#L488"""
    #     labels = jnp.arange(logits.shape[0])
    #     loss = optax.softmax_cross_entropy_with_integer_labels(
    #         logits=logits, labels=labels
    #     ).mean()
    #     return loss

    # def clip_loss(logits_per_image, logits_per_text):
    #     """modified clip loss to accept both logits"""
    #     loss_img = cross_entropy(logits_per_image, axis=0)
    #     loss_txt = cross_entropy(logits_per_text, axis=1)
    #     loss = (loss_img + loss_txt) / 2
    #     return loss

    def clip_loss(logits_per_image, logits_per_text):
        batch_size = logits_per_image.shape[0]
        labels = jnp.arange(batch_size)
        
        loss_i2t = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits_per_image, 
            labels=labels
        )
        
        loss_t2i = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits_per_text, 
            labels=labels
        )
        
        loss_i2t_mean = jnp.mean(loss_i2t)
        loss_t2i_mean = jnp.mean(loss_t2i)
        
        loss = (loss_i2t_mean + loss_t2i_mean) / 2.0
        
        return loss

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        # Debug input shapes
        logger.info("=== Input Shapes ===")
        for k, v in batch.items():
            logger.info(f"{k}: {v.shape}")
        logger.info("===================")

        dropout_rng = jax.random.fold_in(dropout_rng, state.step)

        def compute_loss(params):
            outputs = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)  # Model forward pass
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            loss = clip_loss(logits_per_image, logits_per_text)
            return loss

        dynamic_scale = state.dynamic_scale
        if dynamic_scale:
            grad_fn = dynamic_scale.value_and_grad(compute_loss)
            dynamic_scale, is_fin, loss, grads = grad_fn(state.params)
        else:
            grad_fn = alpa.value_and_grad(compute_loss)
            loss, grads = grad_fn(state.params)

        new_state = state.apply_gradients(grads=grads)

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

        metrics = {
            "loss": loss,
            "learning_rate": linear_decay_lr_schedule_fn(state.step),
        }

        return new_state, metrics, dropout_rng

    # Define eval fn
    def eval_step(params, batch):
        logits = model(**batch, params=params, train=False)
        logits_per_image = logits.logits_per_image
        logits_per_text = logits.logits_per_text
        loss = clip_loss(logits_per_image, logits_per_text)
        metrics = {"loss": loss}
        return metrics

    method = create_alpa_method(
        AlpaMethod(training_args.parallel_strategy), 
        training_args,
    )

    p_train_step = alpa.parallelize(
        train_step,
        method=method,
        donate_argnums=(0,)
    )

    if training_args.do_eval:
        p_eval_step = alpa.parallelize(
            eval_step,
            method=alpa.FollowParallel(
                p_train_step, 
                num_micro_batches=training_args.num_micro_batches
            )
        )                             

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed and grad_accum) = {train_batch_size}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

    if not training_args.skip_memory_metrics:
        server = jax.profiler.start_server(9999)

    train_time = 0
    train_metrics = []
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

    step_ct = 0
    last_time = time.time()

    epochs.write("Initial compilation. This might take some minutes...")
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        rng, input_rng = jax.random.split(rng)

        steps_per_epoch = len(train_dataset) // train_batch_size

        steps_trained_progress_bar = tqdm(
            range(steps_per_epoch),
            desc="Training...",
            position=1,
            leave=False,
            initial=(resume_step // grad_accum_steps),
        )
        for step, batch in enumerate(train_loader):
            cur_step = epoch * (len(train_dataset) // train_batch_size) + step

            # skip to the step from which we are resuming
            if cur_step < resume_step:
                continue
            

            batch = make_batch(
                batch
            )  # Since output of torch.DataLoader is `torch.Tensor`, we need to convert it into `jnp.array`

            print("Input batch shapes:")
            print("pixel_values:", batch["pixel_values"].shape)
            print("input_ids:", batch["input_ids"].shape)
            print("attention_mask:", batch["attention_mask"].shape)

            state, train_metric, dropout_rng = p_train_step(state, batch, dropout_rng)
            train_metrics.append(train_metric)

            if step % grad_accum_steps == 0:
                steps_trained_progress_bar.update(1)

            if dump_debug_info_train_step:
                dump_debug_info_train_step = False
                executable = p_train_step.get_last_executable()
                executable.sync()
                executable.dump_debug_info("alpa_debug_info")
                epochs.write(
                    f"Initial compilation completed. "
                    f"Time elapsed: {time.time() - train_start:.2f} s"
                )

            step_ct += 1
            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                executable.sync()
                latency = (time.time() - last_time) / step_ct
                throughput_tokens = np.prod(batch["input_ids"].shape) / latency
                throughput_tflops = alpa.util.compute_gpt_tflops(
                    batch_size=batch["input_ids"].shape[0],
                    seq_len=batch["input_ids"].shape[1],
                    num_layers=config.vision_config.num_hidden_layers + config.text_config.num_hidden_layers,
                    hidden_size=config.vision_config.hidden_size + config.text_config.hidden_size,
                    vocab_size=config.text_config.vocab_size,
                    num_gpus=alpa.get_global_num_devices(),
                    latency=latency,
                )
                step_ct = 0

                # Save metrics
                train_time += time.time() - train_start
                if HAS_TENSORBOARD:
                    write_train_metric(
                        train_metrics, train_time, cur_step, summary_writer
                    )

                train_metric = jax.tree_map(np.mean, train_metric)

                epochs.write(
                    f"Step... {cur_step} | "
                    f"Loss: {train_metric['loss'].mean():.4f}, "
                    f"Learning Rate: {train_metric['learning_rate'].mean():.5f}, "
                    f"Throughput: {throughput_tokens:.2f} token/s, "
                    f"{throughput_tflops:.2f} TFLOP/s"
                )

                train_metrics = []
                last_time = time.time()

            if training_args.do_eval:
                if (
                    cur_step % (training_args.eval_steps * grad_accum_steps) == 0
                    and cur_step > 0
                    and model_args.eval_strategy == "steps"
                ):
                    # ======================== Evaluating ==============================
                    eval_metrics = []
                    eval_steps = len(eval_dataset) // eval_batch_size
                    eval_iter = iter(eval_loader)
                    for batch in tqdm(
                        eval_loader, desc="Evaluating...", position=2, leave=False
                    ):
                        # Model forward
                        metrics = p_eval_step(state.params, make_batch(batch))
                        eval_metrics.append(metrics)

                        if dump_debug_info_eval_step:
                            dump_debug_info_eval_step = False
                            executable = p_eval_step.get_last_executable()
                            executable.dump_debug_info("alpa_debug_info")

                    # normalize eval metrics
                    eval_metrics = alpa.util.get_metrics(eval_metrics)
                    eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

                    # Print metrics and update progress bar
                    desc = f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']})"
                    epochs.write(desc)
                    epochs.desc = desc

                    # Save metrics
                    if HAS_TENSORBOARD and jax.process_index() == 0:
                        # cur_step = epoch * (len(train_dataset) // train_batch_size)
                        write_eval_metric(eval_metrics, cur_step, summary_writer)
                    if (
                        has_wandb
                        and jax.process_index() == 0
                        and ("wandb" in training_args.report_to)
                    ):
                        _metrics = {
                            f"eval_{k}": mb_item(v) for k, v in eval_metrics.items()
                        }
                        wandb.log({"eval_step": cur_step, **_metrics})

            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                epochs.write("\nSave checkpoint...")
                alpa.prefetch(state.params)
                params = alpa.util.map_to_nparray(state.params)
                model.save_pretrained(training_args.output_dir, params=params)
                tokenizer.save_pretrained(training_args.output_dir)
                if training_args.push_to_hub:
                    repo.push_to_hub(
                        commit_message=f"Saving weights and logs of step {cur_step}",
                        blocking=False,
                    )

    # Eval after training
    if training_args.do_eval:
        eval_metrics = []
        eval_steps = len(eval_dataset) // eval_batch_size
        eval_iter = iter(eval_loader)
        for batch in tqdm(eval_loader, desc="Evaluating...", position=2, leave=False):
            # Model forward
            metrics = p_eval_step(state.params, make_batch(batch))
            eval_metrics.append(metrics)

        # normalize eval metrics
        eval_metrics = alpa.util.get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

        # Print metrics and update progress bar
        desc = f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']})"
        epochs.write(desc)
        epochs.desc = desc

        # Save metrics
        if HAS_TENSORBOARD and jax.process_index() == 0:
            # cur_step = epoch * (len(train_dataset) // train_batch_size)
            write_eval_metric(eval_metrics, cur_step, summary_writer)
        if (
            has_wandb
            and jax.process_index() == 0
            and ("wandb" in training_args.report_to)
        ):
            _metrics = {f"eval_{k}": mb_item(v) for k, v in eval_metrics.items()}
            wandb.log({"eval_step": cur_step, **_metrics})

    # Save the final model
    epochs.write("\nSave the final model...")
    alpa.prefetch(state.params)
    params = alpa.util.map_to_nparray(state.params)
    model.save_pretrained(training_args.output_dir, params=params)
    tokenizer.save_pretrained(training_args.output_dir)

    # # save model after training is over
    # model.save_pretrained(
    #     training_args.output_dir,
    #     params=unreplicate(state.params),
    #     push_to_hub=training_args.push_to_hub,
    #     commit_message=f"Saving weights and logs at step {cur_step}",
    #     repo_name_or_path=training_args.output_dir
    # )

    maybe_stop_wandb()

    alpa.shutdown()


if __name__ == "__main__":
    main()
