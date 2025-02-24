import sys

sys.path.append(".")

from examples.utils import *

import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import alpa
from alpa.model.model_util import TrainState
import jax
import jax.numpy as jnp
import optax
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
from transformers.utils import get_full_repo_name

init_alpa("ray")

logger = setup_logging(__name__)

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments(TrainingArguments):
    pass

@dataclass
class ModelArguments(ModelArguments):
    pass

@dataclass
class DataTrainingArguments(DataTrainingArguments):
    pass


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


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

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

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

    if model_args.model_name_or_path:
        model = FlaxAutoModelForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = FlaxAutoModelForImageClassification.from_config(
            config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
        )

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * alpa.get_global_num_devices()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * alpa.get_global_num_devices()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples])

        batch = {"pixel_values": pixel_values, "labels": labels}
        batch = {k: v.numpy() for k, v in batch.items()}

        return batch

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

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

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

    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw, dynamic_scale=None)

    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        return loss.mean()

    # Define gradient update step fn
    def train_step(state, batch):

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, train=True)[0]
            loss = loss_fn(logits, labels)
            return loss

        grad_fn = alpa.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}

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
    method = create_alpa_method(
        AlpaMethod(training_args.parallel_strategy),
        training_args
    )
    p_train_step = alpa.parallelize(train_step,
                                    method=method,
                                    donate_argnums=(0,))
    if training_args.do_eval:
        p_eval_step = alpa.parallelize(
            eval_step,
            method=alpa.FollowParallel(
                p_train_step,
                num_micro_batches=training_args.per_device_eval_batch_size
            )
        )
    dump_debug_info_train_step = dump_debug_info_eval_step = True

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    last_time = time.time()
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_metrics = []

        steps_per_epoch = len(train_dataset) // train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for step, batch in enumerate(train_loader):
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)

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

        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate:"
            f" {train_metric['learning_rate']}), "
            f"Throughput: {images_per_second:.2f} images/s"
        )

        # Save metrics
        if has_tensorboard and jax.process_index() == 0:
            cur_step = epoch * (len(train_dataset) // train_batch_size)
            write_train_metric(train_metrics, train_time, cur_step, summary_writer)

        # ======================== Evaluating ==============================
        if training_args.do_eval:
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
                write_eval_metric(eval_metrics, cur_step, summary_writer)

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            alpa.prefetch(state.params)
            params = alpa.util.map_to_nparray(state.params)
            model.save_pretrained(training_args.output_dir, params=params)
            if training_args.push_to_hub:
                repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)

    alpa.shutdown()

if __name__ == "__main__":
    main()