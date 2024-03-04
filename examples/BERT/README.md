

The following example fine-tunes BERT on SQuAD:


```bash
python run_qa_alpa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train   \
  --do_eval   \
  --max_seq_length 384 \
  --doc_stride 128 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 12 \
  --output_dir ./bert-qa-squad \
  --eval_steps 1000 \
  --cache_dir ./cache
```

Installation:

```
# First, clone the HF transformers repository to a separate directory (e.g., home)

git clone https://github.com/huggingface/transformers

# Then, install from source when in the transformers directory

pip install -e .

# Finally, install the remaining packages

pip install datasets
pip install evaluate

```