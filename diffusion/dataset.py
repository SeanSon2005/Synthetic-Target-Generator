from datasets import load_dataset

# create dataset
dataset = load_dataset("targetData", data_dir="diffusion/data")
dataset.push_to_hub("alphanumeric_targets")