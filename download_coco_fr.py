import os
import torch
os.environ["HF_HOME"] = "/scratch/shashwat.s/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/shashwat.s/.cache"
from datasets import load_dataset

mscoco = load_dataset('HuggingFaceM4/COCO')

# %%

sample_train = mscoco['train']
sample_train

# %%
sample_train[0]['image']
len(sample_train)

