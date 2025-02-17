# ODPO: Direct Preference Optimization with an Offset

<p align="center">
  <img src="./odpo.png" width=900>
</p>


> :warning: this repo is based on and is an extension of the [DPO repo](https://github.com/eric-mitchell/direct-preference-optimization). Please refer to the original repo for general installation guidelines. We thank the authors for releasing their code.

## What is new to DPO codebase?
- **ODPO loss:** To use this loss, just set `loss=odpo`. You can change the value of `alpha` simply by `loss.alpha=X`. The default value is 1.
- **Data** (under `data/`) to reproduce our results on 3 tasks: sentiment control, toxicity control, and summarization. 
- **Evaluation pipeline** to test KL divergence and rewards.

### Sentiment Control Task
To reproduce the results of paper, the SFT checkpoint should be a finetuned gpt2-large model on IMDB train set.
```bash
python train.py \
model=gpt2-large \
datasets=[imdb] \
loss=odpo \
seed=1 \
loss.beta=0.7 \
loss.alpha=0.5 \
model.name_or_path=[SFT CHECKPOINT] \
exp_name=imdb \
gradient_accumulation_steps=2 \
batch_size=32 \
eval_batch_size=32 \
trainer=FSDPTrainer \
local_run_dir=[LOCAL DIR] \
sample_during_eval=false
```

### Toxicity Control Task 
```bash
python train.py \
model=gpt-neo \
datasets=[toxicity] \
loss=odpo \
seed=1 \
loss.beta=0.05 \
loss.alpha=1. \
model.name_or_path=[SFT CHECKPOINT] \
exp_name=toxicity \
gradient_accumulation_steps=2 \
batch_size=32 \
eval_batch_size=32 \
trainer=FSDPTrainer \
local_run_dir=[LOCAL DIR] \
sample_during_eval=false"
```

### Summarization Task
For this task, we first train the model (without the evaluation loop), and then test the model by sampling with different temperatures. 

For training:
```bash
python train.py \
model=gptj \
datasets=[tldr] \
loss=odpo \
seed=0 \
loss.beta=0.5 \
model.name_or_path=[SFT CHECKPOINT] \
exp_name=tldr \
gradient_accumulation_steps=2 \
batch_size=16 \
max_prompt_length=512 \
max_length=1024 \
eval_batch_size=16 \
trainer=FSDPTrainer \
local_run_dir=[LOCAL DIR] \
sample_during_eval=false
```

Sampling:
```bash
python train.py \
function=eval \
model=gptj \
datasets=[tldr] \
loss=odpo \
seed=0 \
loss.beta=0.5 \
temperature=0.25 \
n_eval_examples=100 \
model.name_or_path=[SFT CHECKPOINT] \
model.archive=tldr/odpo/LATEST/policy.pt \
exp_name=tldr_test \
gradient_accumulation_steps=2 \
max_prompt_length=512 \
max_length=1024 \
eval_batch_size=16 \
trainer=FSDPTrainer \
local_run_dir=[LOCAL DIR] \
sample_during_eval=false
```
**Note:** `model.archive` is the path of the model that we finetuned with ODPO.


# Citing ODPO
If ODPO or this repository is useful in your own research, you can use the following BibTeX entry:

    @inproceedings{amini-etal-2024-direct,
    title = "Direct Preference Optimization with an Offset",
    author = "Amini, Afra  and
    Vieira, Tim  and
    Cotterell, Ryan",
    editor = "Ku, Lun-Wei  and
    Martins, Andre  and
    Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.592/",
    doi = "10.18653/v1/2024.findings-acl.592",
    pages = "9954--9972",
    }
