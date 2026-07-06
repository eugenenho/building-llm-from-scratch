# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 2. Encode the text into token IDs

  `cs336_basics/main.py` trains on pre-tokenized `uint16` arrays (`.npy`), not raw
  text. The trained BPE tokenizer (32K vocab) is committed under `outputs_owt/`
  (`vocab.json` + `merges.txt`), so you don't retrain it — just encode the text you
  downloaded above:

  ```bash
  uv run python -m cs336_basics.encode_data
  ```

  This reads `data/owt_train.txt` / `data/owt_valid.txt` and writes
  `data/owt-train-encoded.npy` / `data/owt-valid-encoded.npy` — the files the
  training config's `train_data_path` / `val_data_path` point to. If any input is
  missing, the script exits and tells you to run the download step first.

  **Note:** encoding the OWT train set (~5 GB) holds the full token stream in memory
  before writing, so it needs a machine with substantial free RAM and takes several
  minutes.
