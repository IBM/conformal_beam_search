# Conformal Beam Search

Implementation and experiment code for the paper:  
[Conformal Autoregressive Generation: Beam Search with Coverage Guarantees](https://arxiv.org/abs/2309.03797)

## Usage

### Reproducing Experiments

The package `confbeam_experiments` implements the experimental code to re-run the experiments in our paper through a
command line interface. Models are specified through YAML config files (see below)

#### Conformal Beam Subsets
The CLI for the experiments on the Conformal Beam Subsets experiments is `confbeam_experiments.sub_beams`
```shell
python -m confbeam_experiments.sub_beams --help
```

```shell
Usage: python -m confbeam_experiments.sub_beams [OPTIONS]

  Sub-beam experiment entry point

Options:
  -x, --experiment [rxn|additions]
                                  # [required]
  -o, --out_path PATH             # [required]
  -c, --config PATH               # [required]
  --sharding INTEGER              # Number of shards of the dataset. Defines batching for beam search
  --help                          # Show this message and exit.
```

#### Dynamic Conformal Beam Search
The CLI for the experiments on the Dynamic Conformal Beam Search experiments is `confbeam_experiments.dyn_beams`

This is more computationally demanding than the Beam Subsets and we therefore separate the prediction and result analysis to allow multiple prediction runs.

```shell
# Prediction CLI
python -m confbeam_experiments.dyn_beams predict --help
```

```shell
Usage: python -m confbeam_experiments.dyn_beams predict
           [OPTIONS]

  Experiment entry point Usage: python -m
  confbeam_experiment.dyn_beams predict -x additions -o
  <output_path> -c config.yaml

Options:
  -x, --experiment [rxn|additions]
                                  # [required]
  -o, --out_path PATH             # [required]
  -c, --config PATH               # [required]
  -n, --n_rep INTEGER             # Number of repetition per alpha
  -a, --alphas FLOAT              # Risk level
  -s, --seed INTEGER
  --pb / --no-pb                  # Show progress bar
  --help                          # Show this message and exit.
```


  
```shell
# Analysis CLI
python -m confbeam_experiments.dyn_beams predict --help
```

```shell
Usage: python -m confbeam_experiments.dyn_beams analyze
           [OPTIONS]

  Aggregation entry point Usage: python -m
  confbeam_experiment.dyn_beams analze --exp_dir <experiment output
  path>

Options:
  --exp_dir PATH  # [required]
  --out_dir PATH
  --help          # Show this message and exit.
```

### Model configuration files

#### Additions task
The addition tasks is performed with a T5 model from HuggingFace. The configuration file is very simple
```yaml
# addition_t5_config.yaml

model_spec_or_checkpoint: "path_to_checkpoint"
# also works:
# model_spec_or_checkpoint: "t5-base"
```

#### Reactions task
The reaction tasks relies on a modified T5 model
```yaml
# rxn_config.yaml
tokenizer_path: <path>/moltok_v2
model_path: <path>/flan-t5-small-rxn
checkpoint_path: <path>/pytorch_model.bin
data_dir_path: <data directory path>
data_save_path: <output path>
```

## Installation

We recommend setting up a [virtual environment](https://docs.python.org/3/library/venv.html).

```shell
# In the conformal_beam_search root directory
python -m venv ./confbeam_venv
source ./confbeam_venv/bin/activate

# Optional to reproduce exactly our environment
pip install -r requirements.txt -U

pip install -e .
```

## Citation

If you use our code or methods, please cite our manuscript:

Deutschmann, N., Alberts, M. and Martínez, M.R., 2023. *Conformal Autoregressive Generation: Beam Search with Coverage
Guarantees*. arXiv preprint [arXiv:2309.03797](https://arxiv.org/abs/2309.03797).

```bibtex
@misc{deutschmann2023conformal,
    title = {Conformal Autoregressive Generation: Beam Search with Coverage Guarantees},
    author = {Nicolas Deutschmann and Marvin Alberts and María Rodríguez Martínez},
    year = {2023},
    eprint = {2309.03797},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

## Code contributors
Nicolas Deutschmann [@ndeutschmann](https://github.com/ndeutschmann)  
Marvin Alberts [@MAlberts99](https://github.com/MAlberts99)

