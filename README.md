# Conformal Beam Search

Implementation and experiment code
for [Conformal Autoregressive Generation: Beam Search with Coverage Guarantees](https://arxiv.org/abs/2309.03797)

## Usage

### Installation

We recommend setting up a [virtual environment](https://docs.python.org/3/library/venv.html).

```shell
# In the conformal_beam_search root directory
python -m venv ./confbeam_venv
source ./confbeam_venv/bin/activate

# Optional to reproduce exactly our environment
pip install -r requirements.txt -U

pip install -e .
```

### Reproducing Experiments

The package `confbeam_experiments` implements the experimental code to re-run the experiments in our paper through a
command line interface.

### Conformal Beam Subsets
The CLI for the experiments on the Conformal Beam Subsets experiments is `confbeam_experiments.sub_beams`
```shell
python -m confbeam_experiments.sub_beams --help

```

### Dynamic Conformal Beam Search
The CLI for the experiments on the Dynamic Conformal Beam Search experiments is `confbeam_experiments.dyn_beams`
```shell
python -m confbeam_experiments.dyn_beams --help

```

### Using the Conformal Decoders

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

