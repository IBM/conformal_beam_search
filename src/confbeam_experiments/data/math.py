#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from functools import partial
from itertools import chain

import numpy as np
from datasets import Dataset, DatasetDict
from pytorch_lightning import seed_everything


def format_data(x1: int, x2: int, y: int, randomize=False):
    """Format a single addition problem"""
    if randomize:
        x1, x2 = np.random.permutation((x1, x2))
    question = f"{x1}+{x2}="
    return {
        "question": question,
        "answer": str(y),
        "digits_left": len(str(x1)),
        "digits_right": len(str(x2)),
        "question_len": len(question),
    }


def generate_additions(x1: np.ndarray, x2: np.ndarray, randomize=False):
    """Generate addition problems from pairs of integers"""
    y = x1 + x2
    for i in range(len(y)):
        yield format_data(x1[i], x2[i], y[i], randomize=randomize)


def full_two_digit_dataset():
    """Create all possible addition problems with one or two digits"""
    x1 = np.arange(100)
    x2 = np.arange(100)
    for i in range(100):
        x2 = np.roll(x2, 1)
        for data in generate_additions(x1, x2, randomize=False):
            yield data


def two_plus_three(n_reuse_3=10, n_additions_total=10000):
    x2 = np.arange(100)
    max_n_iter = (1000 - 100) * 2  # 3digit * permutation
    n_iter = n_additions_total // n_reuse_3 // 100
    assert n_iter < max_n_iter
    for i in range(n_iter):
        x1 = np.random.randint(low=100, high=999, size=100)
        for reuse in range(n_reuse_3):
            x1 = np.random.permutation(x1)
            for data in generate_additions(x1, x2, randomize=True):
                yield data



def m_plus_n(m, n, n_reuse=10, n_additions_total=50000):
    """Generate n_additions_total addition problems with m and n digits in the operands.
    n_addition_total // n_reuse pairs m- and n- digits numbers are generated,
    and problems are created by taking n_reuse permutations
    """
    low_n = 10 ** (n - 1)
    high_n = 10**n - 1
    low_m = 10 ** (m - 1)
    high_m = 10**m - 1
    n_distinct_numbers = n_additions_total // n_reuse
    x1 = np.random.randint(low=low_m, high=high_m, size=n_distinct_numbers)
    x2 = np.random.randint(low=low_n, high=high_n, size=n_distinct_numbers)
    for reuse in range(n_reuse):
        x1 = np.random.permutation(x1)
        x2 = np.random.permutation(x2)
        for data in generate_additions(x1, x2, randomize=True):
            yield data



def get_math_data_generator(n_reuse=10, n_addition_per_problem=50000, seed=32):
    """Create the full data generator with many combinations of digits"""
    seed_everything(seed)
    return chain(
        full_two_digit_dataset(),
        two_plus_three(n_reuse_3=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(3, 3, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(2, 4, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(3, 4, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(4, 4, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(2, 5, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(3, 5, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(4, 5, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(5, 5, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(2, 8, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(4, 6, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
        m_plus_n(3, 7, n_reuse=n_reuse, n_additions_total=n_addition_per_problem),
    )


def get_dataset(n_reuse=10, n_addition_per_problem=50000, seed=32):
    """Create the dataset object with many combinations of digits"""
    generator = partial(
        get_math_data_generator,
        n_reuse=n_reuse,
        n_addition_per_problem=n_addition_per_problem,
        seed=seed,
    )
    ds = Dataset.from_generator(generator, gen_kwargs=dict(), config_name="additions")
    return ds


def split_dataset(n_reuse=10, n_addition_per_problem=10000, seed=32):
    ds = get_dataset(
        n_reuse=n_reuse, n_addition_per_problem=n_addition_per_problem, seed=seed
    )
    datadict = ds.train_test_split()
    ds_train = datadict["train"]
    train_dict = ds_train.train_test_split()
    ds_train = train_dict["train"]
    ds_val = train_dict["test"]
    datadict["train"] = ds_train
    datadict["val"] = ds_val
    return datadict


def tokenize_datasets(datadict: DatasetDict, tokenizer):
    def tokenize(data):
        return tokenizer(data["question"], text_target=data["answer"])

    tokenized_datadict = DatasetDict()
    for k, v in datadict.items():
        tokenized = v.map(tokenize, batched=True)
        tokenized_datadict[k] = tokenized
    return tokenized_datadict
