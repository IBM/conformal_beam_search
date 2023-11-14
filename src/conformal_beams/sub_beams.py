#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from dataclasses import dataclass, InitVar, field
from math import ceil
from typing import Optional, Union

import numpy as np
from scipy import stats as stats
from sklearn.model_selection import train_test_split
from typing_extensions import Self

from conformal_beams.utils import pad_to_match


@dataclass(repr=False)
class SubBeamData:
    """Dataclass for sub-beam predictions"""

    sequences: np.ndarray  # shape (n_sequences, beam_size, max_len_sequences)
    scores: np.ndarray  # shape (n_sequences, beam_size)
    labels: Optional[np.ndarray] = None  # shape (n_sequences, max_len_labels)
    sorted_by_score: InitVar[bool] = False
    ranks: Optional[np.ndarray] = field(
        init=False, default=None
    )  # shape (n_sequences,)
    set_sizes: Optional[np.ndarray] = field(
        init=False, default=None
    )  # shape (n_sequences,)

    def __post_init__(self, sorted_by_score: bool):
        n_sequences, beam_size, _ = self.sequences.shape
        assert self.scores.shape == (
            n_sequences,
            beam_size,
        ), "Shape mismatch: sequences and scores"
        if self.labels is not None:
            assert (
                len(self.labels.shape) == 2 and self.labels.shape[0] == n_sequences
            ), "Shape mismatch: sequences and labels"
        if not sorted_by_score:
            idx = np.argsort(self.scores, axis=1)[:, ::-1]  # Biggest first
            self.scores = np.take_along_axis(self.scores, idx, 1)
            self.sequences = np.take_along_axis(
                self.sequences, idx.reshape(*idx.shape, 1), 1
            )

    def pad_to_match(self, padding_token_id: int):
        self.sequences, self.labels = pad_to_match(
            self.sequences, self.labels, padding_token_id
        )

    @staticmethod
    def concat_optional(vals):
        has_val = np.array([r is not None for r in vals])

        if np.all(has_val):
            return np.concatenate(vals, 0)
        else:
            assert np.all(~has_val), "Mix of present and absent optional data"
            return None

    def split(self, train_size=0.8, test_size=0.2, random_state: Optional[int] = None):
        """Build a random split (train/test)"""
        idx_train, idx_test = train_test_split(
            np.arange(len(self)),
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        train_sbd = SubBeamData(
            sequences=self.sequences[idx_train],
            scores=self.scores[idx_train],
            labels=self.labels[idx_train] if self.labels is not None else None,
            sorted_by_score=False,
        )
        test_sbd = SubBeamData(
            sequences=self.sequences[idx_test],
            scores=self.scores[idx_test],
            labels=self.labels[idx_test] if self.labels is not None else None,
            sorted_by_score=False,
        )
        return train_sbd, test_sbd

    @classmethod
    def concatenate(cls, *sb_datas: Self, padding_token: int):
        """Concatenate N SubBeamData objects, appropriately padding uneven sequences"""
        scores = np.concatenate([sbd.scores for sbd in sb_datas], axis=0)
        ranks = cls.concat_optional([sbd.ranks for sbd in sb_datas])
        set_sizes = cls.concat_optional([sbd.set_sizes for sbd in sb_datas])

        max_seq_len = max([sbd.sequences.shape[-1] for sbd in sb_datas])
        sequences = np.concatenate(
            [
                np.pad(
                    sbd.sequences,
                    [(0, 0), (0, 0), (0, max_seq_len - sbd.sequences.shape[-1])],
                    constant_values=padding_token,
                )
                for sbd in sb_datas
            ],
            axis=0,
        )

        nlabels = len([sbd.labels for sbd in sb_datas if sbd.labels is not None])
        if nlabels == len(sb_datas):
            max_label_len = max([sbd.labels.shape[-1] for sbd in sb_datas])
            labels = np.concatenate(
                [
                    np.pad(
                        sbd.labels,
                        [(0, 0), (0, max_label_len - sbd.labels.shape[-1])],
                        constant_values=padding_token,
                    )
                    for sbd in sb_datas
                ],
                axis=0,
            )
        elif nlabels == 0:
            labels = None
        else:
            raise AssertionError("Mix of present and absent optional data")

        catted = cls(sequences, scores, labels, sorted_by_score=True)
        catted.ranks = ranks
        catted.set_sizes = set_sizes
        return catted

    def __repr__(self):
        return f"{self.__class__.__name__}(n_sequences={len(self.sequences)},beam_size={self.sequences.shape[1]})"

    def __len__(self):
        return len(self.sequences)


class SubBeamConformal:
    """Conformal predictor for conformal sub-beam predictions.
    NB: This operates in an efficient batched way.
    """

    def __init__(self, alpha: float, delta: float, padding_token_id: int):
        self.alpha = alpha  # Conformal risk
        self.delta = delta  # Beam coverage bound risk
        self.padding_token_id = padding_token_id
        # Filled dynamically
        self.beam_size = None
        self.threshold = None
        self.N_cal_covered = None
        self.N_cal = None
        self.beam_cov_bound = None
        self.global_bound = None

    def compute_beam_cov(self, n_tot: int, n_in_beam: int):
        """Fraction of beams containing the true sequence"""
        return n_in_beam / n_tot

    def compute_beam_cov_bound(self, n_tot: int, n_in_beam: int, risk: float):
        """Theoretical beam coverage lower bound at a given risk level"""
        cov_rv = stats.beta(n_in_beam, n_tot + 1 - n_in_beam)
        cov_pac_low = cov_rv.ppf(risk)
        return cov_pac_low

    def compute_ranks(self, data: SubBeamData):
        return compute_ranks(data, self.beam_size)

    def calibrate(self, data: SubBeamData):
        if data.labels is None:
            raise ValueError("Labels required for calibration")
        data.pad_to_match(self.padding_token_id)
        self.beam_size = data.sequences.shape[1]
        if data.ranks is None:
            data = self.compute_ranks(data)
        self.N_cal = len(data.sequences)
        in_beam_mask = data.ranks < self.beam_size
        self.N_cal_covered = int(np.sum(in_beam_mask))

        calibration_scores = np.take_along_axis(
            data.scores[in_beam_mask], data.ranks[in_beam_mask].reshape(-1, 1), 1
        ).flatten()

        alpha_tilde = 1 - ceil((1 - self.alpha) * self.N_cal_covered) / (
            self.N_cal_covered + 1
        )
        self.threshold = np.quantile(calibration_scores, alpha_tilde)

        self.beam_cov_bound = self.compute_beam_cov_bound(
            self.N_cal, self.N_cal_covered, risk=self.delta
        )
        self.global_bound = (1 - self.alpha) * self.beam_cov_bound

        return {
            "alpha": self.alpha,
            "beam_cov_bound": self.beam_cov_bound,
            "global_bound": self.global_bound,
            "threshold": self.threshold,
        }

    def predict(self, data: SubBeamData):
        set_masks = data.scores > self.threshold
        set_sizes = set_masks.sum(-1)
        data.set_sizes = set_sizes
        return data

    def evaluate(self, data: SubBeamData):
        if data.labels is None:
            raise ValueError("Labels required for calibration")
        data.pad_to_match(padding_token_id=self.padding_token_id)
        data = self.predict(data)
        if data.ranks is None:
            data = self.compute_ranks(data)

        beam_covered_mask = data.ranks < self.beam_size
        n_tot = len(beam_covered_mask)
        n_cov_beam = beam_covered_mask.sum()
        n_cov_global = (data.ranks < data.set_sizes).sum()

        cov_beam = n_cov_beam / n_tot
        cov_cond = n_cov_global / n_cov_beam
        cov_global = n_cov_global / n_tot

        mae = np.abs(data.ranks + 1 - data.set_sizes).mean()

        return {
            "cov_beam": cov_beam,
            "cov_cond": cov_cond,
            "cov_global": cov_global,
            "mae": mae,
            "global_bound": self.global_bound,
            "beam_bound": self.beam_cov_bound,
            "alpha": self.alpha,
        }


def compute_ranks(data: SubBeamData, beam_size: int):
    """Compute the rank of the true sequence in the beam if it is there"""
    n, l = data.labels.shape
    matches = data.sequences == data.labels.reshape(n, 1, l)
    match_idxs = np.argwhere(np.all(matches, -1))
    ranks = np.full(shape=len(matches), fill_value=beam_size)
    match_seqids = match_idxs[:, 0]
    match_ranks = match_idxs[:, 1]
    ranks[match_seqids] = match_ranks
    data.ranks = ranks
    return data
