#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
from numbers import Real
from typing import Dict, Tuple, TypeAlias, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing_extensions import TypedDict, NotRequired

from confbeam_experiments.models import ModelManager
from conformal_beams.utils import torch2numpy

ArrayLike: TypeAlias = Union[np.ndarray, torch.Tensor]


class DynamicBeamData(TypedDict):
    """Internal data passed through decoding steps"""

    input_seq: ArrayLike
    input_msk: ArrayLike
    beam: ArrayLike
    beam_scores: ArrayLike
    finished_beams: ArrayLike
    beam_lengths: ArrayLike


class DynamicBeamOutput(DynamicBeamData):
    """Inference output data"""

    beam_sizes: List[int]


class DynamicBeamEvaluation(DynamicBeamOutput):
    """Evaluation output data with label-related information"""

    true_seq: ArrayLike
    ranks: List[Real]
    in_beam: List[bool]
    sample_idx: NotRequired[int]


class DynamicBeamConformal:
    """Conformal predictor for dynamic beam generation

    For the moment, inference is non-batched and proceeds sample-per-sample.
    NB: for low alpha or poorer models large beam sizes would prevent batch sizes >1
    """

    def __init__(self, manager: ModelManager, alpha: float, max_len: int):
        self.manager = manager
        self.alpha = alpha
        self.max_len = max_len
        self.thresholds = None
        self.ks = None

    @property
    def device(self):
        return self.manager.model.device

    def compute_scores(
        self, model_input_data: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-likelihood for each truncation of input sentence on a batch fitting in memory

        Args:
            model_input_data: dictionary of model inputs
            labels: NxL token labels. Uses -100 as a padding token

        Returns: tensor of shape (NxL)
            sequence-level log-likelihood for each sequence truncation. Assumes padding token is predicted
            with p=1 after EOS token
        """
        model_input_data = {k: v.to(self.device) for k, v in model_input_data.items()}
        labels = labels.to(self.device)
        with torch.no_grad():
            pred_res = self.manager.model(**model_input_data)
        shape = pred_res.logits.shape[:-1]
        seq_scores = -F.cross_entropy(
            pred_res.logits.flatten(0, 1), labels.flatten(), reduction="none"
        )

        all_scores = seq_scores.unflatten(0, shape).cumsum(-1) / torch.cumsum(
            (labels != -100).to(int), -1
        )

        return all_scores

    def score_dataloader(self, dataloader: DataLoader):
        """Compute truncated log-likelihood scores on larger-than-GPU-memory dataset"""
        all_scores = []
        all_inputs = []
        all_input_masks = []
        all_labels = []

        for batch in dataloader:
            all_inputs.extend(list(batch["input_ids"]))
            all_input_masks.extend(list(batch["attention_mask"]))
            all_labels.extend(list(batch["labels"]))
            all_scores.append(
                self.compute_scores(batch, labels=batch["labels"]).cpu().numpy()
            )

        max_len = max([s.shape[1] for s in all_scores])
        padding_extra = 3
        n_examples = sum([s.shape[0] for s in all_scores])
        scores_per_len = np.zeros((n_examples, max_len + padding_extra))
        for step in range(max_len + padding_extra):
            start = 0
            for sbatch in all_scores:
                end = len(sbatch) + start
                try:
                    sb_scores = sbatch[:, step]
                except IndexError or RuntimeError:
                    sb_scores = sbatch[:, -1]
                scores_per_len[start:end, step] = sb_scores
                start = end

        return scores_per_len

    def calibrate(self, cal_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute score thresholds and dropped sample counts based on truncated sequence scores
        ($t^{(l)}$ and $k^{(l)}$ in sec. 4.1).

        Args:
            cal_scores: np.ndarray with shape (NxL)
                sequence-level calibration log-likelihoods for each truncation

        Returns: tuple of arrays
            Score thresholds and number of calibration samples dropped at each truncation
        """
        ncal, max_l = cal_scores.shape
        self.max_len = min(max_l, self.max_len)

        mask = np.full(ncal, True)
        thresholds = np.zeros(self.max_len)
        ks = np.zeros(self.max_len)
        for step in range(self.max_len):
            n = np.sum(mask)
            k_conf = np.floor(self.alpha * (n + 1))
            t = np.quantile(cal_scores[mask, step], k_conf / n, method="lower")
            thresholds[step] = t
            mask = mask & (cal_scores[:, step] > t)
            ks[step] = k_conf
        self.thresholds = thresholds
        self.ks = ks
        return thresholds, ks

    def decode(self, input_seq: np.ndarray, input_msk: np.ndarray) -> DynamicBeamOutput:
        """Inference-time decoding on a single sample"""
        input_seq = torch.tensor(input_seq).to(self.device).reshape(1, -1)
        input_msk = torch.tensor(input_msk).to(self.device).reshape(1, -1)
        beam_sizes = []
        decode_data = self.init_beam(
            input_seq,
            input_msk,
        )
        beam_sizes.append(len(decode_data["beam"]))
        for step in range(self.max_len - 1):
            if torch.all(decode_data["finished_beams"]):
                break
            decode_data = self.continue_beam(
                **decode_data,
            )
            beam_sizes.append(len(decode_data["beam"]))
        output = DynamicBeamOutput(beam_sizes=beam_sizes, **decode_data)
        return torch2numpy(output)

    def evaluate_sample(
        self, input_seq: np.ndarray, input_msk: np.ndarray, label: np.ndarray
    ) -> DynamicBeamEvaluation:
        input_seq = torch.tensor(input_seq).to(self.device).reshape(1, -1)
        input_msk = torch.tensor(input_msk).to(self.device).reshape(1, -1)
        label = torch.tensor(label)

        in_beam = []
        ranks = []
        beam_sizes = []
        decode_data = self.init_beam(
            input_seq,
            input_msk,
        )
        in_beam_at_step, rank_at_step = self.check_in_beam(label, decode_data)
        in_beam.append(in_beam_at_step)
        ranks.append(rank_at_step)
        beam_sizes.append(len(decode_data["beam"]))
        for step in range(self.max_len - 1):
            if torch.all(decode_data["finished_beams"]):
                break
            decode_data = self.continue_beam(
                **decode_data,
            )
            in_beam_at_step, rank_at_step = self.check_in_beam(label, decode_data)
            in_beam.append(in_beam_at_step)
            ranks.append(rank_at_step)
            beam_sizes.append(len(decode_data["beam"]))
        output = DynamicBeamEvaluation(
            true_seq=label,
            ranks=ranks,
            beam_sizes=beam_sizes,
            in_beam=in_beam,
            **decode_data,
        )
        return torch2numpy(output)

    def init_beam(self, input_seq, input_msk) -> DynamicBeamData:
        start_token = self.manager.model.config.decoder_start_token_id
        eos_token = self.manager.model.config.eos_token_id
        beam = torch.full(size=(1, 1), fill_value=start_token).to(self.device)

        with torch.no_grad():
            output = self.manager.model(
                input_ids=input_seq, attention_mask=input_msk, decoder_input_ids=beam
            )

        lsm_scores = torch.log_softmax(output.logits[:, -1, :], -1)

        threshold_mask = lsm_scores > self.thresholds[0]
        beam_and_tok_id = torch.argwhere(threshold_mask)
        new_beam_scores = lsm_scores[threshold_mask]

        beam_ids = beam_and_tok_id[:, 0]
        new_token_ids = beam_and_tok_id[:, 1:]

        new_beam_stem = beam.gather(0, beam_ids.unsqueeze(1))
        new_beam = torch.concatenate([new_beam_stem, new_token_ids], 1)
        new_lens = torch.ones(len(new_beam), device=new_beam.device, dtype=int)
        finished_beams = new_beam[:, -1] == eos_token

        return {
            "input_seq": input_seq,
            "input_msk": input_msk,
            "beam": new_beam,
            "beam_scores": new_beam_scores,
            "finished_beams": finished_beams,
            "beam_lengths": new_lens,
        }

    def continue_beam(
        self,
        input_seq,
        input_msk,
        beam,
        beam_scores,
        finished_beams,
        beam_lengths,
    ) -> DynamicBeamData:
        eos_token = self.manager.model.config.eos_token_id
        pad_token = self.manager.model.config.pad_token_id
        beam_size = len(beam)
        # beams sequences [0,t1, t2, .., t(L-1)]. We do the Lth decoding, indexed at L-1
        decoding_step_id = beam.shape[1] - 1
        with torch.no_grad():
            output = self.manager.model(
                input_ids=input_seq.expand(beam_size, -1),
                attention_mask=input_msk.expand(beam_size, -1),
                decoder_input_ids=beam,
            )

        # Normalize logits. Put to 0 for finished beams
        # These are token-level conditional probabilities
        next_logprobs = torch.log_softmax(output.logits[:, -1, :], -1)
        next_logprobs = next_logprobs * (~finished_beams).unsqueeze(-1).to(
            next_logprobs
        )
        # Add the condition logprob to get the sequence probs
        next_logprobs = next_logprobs + beam_scores.unsqueeze(-1)

        # Ensure that finished beams are padded with prob=1 (logprob=0). For now all tokens have prob=1
        # We need to set all other tokens to prob=0 (logprob=-inf)
        # A) Build a mask beamsize*Vocab. True iff: (finished beam) AND (not padding token)
        eos_padding_mask = finished_beams.unsqueeze(-1).expand_as(next_logprobs).clone()
        eos_padding_mask[:, pad_token] = False
        # B) Fill in -inf
        next_logprobs.masked_fill_(eos_padding_mask, -float("inf"))

        # Compute lengths to normalize. All beams that were not finished at the last step get expanded by 1.
        new_beam_lengths = beam_lengths + ~finished_beams

        # Now we can start decoding
        # Threshold scores are computed over length-normalized logprobs
        threshold_mask = (
            next_logprobs / new_beam_lengths.unsqueeze(-1)
            >= self.thresholds[decoding_step_id]
        )

        beam_and_tok_id = torch.argwhere(threshold_mask)
        new_beam_scores = next_logprobs[threshold_mask]

        beam_ids = beam_and_tok_id[:, 0]
        new_token_ids = beam_and_tok_id[:, 1:]

        new_beam_stem = beam[beam_ids]
        new_beam = torch.concatenate([new_beam_stem, new_token_ids], 1)

        just_finished_beams = new_beam[:, -1] == eos_token
        new_finished_beams: torch.Tensor = (
            finished_beams[beam_ids] | just_finished_beams
        )

        new_beam_lengths = new_beam_lengths[beam_ids]
        return dict(
            input_seq=input_seq,
            input_msk=input_msk,
            beam=new_beam,
            beam_scores=new_beam_scores,
            finished_beams=new_finished_beams,
            beam_lengths=new_beam_lengths,
        )

    @staticmethod
    def check_in_beam(true_seq: torch.Tensor, decode_data) -> Tuple[bool, Real]:
        true_len = (true_seq != -100).sum().item()
        dec: torch.Tensor = decode_data["beam"][:, 1 : (true_len + 1)]
        l_dec = dec.shape[-1]
        score_sort_idx = torch.argsort(decode_data["beam_scores"], descending=True)
        matches = torch.all(
            true_seq[:l_dec].unsqueeze(0) == dec[score_sort_idx].cpu(), 1
        )
        in_beam = torch.any(matches, 0).item()
        if in_beam:
            rank = torch.argwhere(matches).flatten().item()
        else:
            rank = float("nan")
        return in_beam, rank
