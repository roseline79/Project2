
# coding=utf-8

# the following codes are based on huggingface/transformers
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py

import os
os.environ.setdefault('WANDB_MODE', 'disabled')

import sys
import logging
import math
# from rouge.rouge_score import Ngrams
import torch
from torch.autograd.grad_mode import F
import transformers
import nltk
import numpy as np
# import time
import json
import warnings
from dataclasses import dataclass, field
import datasets
from datasets import load_dataset
#from evaluate import load as load_metric
from rouge_local.rouge_scorer import RougeScorer
from rouge_score import scoring
import nltk

from typing import Optional, List
# from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version
# from transformers.utils.versions import require_version
from transformers import (
    AutoConfig, AutoTokenizer,
    DataCollatorForSeq2Seq, HfArgumentParser, set_seed
)
# from transformers.deepspeed import is_deepspeed_zero3_enabled

from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments
# from extoracle.utils import greedy_selection
# from nltk.tokenize import sent_tokenize, word_tokenize
# from rouge_score import rouge_scorer, scoring

sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from prettytable import PrettyTable

# from transformers.models.bart.modeling_bart import BartForConditionalGeneration
# from transformers import Seq2SeqTrainer

from trainer_seq2seq import Seq2SeqTrainer
from modeling_bart import BartForConditionalGeneration

from nltk.tokenize import sent_tokenize

import nltk
nltk.download('punkt')

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

logger = logging.getLogger(__name__)

def count_parameters(model, all_param=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    if all_param:
        logger.warning(table)
    logger.warning(f"Total Trainable Params: {total_params}")
    return total_params


def _entropy_from_attention_tensor(attn_tensor: torch.Tensor, eps: float = 1e-10) -> float:
    attn = torch.clamp(attn_tensor.float(), min=eps)
    entropy = -torch.sum(attn * torch.log(attn), dim=-1)
    return float(entropy.mean().item())


def analyze_attention_entropy(model, dataset, output_dir, sample_size=100):
    logger.info(f"*** Analyzing Attention Entropy on {sample_size} samples ***")

    sample_size = min(sample_size, len(dataset))
    if sample_size == 0:
        entropy_file = os.path.join(output_dir, "attention_entropy_stats.json")
        no_data = {
            "mean_entropy": None,
            "std_entropy": None,
            "min_entropy": None,
            "max_entropy": None,
            "num_samples": 0,
            "message": "Dataset is empty; no entropy computed."
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(entropy_file, "w", encoding="utf-8") as f:
            json.dump(no_data, f, indent=2)
        logger.warning(f"No data available for entropy analysis. Saved to: {entropy_file}")
        return no_data

    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
    collected_entropies = []

    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[int(idx)]

            input_ids = torch.tensor([sample["input_ids"]], device=model.device)
            attention_mask = torch.tensor([sample["attention_mask"]], device=model.device)

            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_attentions": True,
                "return_dict": True,
            }

            if "labels" in sample:
                forward_kwargs["labels"] = torch.tensor([sample["labels"]], device=model.device)

            try:
                outputs = model(**forward_kwargs, compute_entropy=True)
            except TypeError:
                outputs = model(**forward_kwargs)

            attention_groups = []
            for key in ["encoder_attentions", "decoder_attentions", "cross_attentions", "attentions"]:
                value = getattr(outputs, key, None)
                if value is not None:
                    attention_groups.append(value)

            for group in attention_groups:
                for layer_attn in group:
                    if layer_attn is not None:
                        collected_entropies.append(_entropy_from_attention_tensor(layer_attn))

    entropy_file = os.path.join(output_dir, "attention_entropy_stats.json")
    os.makedirs(output_dir, exist_ok=True)

    if collected_entropies:
        entropy_stats = {
            "mean_entropy": float(np.mean(collected_entropies)),
            "std_entropy": float(np.std(collected_entropies)),
            "min_entropy": float(np.min(collected_entropies)),
            "max_entropy": float(np.max(collected_entropies)),
            "num_samples": int(len(collected_entropies))
        }
    else:
        entropy_stats = {
            "mean_entropy": None,
            "std_entropy": None,
            "min_entropy": None,
            "max_entropy": None,
            "num_samples": 0,
            "message": "No attention tensors were returned by the model."
        }

    with open(entropy_file, "w", encoding="utf-8") as f:
        json.dump(entropy_stats, f, indent=2)

    logger.info(f"Attention entropy stats saved to: {entropy_file}")
    return entropy_stats


def analyze_cross_attention_entropy(
    model,
    dataset,
    output_dir,
    sample_size=100,
    top_k=5,
    factuality_bin_count=5,
    factuality_field_candidates=("factuality_score", "factuality", "fact_score", "factual_score", "faithfulness_score"),
    eps: float = 1e-10,
):
    logger.info(f"*** Analyzing Cross-Attention Entropy on {sample_size} samples ***")

    sample_size = min(sample_size, len(dataset))
    cross_entropy_file = os.path.join(output_dir, "cross_attention_entropy_stats.json")

    if sample_size == 0:
        no_data = {
            "overall": {
                "mean_entropy": None,
                "std_entropy": None,
                "mean_normalized_entropy": None,
                "std_normalized_entropy": None,
                "mean_topk_mass": None,
                "std_topk_mass": None,
                "mean_gini": None,
                "std_gini": None,
                "num_entries": 0,
                "num_samples": 0,
                "eps": float(eps),
                "normalization": "entropy/log(n_source_tokens)",
                "entropy_log_base": "e",
                "secondary_metrics": {
                    "top_k": int(max(1, int(top_k))),
                    "gini_formula": "(2*sum_i(i*p_(i))-(n+1))/(n-1), p_(i) sorted ascending",
                },
            },
            "per_step": [],
            "per_head": [],
            "per_example": [],
            "factuality_bins": {
                "field_candidates": list(factuality_field_candidates),
                "bin_count": int(max(1, int(factuality_bin_count))),
                "bins": [],
                "message": "No factuality scores available.",
            },
            "message": "Dataset is empty; no cross-attention entropy computed.",
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(cross_entropy_file, "w", encoding="utf-8") as f:
            json.dump(no_data, f, indent=2)
        logger.warning(f"No data available for cross-attention entropy analysis. Saved to: {cross_entropy_file}")
        return no_data

    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)

    overall_entropy = []
    overall_normalized_entropy = []
    overall_topk_mass = []
    overall_gini = []
    per_step_values = {}
    per_head_values = {}
    per_example_values = []

    effective_top_k = max(1, int(top_k))

    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[int(idx)]

            factuality_score = None
            for field_name in factuality_field_candidates:
                if field_name in sample and sample[field_name] is not None:
                    try:
                        factuality_score = float(sample[field_name])
                    except (TypeError, ValueError):
                        factuality_score = None
                    break

            input_ids = torch.tensor([sample["input_ids"]], device=model.device)
            attention_mask = torch.tensor([sample["attention_mask"]], device=model.device)
            source_token_count = int(torch.sum(attention_mask[0]).item())

            if source_token_count <= 0:
                continue

            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_attentions": True,
                "return_dict": True,
            }

            if "labels" in sample:
                forward_kwargs["labels"] = torch.tensor([sample["labels"]], device=model.device)

            try:
                outputs = model(**forward_kwargs, compute_entropy=True)
            except TypeError:
                outputs = model(**forward_kwargs)

            cross_attentions = getattr(outputs, "cross_attentions", None)
            if not cross_attentions:
                continue

            norm_denom = max(math.log(max(source_token_count, 2)), eps)
            example_entropy = []
            example_normalized_entropy = []
            example_topk_mass = []
            example_gini = []

            for layer_idx, layer_attn in enumerate(cross_attentions, start=1):
                if layer_attn is None:
                    continue

                # layer_attn: [batch, num_heads, target_len, source_len]
                usable_source_len = min(source_token_count, int(layer_attn.shape[-1]))
                if usable_source_len <= 0:
                    continue

                attn = layer_attn[:, :, :, :usable_source_len].float()
                entropy_tensor = -torch.sum(attn * torch.log(attn + eps), dim=-1)
                normalized_entropy_tensor = torch.clamp(entropy_tensor / norm_denom, min=0.0, max=1.0)

                topk_k = min(effective_top_k, usable_source_len)
                topk_mass_tensor = torch.topk(attn, k=topk_k, dim=-1).values.sum(dim=-1)

                if usable_source_len > 1:
                    sorted_attn, _ = torch.sort(attn, dim=-1)
                    rank_idx = torch.arange(
                        1,
                        usable_source_len + 1,
                        device=attn.device,
                        dtype=attn.dtype,
                    ).view(1, 1, 1, usable_source_len)
                    gini_tensor = (
                        2.0 * torch.sum(rank_idx * sorted_attn, dim=-1) - (usable_source_len + 1.0)
                    ) / (usable_source_len - 1.0)
                    gini_tensor = torch.clamp(gini_tensor, min=0.0, max=1.0)
                else:
                    gini_tensor = torch.zeros_like(topk_mass_tensor)

                entropy_np = entropy_tensor.detach().cpu().numpy()
                normalized_np = normalized_entropy_tensor.detach().cpu().numpy()
                topk_np = topk_mass_tensor.detach().cpu().numpy()
                gini_np = gini_tensor.detach().cpu().numpy()

                overall_entropy.extend(entropy_np.reshape(-1).tolist())
                overall_normalized_entropy.extend(normalized_np.reshape(-1).tolist())
                overall_topk_mass.extend(topk_np.reshape(-1).tolist())
                overall_gini.extend(gini_np.reshape(-1).tolist())

                example_entropy.extend(entropy_np.reshape(-1).tolist())
                example_normalized_entropy.extend(normalized_np.reshape(-1).tolist())
                example_topk_mass.extend(topk_np.reshape(-1).tolist())
                example_gini.extend(gini_np.reshape(-1).tolist())

                num_heads = entropy_np.shape[1]
                num_steps = entropy_np.shape[2]

                for step_idx in range(num_steps):
                    step_key = int(step_idx + 1)
                    if step_key not in per_step_values:
                        per_step_values[step_key] = {
                            "entropy": [],
                            "normalized_entropy": [],
                            "topk_mass": [],
                            "gini": [],
                        }

                    step_entropy = entropy_np[:, :, step_idx].reshape(-1)
                    step_nentropy = normalized_np[:, :, step_idx].reshape(-1)
                    step_topk = topk_np[:, :, step_idx].reshape(-1)
                    step_gini = gini_np[:, :, step_idx].reshape(-1)
                    per_step_values[step_key]["entropy"].extend(step_entropy.tolist())
                    per_step_values[step_key]["normalized_entropy"].extend(step_nentropy.tolist())
                    per_step_values[step_key]["topk_mass"].extend(step_topk.tolist())
                    per_step_values[step_key]["gini"].extend(step_gini.tolist())

                for head_idx in range(num_heads):
                    head_key = f"L{layer_idx}_H{head_idx + 1}"
                    if head_key not in per_head_values:
                        per_head_values[head_key] = {
                            "entropy": [],
                            "normalized_entropy": [],
                            "topk_mass": [],
                            "gini": [],
                        }

                    head_entropy = entropy_np[:, head_idx, :].reshape(-1)
                    head_nentropy = normalized_np[:, head_idx, :].reshape(-1)
                    head_topk = topk_np[:, head_idx, :].reshape(-1)
                    head_gini = gini_np[:, head_idx, :].reshape(-1)
                    per_head_values[head_key]["entropy"].extend(head_entropy.tolist())
                    per_head_values[head_key]["normalized_entropy"].extend(head_nentropy.tolist())
                    per_head_values[head_key]["topk_mass"].extend(head_topk.tolist())
                    per_head_values[head_key]["gini"].extend(head_gini.tolist())

            if example_entropy:
                per_example_values.append(
                    {
                        "example_index": int(idx),
                        "factuality_score": float(factuality_score) if factuality_score is not None else None,
                        "count": int(len(example_entropy)),
                        "mean_entropy": float(np.mean(example_entropy)),
                        "std_entropy": float(np.std(example_entropy)),
                        "mean_normalized_entropy": float(np.mean(example_normalized_entropy)),
                        "std_normalized_entropy": float(np.std(example_normalized_entropy)),
                        "mean_topk_mass": float(np.mean(example_topk_mass)),
                        "std_topk_mass": float(np.std(example_topk_mass)),
                        "mean_gini": float(np.mean(example_gini)),
                        "std_gini": float(np.std(example_gini)),
                    }
                )

    os.makedirs(output_dir, exist_ok=True)

    if not overall_entropy:
        cross_entropy_stats = {
            "overall": {
                "mean_entropy": None,
                "std_entropy": None,
                "mean_normalized_entropy": None,
                "std_normalized_entropy": None,
                "mean_topk_mass": None,
                "std_topk_mass": None,
                "mean_gini": None,
                "std_gini": None,
                "num_entries": 0,
                "num_samples": int(sample_size),
                "eps": float(eps),
                "normalization": "entropy/log(n_source_tokens)",
                "entropy_log_base": "e",
                "secondary_metrics": {
                    "top_k": int(effective_top_k),
                    "gini_formula": "(2*sum_i(i*p_(i))-(n+1))/(n-1), p_(i) sorted ascending",
                },
            },
            "per_step": [],
            "per_head": [],
            "per_example": [],
            "factuality_bins": {
                "field_candidates": list(factuality_field_candidates),
                "bin_count": int(max(1, int(factuality_bin_count))),
                "bins": [],
                "message": "No factuality scores available.",
            },
            "phase_wise": {
                "boundaries": {
                    "early": [0.0, 0.2],
                    "mid": [0.2, 0.8],
                    "late": [0.8, 1.0],
                    "progress_definition": "(step-1)/max(num_steps-1,1)",
                },
                "phases": {
                    "early": None,
                    "mid": None,
                    "late": None,
                },
            },
            "message": "No cross-attention tensors were returned by the model.",
        }
    else:
        per_step_stats = []
        for step_idx in sorted(per_step_values.keys()):
            step_data = per_step_values[step_idx]
            per_step_stats.append(
                {
                    "step": int(step_idx),
                    "count": int(len(step_data["entropy"])),
                    "mean_entropy": float(np.mean(step_data["entropy"])),
                    "std_entropy": float(np.std(step_data["entropy"])),
                    "mean_normalized_entropy": float(np.mean(step_data["normalized_entropy"])),
                    "std_normalized_entropy": float(np.std(step_data["normalized_entropy"])),
                    "mean_topk_mass": float(np.mean(step_data["topk_mass"])),
                    "std_topk_mass": float(np.std(step_data["topk_mass"])),
                    "mean_gini": float(np.mean(step_data["gini"])),
                    "std_gini": float(np.std(step_data["gini"])),
                }
            )

        max_step_observed = max(per_step_values.keys()) if per_step_values else 0
        phase_values = {
            "early": {"entropy": [], "normalized_entropy": [], "topk_mass": [], "gini": [], "steps": []},
            "mid": {"entropy": [], "normalized_entropy": [], "topk_mass": [], "gini": [], "steps": []},
            "late": {"entropy": [], "normalized_entropy": [], "topk_mass": [], "gini": [], "steps": []},
        }

        for step_idx, step_data in per_step_values.items():
            if max_step_observed <= 1:
                progress = 0.0
            else:
                progress = (float(step_idx) - 1.0) / float(max_step_observed - 1)

            if progress < 0.2:
                phase_key = "early"
            elif progress < 0.8:
                phase_key = "mid"
            else:
                phase_key = "late"

            phase_values[phase_key]["steps"].append(int(step_idx))
            phase_values[phase_key]["entropy"].extend(step_data["entropy"])
            phase_values[phase_key]["normalized_entropy"].extend(step_data["normalized_entropy"])
            phase_values[phase_key]["topk_mass"].extend(step_data["topk_mass"])
            phase_values[phase_key]["gini"].extend(step_data["gini"])

        phase_wise_stats = {
            "boundaries": {
                "early": [0.0, 0.2],
                "mid": [0.2, 0.8],
                "late": [0.8, 1.0],
                "progress_definition": "(step-1)/max(num_steps-1,1)",
            },
            "phases": {},
        }

        for phase_name in ["early", "mid", "late"]:
            phase_data = phase_values[phase_name]
            if phase_data["entropy"]:
                phase_wise_stats["phases"][phase_name] = {
                    "num_steps": int(len(phase_data["steps"])),
                    "step_indices": sorted(phase_data["steps"]),
                    "count": int(len(phase_data["entropy"])),
                    "mean_entropy": float(np.mean(phase_data["entropy"])),
                    "std_entropy": float(np.std(phase_data["entropy"])),
                    "mean_normalized_entropy": float(np.mean(phase_data["normalized_entropy"])),
                    "std_normalized_entropy": float(np.std(phase_data["normalized_entropy"])),
                    "mean_topk_mass": float(np.mean(phase_data["topk_mass"])),
                    "std_topk_mass": float(np.std(phase_data["topk_mass"])),
                    "mean_gini": float(np.mean(phase_data["gini"])),
                    "std_gini": float(np.std(phase_data["gini"])),
                }
            else:
                phase_wise_stats["phases"][phase_name] = None

        per_head_stats = []
        for head_key in sorted(per_head_values.keys()):
            head_data = per_head_values[head_key]
            per_head_stats.append(
                {
                    "head": head_key,
                    "count": int(len(head_data["entropy"])),
                    "mean_entropy": float(np.mean(head_data["entropy"])),
                    "std_entropy": float(np.std(head_data["entropy"])),
                    "mean_normalized_entropy": float(np.mean(head_data["normalized_entropy"])),
                    "std_normalized_entropy": float(np.std(head_data["normalized_entropy"])),
                    "mean_topk_mass": float(np.mean(head_data["topk_mass"])),
                    "std_topk_mass": float(np.std(head_data["topk_mass"])),
                    "mean_gini": float(np.mean(head_data["gini"])),
                    "std_gini": float(np.std(head_data["gini"])),
                }
            )

        per_example_stats = sorted(per_example_values, key=lambda x: x["example_index"])

        per_example_mean_nentropy = [item["mean_normalized_entropy"] for item in per_example_stats]
        per_example_mean_entropy = [item["mean_entropy"] for item in per_example_stats]
        per_example_mean_topk = [item["mean_topk_mass"] for item in per_example_stats]
        per_example_mean_gini = [item["mean_gini"] for item in per_example_stats]

        effective_factuality_bin_count = max(1, int(factuality_bin_count))
        factuality_examples = [
            item for item in per_example_stats
            if item.get("factuality_score") is not None and np.isfinite(item.get("factuality_score"))
        ]

        factuality_bins_stats = {
            "field_candidates": list(factuality_field_candidates),
            "bin_count": int(effective_factuality_bin_count),
            "bins": [],
        }

        if factuality_examples:
            factuality_scores = np.array([float(item["factuality_score"]) for item in factuality_examples], dtype=np.float32)
            score_min = float(np.min(factuality_scores))
            score_max = float(np.max(factuality_scores))

            if score_min >= 0.0 and score_max <= 1.0:
                bin_edges = np.linspace(0.0, 1.0, effective_factuality_bin_count + 1)
                factuality_bins_stats["range"] = [0.0, 1.0]
            else:
                if score_min == score_max:
                    score_min = score_min - 0.5
                    score_max = score_max + 0.5
                bin_edges = np.linspace(score_min, score_max, effective_factuality_bin_count + 1)
                factuality_bins_stats["range"] = [float(score_min), float(score_max)]

            for bin_idx in range(effective_factuality_bin_count):
                left = float(bin_edges[bin_idx])
                right = float(bin_edges[bin_idx + 1])
                if bin_idx == effective_factuality_bin_count - 1:
                    in_bin = [
                        item for item in factuality_examples
                        if left <= float(item["factuality_score"]) <= right
                    ]
                else:
                    in_bin = [
                        item for item in factuality_examples
                        if left <= float(item["factuality_score"]) < right
                    ]

                if in_bin:
                    bin_values = [float(item["mean_normalized_entropy"]) for item in in_bin]
                    bin_stat = {
                        "bin_index": int(bin_idx),
                        "left": left,
                        "right": right,
                        "count": int(len(bin_values)),
                        "mean_normalized_entropy": float(np.mean(bin_values)),
                        "std_normalized_entropy": float(np.std(bin_values)),
                    }
                else:
                    bin_stat = {
                        "bin_index": int(bin_idx),
                        "left": left,
                        "right": right,
                        "count": 0,
                        "mean_normalized_entropy": None,
                        "std_normalized_entropy": None,
                    }
                factuality_bins_stats["bins"].append(bin_stat)
        else:
            factuality_bins_stats["message"] = "No factuality scores available."

        cross_entropy_stats = {
            "overall": {
                "mean_entropy": float(np.mean(overall_entropy)),
                "std_entropy": float(np.std(overall_entropy)),
                "min_entropy": float(np.min(overall_entropy)),
                "max_entropy": float(np.max(overall_entropy)),
                "mean_normalized_entropy": float(np.mean(overall_normalized_entropy)),
                "std_normalized_entropy": float(np.std(overall_normalized_entropy)),
                "min_normalized_entropy": float(np.min(overall_normalized_entropy)),
                "max_normalized_entropy": float(np.max(overall_normalized_entropy)),
                "mean_topk_mass": float(np.mean(overall_topk_mass)),
                "std_topk_mass": float(np.std(overall_topk_mass)),
                "min_topk_mass": float(np.min(overall_topk_mass)),
                "max_topk_mass": float(np.max(overall_topk_mass)),
                "mean_gini": float(np.mean(overall_gini)),
                "std_gini": float(np.std(overall_gini)),
                "min_gini": float(np.min(overall_gini)),
                "max_gini": float(np.max(overall_gini)),
                "num_entries": int(len(overall_entropy)),
                "num_samples": int(sample_size),
                "eps": float(eps),
                "normalization": "entropy/log(n_source_tokens)",
                "entropy_log_base": "e",
                "secondary_metrics": {
                    "top_k": int(effective_top_k),
                    "gini_formula": "(2*sum_i(i*p_(i))-(n+1))/(n-1), p_(i) sorted ascending",
                },
                "per_example_summary": {
                    "num_examples": int(len(per_example_stats)),
                    "mean_of_example_mean_entropy": float(np.mean(per_example_mean_entropy)) if per_example_mean_entropy else None,
                    "std_of_example_mean_entropy": float(np.std(per_example_mean_entropy)) if per_example_mean_entropy else None,
                    "mean_of_example_mean_normalized_entropy": float(np.mean(per_example_mean_nentropy)) if per_example_mean_nentropy else None,
                    "std_of_example_mean_normalized_entropy": float(np.std(per_example_mean_nentropy)) if per_example_mean_nentropy else None,
                    "mean_of_example_mean_topk_mass": float(np.mean(per_example_mean_topk)) if per_example_mean_topk else None,
                    "std_of_example_mean_topk_mass": float(np.std(per_example_mean_topk)) if per_example_mean_topk else None,
                    "mean_of_example_mean_gini": float(np.mean(per_example_mean_gini)) if per_example_mean_gini else None,
                    "std_of_example_mean_gini": float(np.std(per_example_mean_gini)) if per_example_mean_gini else None,
                },
            },
            "per_step": per_step_stats,
            "per_head": per_head_stats,
            "per_example": per_example_stats,
            "factuality_bins": factuality_bins_stats,
            "phase_wise": phase_wise_stats,
        }

    with open(cross_entropy_file, "w", encoding="utf-8") as f:
        json.dump(cross_entropy_stats, f, indent=2)

    logger.info(f"Cross-attention entropy stats saved to: {cross_entropy_file}")
    return cross_entropy_stats


def analyze_evidence_concentration_over_decoding(
    model,
    dataset,
    output_dir,
    sample_size=100,
    max_length=None,
    num_beams=None,
    length_penalty=None,
    no_repeat_ngram_size=None,
    do_sample=False,
    top_p=None,
    temperature=None,
    eps: float = 1e-10,
):
    logger.info(f"*** Analyzing Evidential Concentration over Decoding Steps on {sample_size} samples ***")

    sample_size = min(sample_size, len(dataset))
    if sample_size == 0:
        evidence_file = os.path.join(output_dir, "evidence_concentration_by_step.json")
        no_data = {
            "overall": {
                "mean_entropy": None,
                "mean_concentration": None,
                "mean_total_concentration": None,
                "num_samples": 0,
                "num_steps_observed": 0,
            },
            "per_step": [],
            "message": "Dataset is empty; no evidential decoding metrics computed.",
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(evidence_file, "w", encoding="utf-8") as f:
            json.dump(no_data, f, indent=2)
        logger.warning(f"No data available for evidential analysis. Saved to: {evidence_file}")
        return no_data

    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)

    per_step_values = {}
    overall_entropy = []
    overall_concentration = []
    overall_total_concentration = []
    overall_top1_prob = []
    overall_top5_mass = []
    vocab_size = None

    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[int(idx)]
            input_ids = torch.tensor([sample["input_ids"]], device=model.device)

            generate_kwargs = {
                "input_ids": input_ids,
                "return_dict_in_generate": True,
                "output_scores": True,
                "do_sample": bool(do_sample),
            }

            if "attention_mask" in sample:
                generate_kwargs["attention_mask"] = torch.tensor([sample["attention_mask"]], device=model.device)
            if max_length is not None:
                generate_kwargs["max_length"] = max_length
            if num_beams is not None:
                generate_kwargs["num_beams"] = num_beams
            if length_penalty is not None:
                generate_kwargs["length_penalty"] = length_penalty
            if no_repeat_ngram_size is not None:
                generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
            if top_p is not None:
                generate_kwargs["top_p"] = top_p
            if temperature is not None:
                generate_kwargs["temperature"] = temperature

            generated = model.generate(**generate_kwargs)
            scores = generated.scores if generated is not None else None
            if not scores:
                continue

            for step_idx, step_logits in enumerate(scores, start=1):
                logits = step_logits[0].float()
                probs = torch.softmax(logits, dim=-1)
                if vocab_size is None:
                    vocab_size = int(probs.shape[-1])

                safe_probs = torch.clamp(probs, min=eps)
                entropy = -torch.sum(safe_probs * torch.log(safe_probs)).item()
                norm_denom = max(math.log(max(vocab_size, 2)), eps)
                concentration = 1.0 - (entropy / norm_denom)

                evidence = torch.exp(logits - torch.max(logits))
                alpha = evidence + 1.0
                total_concentration = torch.sum(alpha).item()

                top_k = min(5, probs.numel())
                top5_mass = torch.topk(probs, k=top_k, dim=-1).values.sum().item()
                top1_prob = torch.max(probs).item()

                if step_idx not in per_step_values:
                    per_step_values[step_idx] = {
                        "entropy": [],
                        "concentration": [],
                        "total_concentration": [],
                        "top1_prob": [],
                        "top5_mass": [],
                    }

                per_step_values[step_idx]["entropy"].append(entropy)
                per_step_values[step_idx]["concentration"].append(concentration)
                per_step_values[step_idx]["total_concentration"].append(total_concentration)
                per_step_values[step_idx]["top1_prob"].append(top1_prob)
                per_step_values[step_idx]["top5_mass"].append(top5_mass)

                overall_entropy.append(entropy)
                overall_concentration.append(concentration)
                overall_total_concentration.append(total_concentration)
                overall_top1_prob.append(top1_prob)
                overall_top5_mass.append(top5_mass)

    evidence_file = os.path.join(output_dir, "evidence_concentration_by_step.json")
    os.makedirs(output_dir, exist_ok=True)

    if not overall_entropy:
        evidence_stats = {
            "overall": {
                "mean_entropy": None,
                "mean_concentration": None,
                "mean_total_concentration": None,
                "num_samples": 0,
                "num_steps_observed": 0,
            },
            "per_step": [],
            "message": "No generation scores were returned by model.generate(output_scores=True).",
        }
    else:
        per_step_stats = []
        for step_idx in sorted(per_step_values.keys()):
            step_data = per_step_values[step_idx]
            per_step_stats.append(
                {
                    "step": int(step_idx),
                    "count": int(len(step_data["entropy"])),
                    "mean_entropy": float(np.mean(step_data["entropy"])),
                    "std_entropy": float(np.std(step_data["entropy"])),
                    "mean_concentration": float(np.mean(step_data["concentration"])),
                    "std_concentration": float(np.std(step_data["concentration"])),
                    "mean_total_concentration": float(np.mean(step_data["total_concentration"])),
                    "std_total_concentration": float(np.std(step_data["total_concentration"])),
                    "mean_top1_prob": float(np.mean(step_data["top1_prob"])),
                    "mean_top5_mass": float(np.mean(step_data["top5_mass"])),
                }
            )

        evidence_stats = {
            "overall": {
                "mean_entropy": float(np.mean(overall_entropy)),
                "std_entropy": float(np.std(overall_entropy)),
                "mean_concentration": float(np.mean(overall_concentration)),
                "std_concentration": float(np.std(overall_concentration)),
                "mean_total_concentration": float(np.mean(overall_total_concentration)),
                "std_total_concentration": float(np.std(overall_total_concentration)),
                "mean_top1_prob": float(np.mean(overall_top1_prob)),
                "mean_top5_mass": float(np.mean(overall_top5_mass)),
                "num_samples": int(sample_size),
                "num_steps_observed": int(len(per_step_stats)),
                "vocab_size": int(vocab_size) if vocab_size is not None else None,
                "entropy_log_base": "e",
                "concentration_formula": "1 - H/log(V)",
                "dirichlet_proxy": "alpha = exp(logits - max(logits)) + 1",
            },
            "per_step": per_step_stats,
        }

    with open(evidence_file, "w", encoding="utf-8") as f:
        json.dump(evidence_stats, f, indent=2)

    logger.info(f"Evidential decoding stats saved to: {evidence_file}")
    return evidence_stats


def plot_evidence_concentration_curve(evidence_stats, output_dir, file_name="evidence_concentration_by_step.png"):
    if plt is None:
        warnings.warn("matplotlib is not installed; skipping evidence concentration plot generation.")
        logger.warning("matplotlib is not installed; skipping evidence concentration plot generation.")
        return None

    per_step = evidence_stats.get("per_step", []) if isinstance(evidence_stats, dict) else []
    if not per_step:
        logger.warning("No per-step evidential data found; skipping plot generation.")
        return None

    steps = [int(item["step"]) for item in per_step]
    mean_concentration = [float(item["mean_concentration"]) for item in per_step]
    std_concentration = [float(item.get("std_concentration", 0.0)) for item in per_step]
    mean_entropy = [float(item["mean_entropy"]) for item in per_step]

    lower = [max(0.0, m - s) for m, s in zip(mean_concentration, std_concentration)]
    upper = [min(1.0, m + s) for m, s in zip(mean_concentration, std_concentration)]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(steps, mean_concentration, marker="o", linewidth=2, label="Mean concentration")
    ax1.fill_between(steps, lower, upper, alpha=0.2, label="±1 std concentration")
    ax1.set_xlabel("Decoding step")
    ax1.set_ylabel("Concentration (1 - H/log(V))")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps, mean_entropy, marker="x", linewidth=1.5, color="tab:red", label="Mean entropy")
    ax2.set_ylabel("Entropy (nats)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Evidential Concentration over Decoding Steps")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Evidential concentration plot saved to: {output_path}")
    return output_path


def plot_training_convergence(
    log_history,
    output_dir,
    file_name="training_convergence.png",
    smoothing_window=7,
):
    if plt is None:
        warnings.warn("matplotlib is not installed; skipping training convergence plot generation.")
        logger.warning("matplotlib is not installed; skipping training convergence plot generation.")
        return None

    if not isinstance(log_history, list) or not log_history:
        logger.warning("Trainer log history is empty; skipping training convergence plot generation.")
        return None

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for item in log_history:
        if not isinstance(item, dict):
            continue

        step = item.get("step")
        if step is None:
            continue

        try:
            step = float(step)
        except (TypeError, ValueError):
            continue

        if "loss" in item and item.get("loss") is not None:
            try:
                train_steps.append(step)
                train_losses.append(float(item["loss"]))
            except (TypeError, ValueError):
                pass

        if "eval_loss" in item and item.get("eval_loss") is not None:
            try:
                eval_steps.append(step)
                eval_losses.append(float(item["eval_loss"]))
            except (TypeError, ValueError):
                pass

    if not train_losses and not eval_losses:
        logger.warning("No loss or eval_loss entries found in trainer log history; skipping convergence plot generation.")
        return None

    def _moving_average(values, window_size):
        if not values:
            return []
        window_size = max(1, int(window_size))
        if window_size == 1:
            return list(values)

        arr = np.asarray(values, dtype=np.float32)
        kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
        padded = np.pad(arr, (window_size - 1, 0), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.tolist()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    fig, ax = plt.subplots(figsize=(10, 5))

    if train_losses:
        ax.plot(train_steps, train_losses, marker="o", linewidth=1.2, alpha=0.35, label="Train loss (raw)")
        train_smooth = _moving_average(train_losses, smoothing_window)
        ax.plot(train_steps, train_smooth, linewidth=2.2, label=f"Train loss (MA{max(1, int(smoothing_window))})")

    if eval_losses:
        ax.plot(eval_steps, eval_losses, marker="s", linewidth=1.2, alpha=0.45, label="Eval loss (raw)")
        eval_smooth = _moving_average(eval_losses, max(1, int(smoothing_window // 2)))
        ax.plot(eval_steps, eval_smooth, linewidth=2.0, linestyle="--", label=f"Eval loss (MA{max(1, int(smoothing_window // 2))})")

    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Training convergence plot saved to: {output_path}")
    return output_path


def plot_cross_attention_normalized_entropy_curve(
    cross_entropy_stats,
    output_dir,
    file_name="cross_attention_normalized_entropy_by_step.png",
):
    if plt is None:
        warnings.warn("matplotlib is not installed; skipping cross-attention entropy plot generation.")
        logger.warning("matplotlib is not installed; skipping cross-attention entropy plot generation.")
        return None

    per_step = cross_entropy_stats.get("per_step", []) if isinstance(cross_entropy_stats, dict) else []
    if not per_step:
        logger.warning("No per-step cross-attention entropy data found; skipping plot generation.")
        return None

    steps = [int(item["step"]) for item in per_step]
    mean_nentropy = [float(item["mean_normalized_entropy"]) for item in per_step]
    std_nentropy = [float(item.get("std_normalized_entropy", 0.0)) for item in per_step]

    lower = [max(0.0, m - s) for m, s in zip(mean_nentropy, std_nentropy)]
    upper = [min(1.0, m + s) for m, s in zip(mean_nentropy, std_nentropy)]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, mean_nentropy, marker="o", linewidth=2, label="Mean normalized entropy")
    ax.fill_between(steps, lower, upper, alpha=0.2, label="±1 std")
    ax.set_xlabel("Decoding step")
    ax.set_ylabel("Normalized cross-attention entropy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.title("Per-step Normalized Cross-Attention Entropy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Cross-attention entropy plot saved to: {output_path}")
    return output_path


def plot_cross_attention_step_secondary_metrics(
    cross_entropy_stats,
    output_dir,
    file_name="cross_attention_step_secondary_metrics.png",
):
    if plt is None:
        warnings.warn("matplotlib is not installed; skipping cross-attention secondary metrics plot generation.")
        logger.warning("matplotlib is not installed; skipping cross-attention secondary metrics plot generation.")
        return None

    per_step = cross_entropy_stats.get("per_step", []) if isinstance(cross_entropy_stats, dict) else []
    if not per_step:
        logger.warning("No per-step cross-attention data found; skipping secondary metrics plot generation.")
        return None

    steps = [int(item["step"]) for item in per_step]

    mean_nentropy = [float(item.get("mean_normalized_entropy", 0.0)) for item in per_step]
    std_nentropy = [float(item.get("std_normalized_entropy", 0.0)) for item in per_step]

    mean_topk = [float(item.get("mean_topk_mass", 0.0)) for item in per_step]
    std_topk = [float(item.get("std_topk_mass", 0.0)) for item in per_step]

    mean_gini = [float(item.get("mean_gini", 0.0)) for item in per_step]
    std_gini = [float(item.get("std_gini", 0.0)) for item in per_step]

    overall = cross_entropy_stats.get("overall", {}) if isinstance(cross_entropy_stats, dict) else {}
    secondary = overall.get("secondary_metrics", {}) if isinstance(overall, dict) else {}
    chosen_top_k = secondary.get("top_k") if isinstance(secondary, dict) else None

    output_file_name = file_name
    if chosen_top_k is not None:
        base_name, ext = os.path.splitext(file_name)
        output_file_name = f"{base_name}_topk{chosen_top_k}{ext}"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file_name)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharex=True)

    def _plot_metric(ax, mean_values, std_values, title, ylabel):
        lower = [max(0.0, m - s) for m, s in zip(mean_values, std_values)]
        upper = [min(1.0, m + s) for m, s in zip(mean_values, std_values)]
        ax.plot(steps, mean_values, marker="o", linewidth=1.8)
        ax.fill_between(steps, lower, upper, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    _plot_metric(
        axes[0],
        mean_nentropy,
        std_nentropy,
        "Norm Entropy",
        "Mean",
    )
    _plot_metric(
        axes[1],
        mean_topk,
        std_topk,
        "Top-k Mass",
        "Mean",
    )
    _plot_metric(
        axes[2],
        mean_gini,
        std_gini,
        "Gini",
        "Mean",
    )

    if chosen_top_k is None:
        suptitle = "Per-step Cross-Attention Metrics"
    else:
        suptitle = f"Per-step Cross-Attention Metrics (Top-k={chosen_top_k})"

    plt.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Cross-attention secondary metrics plot saved to: {output_path}")
    return output_path


def plot_cross_attention_phase_normalized_entropy(
    cross_entropy_stats,
    output_dir,
    file_name="cross_attention_phase_normalized_entropy.png",
):
    if plt is None:
        warnings.warn("matplotlib is not installed; skipping cross-attention phase plot generation.")
        logger.warning("matplotlib is not installed; skipping cross-attention phase plot generation.")
        return None

    phase_wise = cross_entropy_stats.get("phase_wise", {}) if isinstance(cross_entropy_stats, dict) else {}
    phases = phase_wise.get("phases", {}) if isinstance(phase_wise, dict) else {}
    if not isinstance(phases, dict) or not phases:
        logger.warning("No phase-wise cross-attention data found; skipping phase plot generation.")
        return None

    ordered_phase_names = ["early", "mid", "late"]
    labels = []
    values = []
    errors = []
    for phase_name in ordered_phase_names:
        phase_data = phases.get(phase_name)
        if isinstance(phase_data, dict) and phase_data.get("mean_normalized_entropy") is not None:
            labels.append(phase_name.capitalize())
            values.append(float(phase_data.get("mean_normalized_entropy", 0.0)))
            errors.append(float(phase_data.get("std_normalized_entropy", 0.0)))

    if not labels:
        logger.warning("Phase-wise metrics exist but have no normalized entropy values; skipping phase plot generation.")
        return None

    overall = cross_entropy_stats.get("overall", {}) if isinstance(cross_entropy_stats, dict) else {}
    secondary = overall.get("secondary_metrics", {}) if isinstance(overall, dict) else {}
    chosen_top_k = secondary.get("top_k") if isinstance(secondary, dict) else None

    output_file_name = file_name
    if chosen_top_k is not None:
        base_name, ext = os.path.splitext(file_name)
        output_file_name = f"{base_name}_topk{chosen_top_k}{ext}"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file_name)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=errors, capsize=4, width=0.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(0.98, value + 0.03),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean normalized entropy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Phase-wise Normalized Cross-Attention Entropy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Cross-attention phase plot saved to: {output_path}")
    return output_path


def plot_cross_attention_entropy_by_factuality_bins(
    cross_entropy_stats,
    output_dir,
    file_name="cross_attention_entropy_by_factuality_bins.png",
):
    if plt is None:
        warnings.warn("matplotlib is not installed; skipping factuality-bin entropy plot generation.")
        logger.warning("matplotlib is not installed; skipping factuality-bin entropy plot generation.")
        return None

    factuality_bins = cross_entropy_stats.get("factuality_bins", {}) if isinstance(cross_entropy_stats, dict) else {}
    bins = factuality_bins.get("bins", []) if isinstance(factuality_bins, dict) else []
    if not bins:
        logger.warning("No factuality-bin data found; skipping factuality-bin entropy plot generation.")
        return None

    labels = []
    means = []
    stds = []
    counts = []
    for item in bins:
        mean_val = item.get("mean_normalized_entropy")
        if mean_val is None:
            continue
        left = float(item.get("left", 0.0))
        right = float(item.get("right", 0.0))
        labels.append(f"[{left:.2f},{right:.2f}{']' if item.get('bin_index') == len(bins)-1 else ')'}")
        means.append(float(mean_val))
        stds.append(float(item.get("std_normalized_entropy", 0.0) or 0.0))
        counts.append(int(item.get("count", 0)))

    if not labels:
        logger.warning("Factuality bins exist but no non-empty bins found; skipping plot generation.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    fig, ax = plt.subplots(figsize=(max(6.5, 1.0 * len(labels) + 2.5), 4.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.7)

    for bar, mean_val, count in zip(bars, means, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(0.98, mean_val + 0.03),
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Factuality bins")
    ax.set_ylabel("Mean normalized entropy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Entropy vs Factuality Bins")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Factuality-bin entropy plot saved to: {output_path}")
    return output_path


def plot_cross_attention_head_specialization_heatmap(
    cross_entropy_stats,
    output_dir,
    file_name="cross_attention_head_specialization_heatmap.png",
):
    if plt is None:
        warnings.warn("matplotlib is not installed; skipping head-specialization heatmap generation.")
        logger.warning("matplotlib is not installed; skipping head-specialization heatmap generation.")
        return None

    per_head = cross_entropy_stats.get("per_head", []) if isinstance(cross_entropy_stats, dict) else []
    if not per_head:
        logger.warning("No per-head data found; skipping head-specialization heatmap generation.")
        return None

    parsed_items = []
    max_layer = 0
    max_head = 0
    for item in per_head:
        head_key = str(item.get("head", ""))
        if not head_key.startswith("L") or "_H" not in head_key:
            continue
        try:
            layer_str, head_str = head_key.split("_H")
            layer_idx = int(layer_str[1:])
            head_idx = int(head_str)
            value = float(item.get("mean_normalized_entropy", np.nan))
        except (TypeError, ValueError):
            continue

        if layer_idx <= 0 or head_idx <= 0:
            continue
        parsed_items.append((layer_idx, head_idx, value))
        max_layer = max(max_layer, layer_idx)
        max_head = max(max_head, head_idx)

    if not parsed_items:
        logger.warning("Per-head keys could not be parsed into layer/head indices; skipping heatmap generation.")
        return None

    matrix = np.full((max_layer, max_head), np.nan, dtype=np.float32)
    for layer_idx, head_idx, value in parsed_items:
        matrix[layer_idx - 1, head_idx - 1] = value

    masked_matrix = np.ma.masked_invalid(matrix)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    fig_w = max(6.5, 0.35 * max_head + 3.0)
    fig_h = max(4.5, 0.35 * max_layer + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(masked_matrix, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)

    if max_head <= 16:
        x_ticks = np.arange(max_head)
    else:
        x_ticks = np.arange(0, max_head, max(1, max_head // 12))
    if max_layer <= 16:
        y_ticks = np.arange(max_layer)
    else:
        y_ticks = np.arange(0, max_layer, max(1, max_layer // 12))

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(i + 1) for i in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(i + 1) for i in y_ticks])

    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title("Head Specialization Heatmap (Mean NEnt)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean normalized entropy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Head-specialization heatmap saved to: {output_path}")
    return output_path


def plot_cross_attention_per_head_normalized_entropy(
    cross_entropy_stats,
    output_dir,
    file_name="cross_attention_normalized_entropy_per_head.png",
    top_k=20,
):
    if plt is None:
        warnings.warn("matplotlib is not installed; skipping per-head cross-attention entropy plot generation.")
        logger.warning("matplotlib is not installed; skipping per-head cross-attention entropy plot generation.")
        return None

    per_head = cross_entropy_stats.get("per_head", []) if isinstance(cross_entropy_stats, dict) else []
    if not per_head:
        logger.warning("No per-head cross-attention entropy data found; skipping plot generation.")
        return None

    sorted_heads = sorted(per_head, key=lambda x: float(x.get("mean_normalized_entropy", 0.0)), reverse=True)
    if top_k is not None:
        top_k = max(1, int(top_k))
        sorted_heads = sorted_heads[:top_k]

    if not sorted_heads:
        logger.warning("No cross-attention heads available after applying top_k; skipping plot generation.")
        return None

    labels = [str(item.get("head", "")) for item in sorted_heads]
    values = [float(item.get("mean_normalized_entropy", 0.0)) for item in sorted_heads]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    fig_width = max(8, min(20, 0.35 * len(labels) + 4))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(range(len(values)), values, width=0.8)
    ax.set_xlabel("Head (layer_head)")
    ax.set_ylabel("Mean normalized entropy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    if top_k is None:
        title = "Per-head Mean Normalized Cross-Attention Entropy"
    else:
        title = f"Top-{len(values)} Per-head Mean Normalized Cross-Attention Entropy"

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Per-head cross-attention entropy plot saved to: {output_path}")
    return output_path


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use a `sortish sampler` or not. Only possible if the underlying datasets are `Seq2SeqDataset` for
        now but will become generally available in the near future.

        It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness for
        the training set.
    predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    generation_max_length (:obj:`int`, `optional`):
        The :obj:`max_length` to use on each evaluation loop when :obj:`predict_with_generate=True`. Will default to
        the :obj:`max_length` value of the model configuration.
    generation_num_beams (:obj:`int`, `optional`):
        The :obj:`num_beams` to use on each evaluation loop when :obj:`predict_with_generate=True`. Will default to the
        :obj:`num_beams` value of the model configuration.
    """

    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    generation_min_length: Optional[int] = field(
        default=None,
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `num_beams` value of the model configuration."
        },
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether use do_sample"
        }
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "top-p nucleus sampling"
        }
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={
            "help": "temperature"
        }
    )
    evaltest_generation_max_length: Optional[int] = field(
        default=None,
    )
    evaltest_generation_num_beams: Optional[int] = field(
        default=None,
    )
    length_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Length penalty passed to model.generate during evaluation/prediction."
        },
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set > 0, all ngrams of this size can only occur once during generation."
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=768,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    use_sampleprompt: Optional[bool] = field(
        default=False
    )
    sampleprompt: Optional[str] = field(
        default=None
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

def run_bart():
    logger.info('torch version:', torch.__version__, 'cuda available:',torch.cuda.is_available())
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch_device == 'cuda':
        logger.info('device count:', torch.cuda.device_count(), torch_device)
        logger.info('cuda device:', torch.cuda.get_device_name(0))

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.sampleprompt = data_args.sampleprompt
    print(f'sampleprompt {data_args.sampleprompt}')
    
    if data_args.sampleprompt == "sampleprom4":
        json_name = "DISCHARGE"
        if "ECHO_" in data_args.dataset_name:
            json_name = "ECHO"
        elif "RADIOLOGY_" in data_args.dataset_name:
            json_name = "RADIOLOGY"
        
        with open("./dataset/" + json_name + "_cluster.json", "r", encoding="utf-8") as read_file:
            cluster_classes = json.load(read_file)
            print(f"json {json_name} _cluster.json loaded, key size = {len(cluster_classes.keys())}")
            training_args.cluster_classes = cluster_classes

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()

    dataset_loglevel = 20
    transformers_loglevel = 30

    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(dataset_loglevel)
    transformers.utils.logging.set_verbosity(transformers_loglevel)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16} (fp16: {training_args.fp16}, bf16: {training_args.bf16})"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)
    
    # if data_args.dataset_name is not None:
    #     raw_datasets = load_dataset(
    #         data_args.dataset_name, 
    #         data_args.dataset_config_name, 
    #         cache_dir=model_args.cache_dir
    #     )
    # else:
    #     data_files = {}
    #     if data_args.train_file is not None:
    #         data_files["train"] = data_args.train_file
    #         extension = data_args.train_file.split(".")[-1]
    #     if data_args.validation_file is not None:
    #         data_files["validation"] = data_args.validation_file
    #         extension = data_args.validation_file.split(".")[-1]
    #     if data_args.test_file is not None:
    #         data_files["test"] = data_args.test_file
    #         extension = data_args.test_file.split(".")[-1]
    #     raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # if training_args.do_train:
    #     column_names = raw_datasets["train"].column_names
    # elif training_args.do_eval:
    #     column_names = raw_datasets["validation"].column_names
    # elif training_args.do_predict:
    #     column_names = raw_datasets["test"].column_names
    # else:
    #     logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
    #     return

    # Get the column names for input/target.
    # dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    dataset_columns = ['source', 'summary']
    column_names = ['source', 'summary']

    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(
            examples, 
            eval_or_predict=False,
            use_sampleprompt=False):

        pad_token_id = tokenizer.pad_token_id
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        labels = None

        if use_sampleprompt:
            sampleprompt = examples[data_args.sampleprompt]
            sampleprompt = tokenizer(sampleprompt, max_length=max_target_length, padding=padding, truncation=True)
            sampleprompt = sampleprompt["input_ids"]
            padded_prompt = []
            for encoded in sampleprompt:
                padded_encoded = list(encoded)
                
                for _ in range(max_target_length - len(encoded)):
                    padded_encoded.append(pad_token_id)
                padded_prompt.append(padded_encoded)
            
            model_inputs["sampleprompt"] = padded_prompt

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        assert len(model_inputs["input_ids"]) == len(model_inputs["labels"]), f'len of model_inputs should == len labels, but is {len(model_inputs["input_ids"])} and {len(model_inputs["labels"])}'

        return model_inputs

    if training_args.do_train:
        # if "train" not in raw_datasets:
        #     raise ValueError("--do_train requires a train dataset")
        # train_dataset = raw_datasets["train"]

        train_dataset = load_dataset('json', data_files=data_args.dataset_name, field='train')
        train_dataset = train_dataset['train']

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
                fn_kwargs={
                            'use_sampleprompt': data_args.use_sampleprompt,
                }
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        # if "validation" not in raw_datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        # eval_dataset = raw_datasets["validation"]
        
        eval_dataset = None
        if 'RADIOLOGY' in data_args.dataset_name:
            # eval set too large, therefore cutting off. 
            # test set remains same size.
            eval_dataset = load_dataset('json', 
                                        data_files=data_args.dataset_name, 
                                        field='eval', 
                                        split='train[:30%]')
        else:
            eval_dataset = load_dataset('json', data_files=data_args.dataset_name, field='eval')
            eval_dataset = eval_dataset['train']

        #if data_args.max_eval_samples is not None:
           #eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
                fn_kwargs={
                            'eval_or_predict': True,
                            'use_sampleprompt': data_args.use_sampleprompt,}
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        # if "test" not in raw_datasets:
            # raise ValueError("--do_predict requires a test dataset")
        # predict_dataset = raw_datasets["test"]

        predict_dataset = load_dataset('json', data_files=data_args.dataset_name, field='test')
        predict_dataset = predict_dataset['train']

        #if data_args.max_predict_samples is not None:
            #predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
                fn_kwargs={
                            'eval_or_predict': True,
                            'use_sampleprompt': data_args.use_sampleprompt,}
            )
    
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )

    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config, # the model to projecting the target sequence to latent states, and projecting back
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        use_sampleprompt=data_args.use_sampleprompt,
        max_target_length=max_target_length,
    )

    # Disable cache during training (stops decoder KV-cache OOMs)
    model.config.use_cache = False

    # Check vocabulary sizes and resize embeddings if needed
    original_vocab_size = model.config.vocab_size
    tokenizer_vocab_size = len(tokenizer)
    
    print(f"Original model vocab size: {original_vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    
    if tokenizer_vocab_size != original_vocab_size:
        print(f"Resizing token embeddings from {original_vocab_size} to {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)
        
        # Initialize new embeddings properly by copying from existing ones
        if tokenizer_vocab_size > original_vocab_size:
            # For newly added tokens, initialize from the mean of existing embeddings
            with torch.no_grad():
                # Get the embeddings
                embeddings = model.get_input_embeddings()
                decoder_embeddings = model.get_output_embeddings()
                
                # Calculate mean of existing embeddings for initialization
                if original_vocab_size > 0:
                    mean_embedding = embeddings.weight[:original_vocab_size].mean(dim=0, keepdim=True)
                    # Initialize new embeddings with the mean
                    embeddings.weight[original_vocab_size:] = mean_embedding.repeat(
                        tokenizer_vocab_size - original_vocab_size, 1
                    )
                    
                    # Also initialize decoder embeddings if they exist and are different
                    if decoder_embeddings is not None and decoder_embeddings != embeddings:
                        mean_decoder_embedding = decoder_embeddings.weight[:original_vocab_size].mean(dim=0, keepdim=True)
                        decoder_embeddings.weight[original_vocab_size:] = mean_decoder_embedding.repeat(
                            tokenizer_vocab_size - original_vocab_size, 1
                        )
        
        print("Token embeddings resized and initialized successfully!")
    else:
        print("Tokenizer and model vocabulary sizes match - no resizing needed.")

    # model.load_state_dict(torch.load("./bart_discharge_prompt/pytorch_model.bin"))

    # torch.save(model.state_dict(), "./bartbasemodel/bartlarge.pth", _use_new_zipfile_serialization=False)

    count_parameters(model, all_param=False)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric, from https://github.com/huggingface/datasets/blob/master/metrics/rouge/rouge.py
    #metric = load_metric("./rouge_metric.py")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    """ def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        preds[preds < 0] = tokenizer.pad_token_id

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ✅ guard against empty strings (add these two lines)
        decoded_preds  = [p if p.strip() else "." for p in decoded_preds]
        decoded_labels = [r if r.strip() else "." for r in decoded_labels]

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Extract a few results from ROUGE
        for key, value in result.items():
            try:
                result[key] = value.mid.fmeasure * 100
            except:
                result[key] = value
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 6) for k, v in result.items()}
        return result """


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds[preds < 0] = tokenizer.pad_token_id

        # Decode
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # ✅ guard against empty strings (prevents scoring edge-cases)
        decoded_preds  = [p if p.strip() else "." for p in decoded_preds]
        decoded_labels = [r if r.strip() else "." for r in decoded_labels]

        # Your existing postprocess: trims & splits into sentences, then joins with "\n"
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # ---- Local ROUGE (no evaluate.load) ----
        scorer = RougeScorer(["rouge1","rouge2","rougeLsum"], use_stemmer=True, split_summaries=False)
        agg = scoring.BootstrapAggregator(confidence_interval=0.95, n_samples=1000)
        for ref, pred in zip(decoded_labels, decoded_preds):
            agg.add_scores(scorer.score(ref, pred))
        res = agg.aggregate()

        # Extract mid.fmeasure and convert to %
        result = {
                "rouge1":    res["rouge1"].mid.fmeasure * 100,
                "rouge2":    res["rouge2"].mid.fmeasure * 100,
                "rougeLsum": res["rougeLsum"].mid.fmeasure * 100,
        }

        # (optional) keep variance like your old metric
        result["rouge1_var"]    = float(np.var([s.fmeasure for s in agg._scores["rouge1"]]))
        result["rouge2_var"]    = float(np.var([s.fmeasure for s in agg._scores["rouge2"]]))
        result["rougeLsum_var"] = float(np.var([s.fmeasure for s in agg._scores["rougeLsum"]]))

        # Length stat
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = float(np.mean(prediction_lens))

        # Round like before
        result = {k: round(v, 6) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save the final model (this will be the best model if load_best_model_at_end=True)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        # Explicitly save the best model in a separate directory for clarity
        if training_args.load_best_model_at_end:
            best_model_path = os.path.join(training_args.output_dir, "best_model")
            trainer.save_model(best_model_path)
            print(f"✅ Best model saved to: {best_model_path}")
            print(f"✅ This model achieved the best {training_args.metric_for_best_model} score during training")

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    
    # num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    # above is original code, we update num_beams and max_length for full evalution and test
    num_beams = training_args.evaltest_generation_num_beams
    max_length = training_args.evaltest_generation_max_length

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        # Ensure we're using the best model for evaluation (if training was done)
        if training_args.do_train and training_args.load_best_model_at_end:
            logger.info("*** Using best model checkpoint for evaluation ***")
            print(f"✅ Using best model (based on {training_args.metric_for_best_model}) for final evaluation")
        
        metrics = trainer.evaluate(
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=training_args.length_penalty,
            no_repeat_ngram_size=training_args.no_repeat_ngram_size,
            metric_key_prefix="eval",
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        analyze_attention_entropy(
            model=model,
            dataset=eval_dataset,
            output_dir=training_args.output_dir,
            sample_size=100
        )
        cross_entropy_stats = analyze_cross_attention_entropy(
            model=model,
            dataset=eval_dataset,
            output_dir=training_args.output_dir,
            sample_size=100,
        )
        plot_cross_attention_normalized_entropy_curve(
            cross_entropy_stats=cross_entropy_stats,
            output_dir=training_args.output_dir,
        )
        plot_cross_attention_step_secondary_metrics(
            cross_entropy_stats=cross_entropy_stats,
            output_dir=training_args.output_dir,
        )
        plot_cross_attention_phase_normalized_entropy(
            cross_entropy_stats=cross_entropy_stats,
            output_dir=training_args.output_dir,
        )
        plot_cross_attention_entropy_by_factuality_bins(
            cross_entropy_stats=cross_entropy_stats,
            output_dir=training_args.output_dir,
        )
        plot_cross_attention_head_specialization_heatmap(
            cross_entropy_stats=cross_entropy_stats,
            output_dir=training_args.output_dir,
        )
        plot_cross_attention_per_head_normalized_entropy(
            cross_entropy_stats=cross_entropy_stats,
            output_dir=training_args.output_dir,
        )
        evidence_stats = analyze_evidence_concentration_over_decoding(
            model=model,
            dataset=eval_dataset,
            output_dir=training_args.output_dir,
            sample_size=100,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=training_args.length_penalty,
            no_repeat_ngram_size=training_args.no_repeat_ngram_size,
            do_sample=training_args.do_sample,
            top_p=training_args.top_p,
            temperature=training_args.temperature,
        )
        plot_evidence_concentration_curve(
            evidence_stats=evidence_stats,
            output_dir=training_args.output_dir,
        )

    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        # Ensure we're using the best model for predictions
        if training_args.do_train and training_args.load_best_model_at_end:
            logger.info("*** Using best model checkpoint for predictions ***")
            print(f"✅ Using best model (based on {training_args.metric_for_best_model}) for test predictions")
        
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=training_args.length_penalty,
            no_repeat_ngram_size=training_args.no_repeat_ngram_size,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding='utf-8') as writer:
                    to_write_str = ''
                    for to_write in range(len(predictions)):
                        # Remove internal newlines and extra spaces for each prediction
                        single_line_pred = ' '.join(str(predictions[to_write]).split())
                        to_write_str += single_line_pred + "\n"
                        if (to_write + 1 == len(predictions)):
                            to_write_str += "\n"
                    writer.write(to_write_str)

                to_write_dict = dict()
                for _pred in predictions:
                    to_write_dict[len(to_write_dict)] = _pred
                json_name = "gens.json"
                with open(os.path.join(training_args.output_dir, json_name), 'w', encoding='utf-8') as write_f:
                    write_f.write(json.dumps(to_write_dict))

        if not training_args.do_eval:
            analyze_attention_entropy(
                model=model,
                dataset=predict_dataset,
                output_dir=training_args.output_dir,
                sample_size=100
            )
            cross_entropy_stats = analyze_cross_attention_entropy(
                model=model,
                dataset=predict_dataset,
                output_dir=training_args.output_dir,
                sample_size=100,
            )
            plot_cross_attention_normalized_entropy_curve(
                cross_entropy_stats=cross_entropy_stats,
                output_dir=training_args.output_dir,
            )
            plot_cross_attention_step_secondary_metrics(
                cross_entropy_stats=cross_entropy_stats,
                output_dir=training_args.output_dir,
            )
            plot_cross_attention_phase_normalized_entropy(
                cross_entropy_stats=cross_entropy_stats,
                output_dir=training_args.output_dir,
            )
            plot_cross_attention_entropy_by_factuality_bins(
                cross_entropy_stats=cross_entropy_stats,
                output_dir=training_args.output_dir,
            )
            plot_cross_attention_head_specialization_heatmap(
                cross_entropy_stats=cross_entropy_stats,
                output_dir=training_args.output_dir,
            )
            plot_cross_attention_per_head_normalized_entropy(
                cross_entropy_stats=cross_entropy_stats,
                output_dir=training_args.output_dir,
            )
            evidence_stats = analyze_evidence_concentration_over_decoding(
                model=model,
                dataset=predict_dataset,
                output_dir=training_args.output_dir,
                sample_size=100,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=training_args.length_penalty,
                no_repeat_ngram_size=training_args.no_repeat_ngram_size,
                do_sample=training_args.do_sample,
                top_p=training_args.top_p,
                temperature=training_args.temperature,
            )
            plot_evidence_concentration_curve(
                evidence_stats=evidence_stats,
                output_dir=training_args.output_dir,
            )
                
    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)

    if training_args.do_train or training_args.do_eval:
        plot_training_convergence(
            log_history=trainer.state.log_history,
            output_dir=training_args.output_dir,
        )
    
    # model = BartForConditionalGeneration.from_pretrained(
    #     training_args.output_dir,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    #     # tokenizer=tokenizer,
    # )

    return

if __name__ == '__main__':
    run_bart()
