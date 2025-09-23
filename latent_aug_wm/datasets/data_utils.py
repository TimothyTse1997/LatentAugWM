import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence


def lens_to_mask(
    t: int["b"], length: int | None = None
) -> bool["b n"]:  # noqa: F722 F821
    if not length is not None:
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def get_duration_from_text(
    ref_audio_len, ref_text, gen_text, local_speed=1.0, hop_length=256
):
    ref_text_len = len(ref_text.encode("utf-8"))
    gen_text_len = len(gen_text.encode("utf-8"))
    duration = ref_audio_len + int(
        ref_audio_len / ref_text_len * gen_text_len / local_speed
    )
    return duration


def get_max_duration(lens, ref_texts, gen_texts):
    durations = [
        get_duration_from_text(*inputs) for inputs in zip(lens, ref_texts, gen_texts)
    ]
    return durations


def batch_filter(
    lens, cond, ref_texts, gen_texts, final_batch_size=32, allowed_padding=16000
):
    durations = get_max_duration(lens, ref_texts, gen_texts)
    if max(durations) - min(durations) < allowed_padding:
        return (
            lens[:final_batch_size],
            cond[:final_batch_size],
            ref_texts[:final_batch_size],
            gen_texts[:final_batch_size],
        )

    batch_size = len(durations)
    sorted_batch_index = [x for _, x in sorted(zip(durations, range(batch_size)))]
    inverse_sorted_batch_index = list(reversed(sorted_batch_index))

    all_batch_indexs = set(range(batch_size))

    def check_duration(current_ids):
        current_durations = [d for i, d in enumerate(durations) if i in current_ids]
        return (max(current_durations) - min(current_durations)) < allowed_padding

    def select_from_index(target_list, indexs):
        return [x for i, x in enumerate(target_list) if i in indexs]

    final_ids = None
    for i in range(batch_size):
        lower_batch_ids = set(sorted_batch_index[:i])
        higher_batch_ids = set(inverse_sorted_batch_index[:i])

        if check_duration(all_batch_indexs - higher_batch_ids):
            final_ids = all_batch_indexs - higher_batch_ids
        elif check_duration(all_batch_indexs - lower_batch_ids):
            final_ids = all_batch_indexs - lower_batch_ids
        elif check_duration(all_batch_indexs - lower_batch_ids - higher_batch_ids):
            final_ids = all_batch_indexs - lower_batch_ids - higher_batch_ids
        if final_ids is not None:
            break
    assert final_ids is not None

    return {
        "lens": select_from_index(final_ids, final_ids),
        "cond": select_from_index(cond, final_ids),
        "ref_texts": select_from_index(ref_texts, final_ids),
        "gen_texts": select_from_index(gen_texts, final_ids),
    }


class F5TTSCollator:
    def __call__(self, batch):
        ref_mels = [b["ref_mel"] for b in batch]
        wav_fnames = [b["wav_fname"] for b in batch]
        ref_texts = [b["ref_text"] for b in batch]

        cond = pad_sequence(ref_mels, batch_first=True)

        batch_size, cond_seq_len, _ = cond.shape
        # lens = torch.full((batch_size,), cond_seq_len, dtype=torch.long)
        lens = [ref_mels.shape[1] for mel in ref_mels]
        return {
            "cond": cond,
            "lens": lens,
            "wav_fnames": wav_fnames,
            "ref_texts": ref_texts,
        }
