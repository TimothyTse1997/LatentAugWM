import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence


def select_from_index(target_list, indexs):
    return [x for i, x in enumerate(target_list) if i in indexs]


def lens_to_mask(t=None) -> bool:  # noqa: F722 F821
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
    lens,
    cond,
    ref_texts,
    gen_texts,
    final_batch_size=32,
    allowed_padding=16000,
    durations=None,
    max_duration=1500,
    **kwargs,
):

    if durations is None:
        durations = get_max_duration(lens, ref_texts, gen_texts)

    index_exceed_max_dur = [i for i, d in enumerate(durations) if i > max_duration]

    list_exclude_from_indexs = lambda target, indexs: [
        t for i, t in enumerate(target) if i not in indexs
    ]
    lens = list_exclude_from_indexs(lens, index_exceed_max_dur)
    cond = list_exclude_from_indexs(cond, index_exceed_max_dur)
    ref_texts = list_exclude_from_indexs(ref_texts, index_exceed_max_dur)
    gen_texts = list_exclude_from_indexs(gen_texts, index_exceed_max_dur)
    durations = list_exclude_from_indexs(durations, index_exceed_max_dur)

    # print("current duration", durations)
    batch_size = len(durations)
    all_batch_indexs = set(range(batch_size))

    if max(durations) - min(durations) <= allowed_padding:
        return {
            "lens": lens[:final_batch_size],
            "cond": cond[:final_batch_size],
            "ref_texts": ref_texts[:final_batch_size],
            "gen_texts": gen_texts[:final_batch_size],
            "durations": durations[:final_batch_size],
        }, all_batch_indexs

    sorted_batch_index = [x for _, x in sorted(zip(durations, range(batch_size)))]
    inverse_sorted_batch_index = list(reversed(sorted_batch_index))

    def check_duration(current_ids):
        if not current_ids:
            return False
        current_durations = [d for i, d in enumerate(durations) if i in current_ids]
        # print(current_durations, (max(current_durations) - min(current_durations)))
        return (max(current_durations) - min(current_durations)) <= allowed_padding

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
        "lens": select_from_index(lens, final_ids)[:final_batch_size],
        "cond": select_from_index(cond, final_ids)[:final_batch_size],
        "ref_texts": select_from_index(ref_texts, final_ids)[:final_batch_size],
        "gen_texts": select_from_index(gen_texts, final_ids)[:final_batch_size],
        "durations": select_from_index(durations, final_ids)[:final_batch_size],
    }, final_ids


def recursive_batch_filtering(lens, cond, ref_texts, gen_texts, **kwargs):

    current_batch_indexs = set(range(len(lens)))
    data = {
        "lens": lens,
        "cond": cond,
        "ref_texts": ref_texts,
        "gen_texts": gen_texts,
    }
    current_data = data
    all_batches = []
    current_durations = kwargs.get("durations", None)
    while True:
        # try:
        if True:
            # print("current batch size", len(current_data['lens']))
            if len(current_data["lens"]) == 0:
                break
            filtered_batch, current_ids = batch_filter(**current_data, **kwargs)

            all_batches.append(filtered_batch)
            current_batch_indexs -= current_ids
            current_data = {
                k: select_from_index(v, current_batch_indexs)
                for k, v in current_data.items()
            }
            if current_durations is not None:
                kwargs["durations"] = select_from_index(
                    kwargs["durations"], current_batch_indexs
                )

            if len(current_batch_indexs) < kwargs["final_batch_size"]:
                break

    return all_batches


class F5TTSCollator:
    # def __init__(self, allowed_padding=100, final_batch_size=32):
    #     self.allowed_padding = allowed_padding
    #     self.final_batch_size = final_batch_size

    def __call__(self, batch):
        ref_mels = [b["ref_mel"][0] for b in batch]
        wav_fnames = [b["wav_fname"] for b in batch]
        ref_texts = [b["ref_text"] for b in batch]
        lens = [mel.shape[0] for mel in ref_mels]

        # cond = pad_sequence(ref_mels, batch_first=True)
        # batch_size, cond_seq_len, _ = cond.shape
        # lens = torch.full((batch_size,), cond_seq_len, dtype=torch.long)
        return {
            "cond": ref_mels,  # cond,
            "lens": lens,
            "wav_fnames": wav_fnames,
            "ref_texts": ref_texts,
        }


if __name__ == "__main__":
    lens = [0, 10, 11, 12, 13, 14, 18, 19, 20, 25, 25, 25]
    durations = lens  # [0, 10, 11, 12, 13, 14]
    allowed_padding = 4
    cond = torch.zeros((len(lens), 14, 11))
    ref_texts = [f"{i} I go to school by bus" for i in range(6)]
    gen_texts = [f"{i} how about you?" for i in range(6)]
    # print(batch_filter(
    #    lens, cond, ref_texts, gen_texts,
    #    final_batch_size=10, allowed_padding=allowed_padding,
    #    durations=durations
    # )["lens"])

    for b in recursive_batch_filtering(
        lens,
        cond,
        ref_texts,
        gen_texts,
        final_batch_size=3,
        allowed_padding=allowed_padding,
        durations=durations,
    ):
        print(b["lens"])
