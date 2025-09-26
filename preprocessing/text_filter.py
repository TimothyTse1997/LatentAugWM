import random
import json
from pathlib import Path
from tqdm import tqdm
from functools import partial

import numpy as np
import torch

import torchaudio
from datasets import load_dataset

import matplotlib.pyplot as plt


def plot_histogram_with_std(
    data,
    filename="histogram_with_std.png",
    bins=30,
    color="skyblue",
    edgecolor="black",
    dpi=300,
):
    """
    Plots a histogram of the input data with vertical dashed lines at ±1 standard deviation from the mean,
    and saves it as an image file.

    Parameters:
    - data (array-like): Input numerical data.
    - filename (str): File name to save the image (e.g., 'output.png').
    - bins (int): Number of bins for the histogram. Default is 30.
    - color (str): Color of the histogram bars. Default is 'skyblue'.
    - edgecolor (str): Color of the bar edges. Default is 'black'.
    - dpi (int): Resolution of the saved image in dots per inch. Default is 300.
    """

    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=0.7)

    plt.axvline(mean, color="red", linestyle="--", linewidth=2, label="Mean")
    plt.axvline(mean - std, color="green", linestyle="--", linewidth=2, label="-1 SD")
    plt.axvline(mean + std, color="green", linestyle="--", linewidth=2, label="+1 SD")

    plt.title("Histogram with ±1 Standard Deviation")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename, dpi=dpi)
    plt.close()


def get_ref_text_from_wav(wav_path):
    wav_path = wav_path.absolute()
    wav_parent, wav_name = wav_path.parent, wav_path.name
    wav_name = wav_name.split(".")[0]
    text_fname = wav_parent / f"{wav_name}.normalized.txt"
    with open(text_fname, "r") as f:
        ref_text = f.readline().replace("\n", "")
    return ref_text


def max_char_fn(ref_text_len, ref_audio_len, sr=24000, speed=1.0):
    return int(ref_text_len / (ref_audio_len / sr) * (22 - ref_audio_len / sr) * speed)


def main():
    tts_dataset_path = Path("/home/tst000/projects/datasets/LibriTTS/train-clean-100/")
    ref_audio_dict = {
        speaker_path.name: list(speaker_path.glob("**/*.wav"))
        for speaker_path in tts_dataset_path.glob("*")
    }
    all_max_char = []
    for k, v in tqdm(list(ref_audio_dict.items()), position=0):
        for ref_audio in tqdm(v, position=1):
            audio, sr = torchaudio.load(ref_audio)
            ref_text = get_ref_text_from_wav(ref_audio)
            ref_text_len = len(ref_text.encode("utf-8"))
            ref_audio_len = audio.shape[-1]

            max_char = max_char_fn(ref_text_len, ref_audio_len)
            all_max_char.append(max_char)
    plot_histogram_with_std(all_max_char)


def create_all_useful_path_for_dataset():
    save_file = "/home/tst000/projects/datasets/selected_ref_files.txt"
    tts_dataset_path = Path("/home/tst000/projects/datasets/LibriTTS/train-clean-100/")
    ref_audio_dict = {
        speaker_path.name: list(speaker_path.glob("**/*.wav"))
        for speaker_path in tts_dataset_path.glob("*")
    }
    all_data = []
    for k, v in tqdm(list(ref_audio_dict.items()), position=0):
        for ref_audio in tqdm(v, position=1):
            audio, sr = torchaudio.load(ref_audio)
            ref_text = get_ref_text_from_wav(ref_audio)
            ref_text_len = len(ref_text.encode("utf-8"))
            ref_audio_len = audio.shape[-1]

            max_char = max_char_fn(ref_text_len, ref_audio_len)
            if max_char < 180:
                continue
            all_data.append(ref_audio.absolute())

    with open(save_file, "w") as f:
        for row in all_data:
            f.write(str(row) + "\n")
    # plot_histogram_with_std(all_max_char)


def create_all_useful_text_tts_data():

    text_dataset = load_dataset("agentlans/high-quality-english-sentences")["train"][
        "text"
    ]
    save_file = "/home/tst000/projects/datasets/selected_gen_text.txt"

    total_num_text = 100000
    all_text = []
    num_text = 0
    for text in text_dataset:
        if num_text > total_num_text:
            break
        if "[NAME]" in text:
            continue
        if len(text.encode("utf-8")) > 180:
            continue
        num_text += 1
        all_text.append(text)

    with open(save_file, "w") as f:
        for row in all_text:
            f.write(str(row) + "\n")


def split_text_dataset(
    input_fname,
    output_train_fname=None,
    output_test_fname=None,
    output_eval_fname=None,
    frac=0.05,
):

    with open(input_fname, "r") as f:
        text_list = [line.rstrip() for line in f]

    total_data_count = len(text_list)
    test_eval_amount = max(int(frac * 2 * total_data_count), 2)
    print(len(text_list))

    random.shuffle(text_list)
    train_split = text_list[:-test_eval_amount]
    test_size = test_eval_amount // 2
    eval_test_split = text_list[-test_eval_amount:]
    test_split, eval_split = eval_test_split[:test_size], eval_test_split[test_size:]
    print(len(train_split), len(test_split), len(eval_split))

    def save_file_fn(save_file, all_data):
        with open(save_file, "w") as f:
            for row in all_data:
                f.write(str(row) + "\n")

    save_file_fn(output_train_fname, train_split)
    save_file_fn(output_test_fname, test_split)
    save_file_fn(output_eval_fname, eval_split)
    # return


if __name__ == "__main__":
    # main()
    # create_all_useful_path_for_dataset()
    # create_all_useful_text_tts_data()
    # split_text_dataset(
    #     "/home/tst000/projects/datasets/selected_ref_files.txt",
    #     output_train_fname="/home/tst000/projects/datasets/selected_ref_files_train.txt",
    #     output_test_fname="/home/tst000/projects/datasets/selected_ref_files_test.txt",
    #     output_eval_fname="/home/tst000/projects/datasets/selected_ref_files_eval.txt",
    # )
    split_text_dataset(
        "/home/tst000/projects/datasets/selected_gen_text.txt",
        output_train_fname="/home/tst000/projects/datasets/selected_gen_text_train.txt",
        output_test_fname="/home/tst000/projects/datasets/selected_gen_text_test.txt",
        output_eval_fname="/home/tst000/projects/datasets/selected_gen_text_eval.txt",
    )
