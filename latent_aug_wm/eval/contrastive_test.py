import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.stats import norm, percentileofscore
from tqdm import tqdm

import matplotlib.pyplot as plt
from f5_tts.infer.utils_infer import save_spectrogram

plt.rcParams.update({"font.size": 18})


def default_distance_function(A, B):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(A, B)


class SimilarityChecker:
    def __init__(
        self, generation_json: str, distance_function=default_distance_function
    ):
        self.generation_json = generation_json
        self.generation_list = json.load(open(self.generation_json, "r"))
        self.distance_function = distance_function
        # [{"orig_noise": "/path/", "generated_tensor": "/path/"}]

    def normalize_tensor(self, tensor):
        return tensor.flatten().unsqueeze(0)

    def get_source_similarity(self, orig_noise, generated_tensor):
        assert orig_noise.shape == generated_tensor.shape
        return self.distance_function(
            self.normalize_tensor(generated_tensor), self.normalize_tensor(orig_noise)
        )

    def get_distribution_2d(
        self, orig_noise: torch.Tensor, generated_tensor: torch.Tensor, N=100
    ):
        assert orig_noise.shape == generated_tensor.shape
        tensor_shape = torch.prod(torch.tensor(orig_noise.shape))
        random_sampled_noise = torch.randn(N, tensor_shape)

        flatten_target = self.normalize_tensor(generated_tensor)
        return self.distance_function(
            random_sampled_noise,
            self.normalize_tensor(flatten_target),
        )

    def get_distribution_2d_backup(
        self, orig_noise: torch.Tensor, generated_tensor: torch.Tensor, N=100
    ):
        # used for sanity check
        assert orig_noise.shape == generated_tensor.shape
        random_sampled_noise = [
            torch.randn_like(generated_tensor.flatten()) for _ in range(N)
        ]
        random_sampled_noise = torch.stack(random_sampled_noise)

        flatten_target = generated_tensor.flatten()
        return default_distance_function(random_sampled_noise, flatten_target)

    def get_source_similarity_backup(self, orig_noise, generated_tensor):
        assert orig_noise.shape == generated_tensor.shape
        return default_distance_function(
            generated_tensor.flatten().unsqueeze(0), orig_noise.flatten()
        )

    def get_p_value(self, distribution, source_distance):
        return 1 - percentileofscore(distribution.numpy(), source_distance.item()) / 100

    def generate_image_from_single_distribution(
        self, source_noise, target_tensor, N=100, suffix="", fname=""
    ):

        distribution = self.get_distribution_2d_backup(source_noise, target_tensor, N=N)
        source_distance = self.get_source_similarity_backup(source_noise, target_tensor)

        # Plot the fitted Gaussian
        plt.hist(
            distribution.numpy(), bins=20, color="skyblue", edgecolor="black", alpha=0.7
        )
        plt.axvline(
            source_distance.item(),
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Source Cosine = {source_distance.item():.4f}",
        )
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.title(
            f"Distribution of Random vs. Originated noise's Cosine Similarities {suffix}",
            wrap=True,
        )  # fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

        # compare source noise and simulated noise distribution
        original_shape = torch.tensor(source_noise.shape)
        rand_values = torch.randn(torch.prod(original_shape))

        new_fname = "dis_sanity_check_" + fname.name
        new_path = fname.parent / new_fname
        print(rand_values.shape, source_noise.shape)
        assert rand_values.shape == source_noise.flatten().shape
        print("rand_values", rand_values.mean(), rand_values.std())
        print(
            "source_noise.flatten()",
            source_noise.flatten().mean(),
            source_noise.flatten().std(),
        )

        plt.hist(
            rand_values.numpy(),
            bins=20,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
            label="newly sampled noise",
        )
        plt.hist(
            source_noise.flatten().numpy(),
            bins=20,
            color="orange",
            edgecolor="black",
            alpha=0.7,
            label="source noise",
        )
        plt.xlabel("noise sampl value")
        plt.ylabel("Frequency")
        plt.title(
            f"Distribution of noise comparing with newly sampled noise {suffix}",
            wrap=True,
        )  # fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(new_path, dpi=300)
        plt.close()

    def pad_to_same_length(self, source_noise, target_tensor):
        # print("source_noise", source_noise.shape)
        # print("target_tensor", target_tensor.shape)
        length = source_noise.shape[-1]
        target_length = target_tensor.shape[-1]
        result = None
        if target_length == length:
            result = target_tensor
        elif target_length > length:
            result = target_tensor[:, :, :length]
        else:
            length_diff = length - target_length
            result = F.pad(target_tensor, (0, length_diff), "constant", 0)
        # print(result.shape, source_noise.shape)
        assert result.shape == source_noise.shape
        return result

    def get_all_p_values(self, N=500, tensor_key="generated_tensor"):
        all_p_value = []

        for i, metadata in enumerate(tqdm(self.generation_list)):
            orig_noise_path, generated_tensor_path = (
                metadata["orig_noise"],
                metadata[tensor_key],
            )
            source_noise = torch.load(orig_noise_path)
            target_tensor = torch.load(generated_tensor_path)
            target_tensor = self.pad_to_same_length(source_noise, target_tensor)

            distribution = self.get_distribution_2d(source_noise, target_tensor, N=N)
            source_distance = self.get_source_similarity(source_noise, target_tensor)

            p_value = self.get_p_value(distribution, source_distance)
            all_p_value.append(p_value)
        return all_p_value

    def run(
        self,
        single_suffix,
        single_fname,
        p_value_suffix,
        p_value_fname,
        N=500,
        tensor_key="generated_tensor",
    ):
        all_p_value = []

        success_result = []
        for i, metadata in enumerate(tqdm(self.generation_list)):
            orig_noise_path, generated_tensor_path = (
                metadata["orig_noise"],
                metadata[tensor_key],
            )
            source_noise = torch.load(orig_noise_path)
            target_tensor = torch.load(generated_tensor_path)
            target_tensor = self.pad_to_same_length(source_noise, target_tensor)

            distribution = self.get_distribution_2d(source_noise, target_tensor, N=N)
            source_distance = self.get_source_similarity(source_noise, target_tensor)

            p_value = self.get_p_value(distribution, source_distance)
            all_p_value.append(p_value)

            success = (source_distance.numpy() > distribution.numpy()).mean()
            success_result.append(success)

            if i == 0:
                self.generate_image_from_single_distribution(
                    source_noise,
                    target_tensor,
                    N=100,
                    suffix=single_suffix,
                    fname=single_fname,
                )

        # print(all_p_value[:10])
        # plt.hist(all_p_value, bins=20, color='red', edgecolor='black', alpha=0.7, density=True)
        plt.hist(
            all_p_value,
            bins=20,
            color="red",
            edgecolor="black",
            alpha=0.7,
            density=False,
        )
        plt.xlabel("Cosine Similarity p value")
        plt.ylabel("Frequency")
        plt.title(
            f"Distribution of Cosine Similarity p value {p_value_suffix}",
            wrap=True,
        )  # fontsize=12)
        # plt.legend()
        plt.tight_layout()
        plt.savefig(p_value_fname, dpi=300)
        plt.close()

        return np.mean(success_result)

    def run_from_audio_tensor(
        self,
        single_suffix,
        single_fname,
        p_value_suffix,
        p_value_fname,
        N=500,
        use_mp3=False,
    ):
        # assume only audio is needed
        all_p_value = []
        for i, metadata in enumerate(tqdm(self.generation_list)):
            orig_noise_path = metadata["orig_noise"]
            target_id = Path(orig_noise_path).name.split(".")[0]

            audio_type = "mp3" if use_mp3 else "wav"

            audio_dir = Path(orig_noise_path).absolute().parents[1] / f"{audio_type}/"
            print(audio_dir)
            assert audio_dir.exists()
            audio_fname = target_id + f".{audio_type}"
            audio_path = audio_dir / audio_fname
            assert audio_path.exists()

            source_noise = torch.load(orig_noise_path)
            # target_tensor = torch.load(generated_tensor_path)
            target_tensor, sr = torchaudio.load(audio_path)
            print(orig_noise_path, audio_path)
            assert target_tensor.shape == source_noise.shape

            distribution = self.get_distribution_2d(source_noise, target_tensor, N=N)
            source_distance = self.get_source_similarity(source_noise, target_tensor)

            p_value = self.get_p_value(distribution, source_distance)
            all_p_value.append(p_value)

            if i == 0:
                self.generate_image_from_single_distribution(
                    source_noise,
                    target_tensor,
                    N=100,
                    suffix=single_suffix,
                    fname=single_fname,
                )

        # print(all_p_value[:10])
        # plt.hist(all_p_value, bins=20, color='red', edgecolor='black', alpha=0.7, density=True)
        plt.hist(
            all_p_value,
            bins=20,
            color="red",
            edgecolor="black",
            alpha=0.7,
            density=False,
        )
        plt.xlabel("Cosine Similarity p value")
        plt.ylabel("Frequency")
        plt.title(
            f"Distribution of Cosine Similarity p value {p_value_suffix}",
            wrap=True,
        )  # fontsize=12)
        # plt.legend()
        plt.tight_layout()
        plt.savefig(p_value_fname, dpi=300)
        plt.close()

    def run_mp3(
        self,
        single_suffix,
        single_fname,
        p_value_suffix,
        p_value_fname,
        N=500,
        target_format="mp3",
    ):
        all_p_value = []
        success_result = []
        for i, metadata in enumerate(tqdm(self.generation_list)):
            orig_noise_path, generated_tensor_path = (
                metadata["orig_noise"],
                metadata[f"{target_format}_mel"],
            )
            source_noise = torch.load(orig_noise_path)
            target_tensor = torch.load(generated_tensor_path)
            print(orig_noise_path, generated_tensor_path)

            distribution = self.get_distribution_2d(source_noise, target_tensor, N=N)
            source_distance = self.get_source_similarity(source_noise, target_tensor)

            p_value = self.get_p_value(distribution, source_distance)
            all_p_value.append(p_value)
            success = (source_distance > distribution).mean()
            success_result.append(success)

            if i == 0:
                self.generate_image_from_single_distribution(
                    source_noise,
                    target_tensor,
                    N=100,
                    suffix=single_suffix,
                    fname=single_fname,
                )

        # print(all_p_value[:10])
        # plt.hist(all_p_value, bins=20, color='red', edgecolor='black', alpha=0.7, density=True, normed=False)
        plt.hist(
            all_p_value,
            bins=20,
            color="red",
            edgecolor="black",
            alpha=0.7,
            density=False,
        )
        plt.xlabel("Cosine Similarity p value")
        plt.ylabel("Frequency")
        plt.title(
            f"Distribution of Cosine Similarity p value {p_value_suffix}",
            wrap=True,
        )  # fontsize=12)
        # plt.legend()
        plt.tight_layout()
        plt.savefig(p_value_fname, dpi=300)
        plt.close()

        return np.mean(success_result)


# def p_value_accuracy_test(
#     model_name,
#     tensor_key="generated_tensor",
#     special_suffix="",
#     meta_data_fname="metadata.json",
#     special_plot_suffix="",
#     distance_function=default_distance_function,
# ):
#     generation_json = f"/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/{model_name}/{meta_data_fname}"
#     checker = SimilarityChecker(
#         generation_json=generation_json,
#         distance_function=distance_function
#     )
#     p_values = checker.get_all_p_values(N=500, tensor_key=tensor_key)


def main(
    model_name,
    tensor_key="generated_tensor",
    special_suffix="",
    meta_data_fname="metadata.json",
    special_plot_suffix="",
):
    generation_json = f"/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/{model_name}/{meta_data_fname}"
    checker = SimilarityChecker(generation_json=generation_json)

    save_dir = Path(
        f"/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/{model_name}/plots"
    )
    if not save_dir.exists():
        save_dir.mkdir()

    single_suffix = model_name + special_plot_suffix
    single_fname = save_dir / f"{model_name}_dist_example_N_500{special_suffix}.pdf"
    p_value_suffix = model_name + special_plot_suffix
    p_value_fname = save_dir / f"{model_name}_pvalue_dist_N_500{special_suffix}.pdf"

    acc = checker.run(
        single_suffix,
        single_fname,
        p_value_suffix,
        p_value_fname,
        N=500,
        tensor_key=tensor_key,
    )
    return acc


def main_mp3(model_name, target_format="mp3"):
    generation_json = f"/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/{model_name}/{target_format}_metadata.json"
    checker = SimilarityChecker(generation_json=generation_json)

    save_dir = Path(
        f"/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/{model_name}/plots"
    )
    if not save_dir.exists():
        save_dir.mkdir()

    single_suffix = model_name + f" ({target_format})"
    single_fname = save_dir / f"{model_name}_dist_example_N_500_{target_format}.pdf"
    p_value_suffix = model_name + f" ({target_format})"
    p_value_fname = save_dir / f"{model_name}_pvalue_dist_N_500_{target_format}.pdf"

    checker.run_mp3(
        single_suffix,
        single_fname,
        p_value_suffix,
        p_value_fname,
        N=500,
        target_format=target_format,
    )


def main_audio(model_name, use_mp3=False):
    generation_json = f"/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/{model_name}/metadata.json"
    checker = SimilarityChecker(generation_json=generation_json)

    save_dir = Path(
        f"/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/{model_name}/plots"
    )
    if not save_dir.exists():
        save_dir.mkdir()

    additional_suffix = " (mp3)" if use_mp3 else ""
    additional_path_suffix = "_mp3" if use_mp3 else "_audio"

    single_suffix = model_name + additional_suffix
    single_fname = (
        save_dir / f"{model_name}_dist_example_N_500{additional_path_suffix}.pdf"
    )
    p_value_suffix = model_name + additional_suffix
    p_value_fname = (
        save_dir / f"{model_name}_pvalue_dist_N_500{additional_path_suffix}.pdf"
    )

    checker.run_from_audio_tensor(
        single_suffix,
        single_fname,
        p_value_suffix,
        p_value_fname,
        N=500,
        use_mp3=use_mp3,
    )


if __name__ == "__main__":
    from latent_aug_wm.eval.convert_mp3 import MP3_BIT_RATE

    # model_name = "matchatts"
    # model_name = "diffwave"
    # model_name = "f5tts"
    for model_name in ["diffwave", "matchatts", "f5tts"]:
        main(
            model_name,
            tensor_key="generated_tensor",
            special_suffix="original_removed_density",
            meta_data_fname="metadata.json",
            special_plot_suffix="(no augmentation)",
        )

    # main(
    #     model_name,
    #     tensor_key="aug_mel",
    #     special_suffix="_hifigan_22k",
    #     meta_data_fname="hifigan_22k_metadata.json",
    #     special_plot_suffix="(hifigan 22k)"
    # )

    # main(
    #     model_name,
    #     tensor_key="mp4_mel",
    #     special_suffix="_mp4",
    #     meta_data_fname="mp4_metadata.json",
    #     special_plot_suffix="(mp4)"
    # )

    # for bit_rate in MP3_BIT_RATE:
    #     main(
    #         model_name,
    #         tensor_key=f"mp3_{bit_rate}_mel",
    #         special_suffix=f"mp3_{bit_rate}",
    #         meta_data_fname=f"mp3_{bit_rate}_metadata.json",
    #         special_plot_suffix=f"(mp3 {bit_rate})"
    #     )

    # main_audio(model_name, use_mp3=True)

    # save_dir = Path("/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts/")
    # all_paths = list(save_dir.glob("wavLowPass_*_metadata.json"))
    # assert(all_paths)
    # for meta_datapath in save_dir.glob("wavLowPass_*.json"):
    #     current_cutoff = meta_datapath.name.split("_")[1]
    #     acc = main(
    #         model_name,
    #         tensor_key=f"wav_mel",
    #         special_suffix=f"lowpass_{current_cutoff}",
    #         meta_data_fname=meta_datapath.name,
    #         special_plot_suffix=f"(lowpass {current_cutoff})"
    #     )
    #     print(f"current cutoff: {current_cutoff}, acc: {acc}")
