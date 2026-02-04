from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment

MP3_BIT_RATE = ["32k", "96k", "128k", "160k", "192k", "256k", "320k"]


def mp3_file_conversion(
    source_file,
    target_file,
    sampling_rate=16000,
    target_bit_rate=None,
    new_format="mp3",
):
    if target_bit_rate is None:
        AudioSegment.from_wav(source_file).export(target_file, format=new_format)
    else:
        AudioSegment.from_wav(source_file).export(
            target_file, format=new_format, bitrate=target_bit_rate
        )


def main(
    target_dir="/home/tst000/projects/tst000/datasets/f5tts_random_audio/",
    output_dir=None,
    new_format="mp3",
    target_bit_rate=None,
):

    if output_dir is None:
        output_dir = target_dir
    output_dir = Path(output_dir)

    get_name = lambda x: x.name.split(".")[0] + f".{new_format}"
    get_new_path = lambda x: (output_dir / get_name(x))

    all_paths = list(Path(target_dir).glob("*.wav"))
    assert target_bit_rate in MP3_BIT_RATE
    for p in tqdm(all_paths):

        if target_bit_rate is None:
            AudioSegment.from_wav(p).export(get_new_path(p), format=new_format)
        else:
            AudioSegment.from_wav(p).export(
                get_new_path(p), format=new_format, bitrate=target_bit_rate
            )


if __name__ == "__main__":
    model_name = "f5tts"
    # model_name = "diffwave"

    target_format = "mp3"
    # target_format = "mp4"

    project_dir = f"/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/{model_name}/"
    # target_dir = project_dir + "wav/"

    # output_dir = project_dir + f"{target_format}/"
    # if not Path(output_dir).exists(): Path(output_dir).mkdir()

    # #main(target_dir=target_dir, output_dir=output_dir)
    # main(target_dir=target_dir, output_dir=output_dir, new_format=target_format)

    for target_bit_rate in MP3_BIT_RATE:
        target_dir = project_dir + f"wav/"

        output_dir = project_dir + f"{target_format}_{target_bit_rate}/"
        if not Path(output_dir).exists():
            Path(output_dir).mkdir()

        # #main(target_dir=target_dir, output_dir=output_dir)
        main(
            target_dir=target_dir,
            output_dir=output_dir,
            new_format=target_format,
            target_bit_rate=target_bit_rate,
        )
