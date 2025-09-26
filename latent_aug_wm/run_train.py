from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from loguru import logger

import torch

torch.multiprocessing.set_sharing_strategy("file_system")


def load_hyper_config(fname, override_fname=None, return_string=True, override_keys={}):
    import json

    if override_fname:
        with open(override_fname, "r") as f:
            overrides = json.load(f)
    else:
        overrides = None

    if override_keys and not overrides:
        overrides = override_keys
    elif override_keys and overrides:
        overrides.update(override_keys)

    with open(fname, "r") as f:
        yaml_str = f.read()
        train_modules = load_hyperpyyaml(yaml_str, overrides)

    if return_string:
        return train_modules, yaml_str
    else:
        return train_modules


def add_git_context(checkpoint_dir):
    import git

    repo = git.Repo(search_parent_directories=True)
    commit_id = repo.head.object.hexsha

    t = repo.head.commit.tree
    diff = repo.git.diff(t)
    with open(add_name_to_dir("commit_id.txt", checkpoint_dir, id_dir=False), "w") as f:
        f.write(commit_id)
    with open(add_name_to_dir("git_diff.txt", checkpoint_dir, id_dir=False), "w") as f:
        f.write(diff)


def combine_checkpoint_dir_and_version(checkpoint_dir, version):
    checkpoint_dir += f"version_{version}/"
    return checkpoint_dir


def add_name_to_dir(name, directory, id_dir=True):
    if id_dir:
        return directory + name + "/"
    else:
        return directory + name


def create_dict(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    return


def save_config(config_string, training_config):

    checkpoint_dir = training_config["checkpoint_dir"]

    logger.info(f"saving backup config in {checkpoint_dir}")

    # if "version" in training_config:
    #    checkpoint_dir += f"version_{training_config['version']}/"
    create_dict(checkpoint_dir)
    add_git_context(checkpoint_dir)

    with open(checkpoint_dir + "training_config.yaml", "w") as f:
        f.write(config_string)


def create_override_dict(config):
    override_keys = config.get("override_keys", {})
    save_fname = config.get("override_json_fname", "override_keys.json")
    import json

    with open(save_fname, "w") as f:
        json.dump(override_keys, f)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    config_fname = getattr(parser.parse_args(), "config")

    # load, and build all modules from config
    config, yaml_str = load_hyper_config(config_fname)

    save_config(yaml_str, config)
    create_override_dict(config)

    # only use "train" key in the dict for training
    training_dict = config["train"]
    kwargs = (
        training_dict["training_kwargs"] if training_dict["training_kwargs"] else {}
    )
    training_dict["trainer"].fit(*training_dict["training_args"], **kwargs)
