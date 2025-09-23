from setuptools import setup
import setuptools

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="latent_aug_wm",
    packages=setuptools.find_packages(exclude=("model/",)),
    install_requires=required,
    # install_requires=[],
    extras_require={},
)
