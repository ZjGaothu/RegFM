"""
RegFM setup.

The customized `transformers` package lives under `src/transformers`.
Installing this project (pip install -e .) also installs that package, so that
`import transformers` resolves to the local copy.
"""

import shutil
from pathlib import Path

from setuptools import find_packages, setup

# Remove stale egg-info dirs that can break editable installs
for egg_name in ("transformers.egg-info", "regfm.egg-info"):
    stale_egg_info = Path(__file__).parent / egg_name
    if stale_egg_info.exists():
        print(f"Warning: {stale_egg_info} exists; removing it for a clean editable install.")
        shutil.rmtree(stale_egg_info)

extras = {
    "torch": ["torch"],
    "train": ["tensorboard"],
}
extras["all"] = extras["torch"] + extras["train"]

setup(
    name="regfm",
    version="0.1.0",
    author="Zijing Gao",
    author_email="gzj21@mails.tsinghua.edu.cn",
    description="RegFM: a regulatory foundation model for gene expression prediction",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="genomics gene-expression regulatory deep-learning transformer",
    url="https://github.com/<your-org>/RegFM",
    # Everything under src/ is the install root:
    #   src/transformers/  -> import transformers
    #   src/model.py       -> import model
    #   ...
    package_dir={"": "src"},
    packages=find_packages("src"),
    py_modules=[
        "main",
        "model",
        "module",
        "dataset",
        "utils",
    ],
    install_requires=[
        "numpy",
        "tokenizers==0.15.2",
        "boto3",
        "filelock",
        "requests",
        "tqdm>=4.27",
        "regex!=2019.12.17",
        "sentencepiece",
        "sacremoses",
    ],
    extras_require=extras,
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
