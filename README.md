<div align="center">

# Zatom

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

<!-- <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br> -->

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->

<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

Official repository of Zatom, a multimodal energy-based all-atom transformer

## Installation

### Default

> Note: We recommend installing `zatom` in a clean Python environment, using `conda` or otherwise.

```bash
# Clone project
git clone https://github.com/amorehead/zatom
cd zatom

# [OPTIONAL] Create Conda environment
conda create -n zatom python=3.10 gcc=11.4.0 gxx=11.4.0 libstdcxx=14.1.0 libstdcxx-ng=14.1.0 libgcc=14.1.0 libgcc-ng=14.1.0 compilers=1.5.2
conda activate zatom

# Install requirements
pip install -e .[cuda]

# [OPTIONAL] Install pre-commit hooks
pre-commit install
```

> Note: If you are installing on systems without access to CUDA GPUs, remove `[cuda]` from the above commands. Be aware that the CPU version will be significantly slower than the GPU version.

### Docker

For sake of reproducibility, one can alternatively build a Docker image for `zatom`.

```bash
# Clone project
git clone https://github.com/amorehead/zatom
cd zatom

# E.g., to build image on local machine
docker build --platform linux/amd64 --build-arg GITHUB_TOKEN=your_token_value --no-cache -t zatom:0.0.1 - < Dockerfile
# Skip the following three steps if not using NERSC cluster
docker login registry.nersc.gov
docker tag zatom:0.0.1 registry.nersc.gov/dasrepo/amorehead/zatom:0.0.1
docker push registry.nersc.gov/dasrepo/amorehead/zatom:0.0.1

# E.g., alternatively, to build image on NERSC cluster
podman-hpc build --platform linux/amd64 --build-arg GITHUB_TOKEN=your_token_value --no-cache -t zatom:0.0.1 - < Dockerfile
podman-hpc migrate zatom:0.0.1
podman-hpc login registry.nersc.gov
podman-hpc tag zatom:0.0.1 registry.nersc.gov/dasrepo/amorehead/zatom:0.0.1
podman-hpc push registry.nersc.gov/dasrepo/amorehead/zatom:0.0.1

# If using NERSC cluster, prepare image with Shifter
shifterimg login registry.nersc.gov
shifterimg -v pull registry.nersc.gov/dasrepo/amorehead/zatom:0.0.1
```

> Note: The Docker image is ~30 GB in size. Make sure you have enough storage space beforehand to build it.
