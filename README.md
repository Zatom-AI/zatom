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
conda create -n zatom python=3.11
conda activate zatom

# Install requirements
pip install -e .[cuda]

# [OPTIONAL] Install pre-commit hooks
pre-commit install
```

> Note: If you are installing on systems without access to CUDA GPUs, remove `[cuda]` from the above commands. Be aware that the CPU version will be significantly slower than the GPU version.

### Docker

To simplify installation, one can alternatively build a Docker image for `zatom`.

```bash
# Set up temporary directory for Docker image builds
mkdir ../docker_zatom/ && cp Dockerfile ../docker_zatom/ && cd ../docker_zatom/
git clone https://github.com/amorehead/zatom # Simply `cd zatom && git pull origin main && cd ../` if already cloned

# E.g., on local machine
docker build --platform linux/amd64 --build-arg GITHUB_TOKEN=your_token_value --no-cache -t zatom:0.0.1 .
# Skip the following three steps if not using NERSC cluster
docker login registry.nersc.gov
docker tag zatom:0.0.1 registry.nersc.gov/dasrepo/amorehead/zatom:0.0.1
docker push registry.nersc.gov/dasrepo/amorehead/zatom:0.0.1

# E.g., alternatively, on NERSC cluster
podman-hpc build --platform linux/amd64 --build-arg GITHUB_TOKEN=your_token_value --no-cache -t zatom:0.0.1 .
podman-hpc migrate zatom:0.0.1
podman-hpc tag zatom:0.0.1 registry.nersc.gov/dasrepo/amorehead/zatom:0.0.1
podman-hpc push registry.nersc.gov/dasrepo/amorehead/zatom:0.0.1

# On NERSC cluster
shifterimg login registry.nersc.gov
shifterimg -v pull registry.nersc.gov/dasrepo/amorehead/zatom:0.0.1

# Return to original repository
cd ../zatom/
```