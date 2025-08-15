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

> Note: We recommend installing `zatom` in a clean Python environment, using `conda` or otherwise.

```bash
# clone project
git clone https://github.com/amorehead/zatom
cd zatom

# [OPTIONAL] create Conda environment
conda create -n zatom python=3.10
conda activate zatom

# install requirements
pip install -e .[cuda]

# [OPTIONAL] install pre-commit hooks
pre-commit install
```

> Note: If you are installing on systems without access to CUDA GPUs, remove `[cuda]` from the above commands. Be aware that the CPU version will be significantly slower than the GPU version.
