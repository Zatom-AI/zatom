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

> Note: Make sure to create a `.env` file, for which you can reference `.env.example` as an example.

### Default

> Note: We recommend installing `zatom` in a clean Python environment, using `conda` or otherwise.

For example, to install `conda`, one can use the following commands.

```bash
wget "https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Miniforge3-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # or `source ~/.zshrc` - alternatively, one can restart their shell session to achieve the same result
```

With `conda` available, one can build a virtual environment for `zatom`.

```bash
# Clone project, making sure Git LFS is installed beforehand (https://git-lfs.com/)
git clone https://github.com/amorehead/zatom
cd zatom

# [OPTIONAL] Create Conda environment (for Linux)
conda create -n zatom -c conda-forge python=3.10 gcc=11.4.0 gxx=11.4.0 libstdcxx=14.1.0 libstdcxx-ng=14.1.0 libgcc=14.1.0 libgcc-ng=14.1.0 compilers=1.5.2
conda activate zatom

# [OPTIONAL] Alternatively, create Conda environment (for macOS)
conda create -n zatom -c conda-forge python=3.10 clang=18 clangxx=18 libcxx=18 libcxx-devel=18 libgfortran5=15.1.0 lld=20.1.7 pybind11=3.0.0
conda activate zatom

# [OPTIONAL] Install `pyeqeq` (for macOS)
export CC=clang
export CXX=clang++
export CPPFLAGS="-isystem $CONDA_PREFIX/include -isystem $CONDA_PREFIX/include/c++/v1"
export CXXFLAGS="-std=c++17"
export LDFLAGS="-fuse-ld=lld -L$CONDA_PREFIX/lib"
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"
pip install --no-build-isolation pyeqeq
unset DYLD_LIBRARY_PATH

# Install requirements
pip install -e .[cuda]

# [OPTIONAL] Install pre-commit hooks
pre-commit install
```

> Note: If you are installing on systems without access to CUDA GPUs (namely macOS or ROCm systems), remove `[cuda]` from the above commands. For macOS specifically, make sure to set `e{n}coder.fused_attn=true` and `e{n}coder.jvp_attn=false` as well as `data.datamodule.batch_size.{train,val,test}=128`. Be aware that the CPU-only version (e.g., without macOS's MPS GPU backend) will be significantly slower than the GPU version.

### Docker

For sake of reproducibility, one can alternatively build a (CUDA-based) Docker image for `zatom`.

```bash
# Clone project, making sure Git LFS is installed beforehand (https://git-lfs.com/)
git clone https://github.com/amorehead/zatom
cd zatom

# Enable BuildKit to securely pass GitHub access token to Docker
export DOCKER_BUILDKIT=1
export GITHUB_TOKEN=your_token_value

# E.g., to build image on local machine
docker build --platform linux/amd64 --secret id=github_token,env=GITHUB_TOKEN --no-cache -t zatom:0.0.1 - < Dockerfile
# Skip the following three steps if not using NERSC cluster
docker login registry.nersc.gov
docker tag zatom:0.0.1 registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1
docker push registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1

# E.g., alternatively, to build image on NERSC cluster
podman-hpc build --platform linux/amd64 --secret id=github_token,env=GITHUB_TOKEN --no-cache -t zatom:0.0.1 - < Dockerfile
podman-hpc migrate zatom:0.0.1
podman-hpc login registry.nersc.gov
podman-hpc tag zatom:0.0.1 registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1
podman-hpc push registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1

# If using NERSC cluster, prepare image with Shifter
shifterimg login registry.nersc.gov
shifterimg -v pull registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1
```

> Note: The Docker image is ~30 GB in size. Make sure you have enough storage space beforehand to build it.

## Training

See the `[DEFAULT] train_ebm.py` config within `.vscode/launch.json` for an example of how to train a model.

## Evaluation

Consider using [`Protein Viewer`](https://marketplace.visualstudio.com/items?itemName=ArianJamasb.protein-viewer) for VS Code to visualize molecules and using [`VESTA`](https://jp-minerals.org/vesta/en/) locally to visualize materials. Running [`PyMOL`](https://www.pymol.org/) locally may also be useful for aligning/comparing two molecules.

## Acknowledgements

`zatom` builds upon the source code and data from the following projects:

- [all-atom-diffusion-transformer](https://github.com/facebookresearch/all-atom-diffusion-transformer)
- [EBT](https://github.com/alexiglad/EBT)
- [jvp_flash_attention](https://github.com/amorehead/jvp_flash_attention)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [ProteinWorkshop](https://github.com/a-r-j/ProteinWorkshop)
- [posebusters](https://github.com/maabuu/posebusters)

We thank all their contributors and maintainers!
