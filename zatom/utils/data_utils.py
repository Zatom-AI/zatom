import gzip
import os
import pickle  # nosec
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ase
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from mendeleev import element
from torch_geometric.data import Dataset
from tqdm import tqdm

from zatom.utils import pylogger
from zatom.utils.typing_utils import typecheck

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@typecheck
def extract_tar_gz(file_path: str, extract_to: str, verbose: bool = True):
    """Extract a `tar.gz` file.

    Args:
        file_path: The path to the tar.gz file.
        extract_to: The directory to extract the contents to.
    """
    if verbose:
        log.info(f"Extracting {file_path} to {extract_to}...")

    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)  # nosec

    if verbose:
        log.info(f"Extracted {file_path} to {extract_to}.")


@typecheck
def get_omol25_per_atom_energy_and_stats(
    dataset: Dataset,
    coef_path: str | None = None,
    recalculate: bool = False,
    include_hof: bool = False,
) -> tuple[dict, dict]:
    """Get energy per-atom energy for referencing plus get mean, std, and average number of nodes,
    either loading from file or computing.

    Args:
        dataset: An OMol25 dataset instance
        coef_path: Path to save/load coefficients. If None, coefficients won't be saved.
        recalculate: Force recalculation even if file exists.
        include_hof: Whether to include heat of formation in energy referencing.

    Returns:
        Per-element energy coefficients and dataset statistics.
    """
    # Try to load from file if path provided and not forcing recalculation
    coef_path = os.path.join(coef_path, f"per_atom_energy_hof_{str(include_hof)}.pkl")

    if os.path.exists(coef_path) and not recalculate:
        log.info(f"Loading OMol25 energy normalization coefficients from {coef_path}")
        with open(coef_path, "rb") as f:
            stats = pickle.load(f)  # nosec
            per_elm_energy = stats["per_elm_energy"]
            dataset_stats = stats["dataset_stats"]

            return per_elm_energy, dataset_stats

    return compute_per_atom_energy_and_stat(dataset, save_path=coef_path, include_hof=include_hof)


@typecheck
def get_mptrj_stats(
    dataset: Dataset,
    coef_path: str | None = None,
    recalculate: bool = False,
) -> Dict[str, Any]:
    """Get mean and std of energy, either loading from file or computing.

    Args:
        dataset: An MPtrj dataset instance
        coef_path: Path to save/load coefficients. If None, coefficients won't be saved.
        recalculate: Force recalculation even if file exists.

    Returns:
        Dataset statistics.
    """
    # Try to load from file if path provided and not forcing recalculation
    coef_path = os.path.join(coef_path, "stats.pkl")

    if os.path.exists(coef_path) and not recalculate:
        log.info(f"Loading MPtrj energy normalization coefficients from {coef_path}")
        with open(coef_path, "rb") as f:
            dataset_stats = pickle.load(f)  # nosec
            return dataset_stats

    energies = []
    for idx in tqdm(dataset.indices()):
        data = dataset.get(idx)
        energy_dft = data.y[0, 0].item()
        energies.append(energy_dft)

    energies = np.array(energies)
    mean_energy = float(np.mean(energies))
    scale = float(np.std(energies))
    dataset_stats = {
        "shift": mean_energy,
        "scale": scale,
    }

    log.info(
        f"MPtrj dataset statistics - energy mean: {dataset_stats['shift']:.4f}, scale (std): {dataset_stats['scale']:.4f}"
    )
    if coef_path:
        os.makedirs(os.path.dirname(coef_path), exist_ok=True)
        with open(coef_path, "wb") as f:
            pickle.dump(dataset_stats, f)
        log.info(f"Saved MPtrj dataset statistics to {coef_path}")

    return dataset_stats


@typecheck
def get_matbench_stats(
    dataset: Dataset,
    task_name: str,
    coef_path: str | None = None,
    recalculate: bool = False,
) -> Dict[str, Any]:
    """Get mean and std of property, either loading from file or computing.

    Args:
        dataset: An Matbench dataset instance.
        task_name: Name of the property task.
        coef_path: Path to save/load coefficients. If None, coefficients won't be saved.
        recalculate: Force recalculation even if file exists.

    Returns:
        Dataset statistics.
    """
    # Try to load from file if path provided and not forcing recalculation
    coef_path = os.path.join(coef_path, f"{task_name}_stats.pkl")

    if os.path.exists(coef_path) and not recalculate:
        log.info(f"Loading Matbench property normalization coefficients from {coef_path}")
        with open(coef_path, "rb") as f:
            dataset_stats = pickle.load(f)  # nosec
            return dataset_stats

    properties = []
    for idx in tqdm(dataset.indices()):
        data = dataset.get(idx)
        property_value = data.y
        properties.append(property_value)

    properties = torch.cat(properties)
    mean_property = properties.mean(dim=0, keepdim=True)
    scale = properties.std(dim=0, keepdim=True)
    dataset_stats = {
        "shift": mean_property,
        "scale": scale,
    }

    log.info(
        f"Matbench dataset statistics - property mean: {dataset_stats['shift']}, scale (std): {dataset_stats['scale']}"
    )
    if coef_path:
        os.makedirs(os.path.dirname(coef_path), exist_ok=True)
        with open(coef_path, "wb") as f:
            pickle.dump(dataset_stats, f)
        log.info(f"Saved Matbench dataset statistics to {coef_path}")

    return dataset_stats


@typecheck
def compute_per_atom_energy_and_stat(
    dataset: Dataset,
    save_path: str | None = None,
    include_hof: bool = True,
    use_rmsd: bool = True,
) -> tuple[dict, dict]:
    """Compute per-element energy coefficients using linear regression.

    The formula used is: E_ref = E_DFT - Σ[E_i,DFT - ΔH_f,i]

    Args:
        dataset: An OMol25 dataset instance.
        save_path: Path to save the coefficients (optional).
        include_hof: Whether to include heat of formation in energy referencing.
        use_rmsd: Whether to use RMSD with Bessel's correction (True) or std (False) for scale.

    Returns:
        A tuple of (per_elm_energy, dataset_stats) where dataset_stats contains
        mean, std, and avg_num_nodes of normalized energies.
    """
    log.info("Computing OMol25 per-element energy for referencing")
    num_elements = len(dataset.dataset_info["atom_decoder"])
    log.info(f"Using {num_elements} elements")

    # Get heat of formation values for all elements
    log.info("Getting heat of formation values from mendeleev")
    heat_of_formation = {}
    for elem_symbol in dataset.dataset_info["atom_decoder"]:
        if include_hof:
            try:
                elem = element(elem_symbol)
                hof = elem.heat_of_formation if elem.heat_of_formation is not None else 0.0
                heat_of_formation[elem_symbol] = hof * 0.0103642  # Convert kJ/mol to eV
                # log.info(f"{elem_symbol}: ΔH_f = {elem.heat_of_formation} eV")
            except Exception as e:
                log.warning(
                    f"While computing energy, could not get heat of formation (HOF) for {elem_symbol} due to error: {e}"
                )
                heat_of_formation[elem_symbol] = 0.0
        else:
            # If not including HOF, set to zero
            heat_of_formation[elem_symbol] = 0.0

    K_matrix = []  # Element counts for each molecule
    E_dft = []  # DFT energy for each molecule
    num_atoms_list = []  # Number of atoms per molecule for avg_num_nodes

    # Loop through dataset to build K matrix and E_dft vector
    log.info("Building K matrix and E_DFT vector from dataset")
    for idx in tqdm(dataset.indices()):
        atoms = dataset.dataset.get_atoms(idx)
        energy_dft = atoms.get_potential_energy()

        # Count elements in this molecule
        element_counts = {}
        for symbol in atoms.get_chemical_symbols():
            if symbol in element_counts:
                element_counts[symbol] += 1
            else:
                element_counts[symbol] = 1

        # Create a fixed-size row vector for this molecule's element counts
        K_row = np.zeros(num_elements, dtype=np.float32)
        for i, element_symbol in enumerate(dataset.dataset_info["atom_decoder"]):
            K_row[i] = element_counts.get(element_symbol, 0)

        K_matrix.append(K_row)
        E_dft.append(energy_dft)
        num_atoms_list.append(len(atoms))

    K = np.array(K_matrix)
    E_dft = np.array(E_dft)
    num_atoms_array = np.array(num_atoms_list)

    log.info(f"K matrix shape: {K.shape}, E_dft shape: {E_dft.shape}")
    log.info(
        f"Energy range: min={E_dft.min():.4f}, max={E_dft.max():.4f}, mean={E_dft.mean():.4f}"
    )

    # Reduce the composition matrix to only features that are non-zero to improve rank
    mask = K.sum(axis=0) != 0.0
    reduced_K = K[:, mask]
    log.info(
        f"Reduced K matrix shape: {reduced_K.shape} (filtered out {K.shape[1] - reduced_K.shape[1]} zero columns)"
    )

    # Replace sklearn with numpy.linalg.lstsq
    log.info("Solving linear regression K*P = E_DFT using numpy.linalg.lstsq")
    coeffs_reduced, residuals, rank, s = np.linalg.lstsq(reduced_K, E_dft, rcond=None)

    # Extract isolated atomic energies E_i,DFT
    E_isolated = {}
    coeffs = np.zeros(K.shape[1])
    coeffs[mask] = coeffs_reduced

    for i, element_symbol in enumerate(dataset.dataset_info["atom_decoder"]):
        E_isolated[element_symbol] = coeffs[i]

    # Calculate predictions and R² score manually
    E_pred = K @ coeffs
    ss_total = np.sum((E_dft - np.mean(E_dft)) ** 2)
    ss_residual = np.sum((E_dft - E_pred) ** 2)
    r2_score = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

    log.info(f"\nLinear regression R² score: {r2_score:.6f}")
    log.info(f"Mean absolute error: {np.mean(np.abs(E_pred - E_dft)):.4f} eV")

    # Now calculate the per-atom energy contributions (E_i,DFT - ΔH_f,i) for normalization
    per_elm_energy = {}
    for element_symbol in dataset.dataset_info["atom_decoder"]:
        per_elm_energy[element_symbol] = (
            E_isolated[element_symbol] - heat_of_formation[element_symbol]
        )

    log.info("\nLearned per-element coefficients:")
    for element_symbol in dataset.dataset_info["atom_decoder"][:10]:
        if element_symbol in per_elm_energy:
            e_isolated = E_isolated[element_symbol]
            hof_val = heat_of_formation[element_symbol]
            p_val = per_elm_energy[element_symbol]
            log.info(
                f"{element_symbol}: E_DFT={e_isolated:.4f} eV, ΔH_f={hof_val:.4f} eV, P=(E_DFT-ΔH_f)={p_val:.4f} eV"
            )

    E_norm = []
    for idx, (k_row, e_orig) in enumerate(zip(K, E_dft)):
        sum_contributions = 0
        for i, element_symbol in enumerate(dataset.dataset_info["atom_decoder"]):
            if k_row[i] > 0:
                sum_contributions += k_row[i] * per_elm_energy[element_symbol]

        e_norm = e_orig - sum_contributions
        E_norm.append(e_norm)

    E_norm = np.array(E_norm)
    log.info(
        f"\nNormalized energy range: min={E_norm.min():.4f}, max={E_norm.max():.4f}, mean={E_norm.mean():.4f}"
    )

    # Compute dataset statistics from normalized energies
    mean_energy = float(np.mean(E_norm))

    if use_rmsd:
        # Use RMSD with Bessel's correction when mean != 0
        rmsd_correction = 0 if mean_energy == 0.0 else 1
        scale = float(
            np.sqrt(np.sum((E_norm - mean_energy) ** 2) / max(len(E_norm) - rmsd_correction, 1))
        )
        scale_method = "RMSD"
    else:
        # Use standard deviation (old method)
        scale = float(np.std(E_norm))
        scale_method = "std"

    dataset_stats = {
        "shift": mean_energy,
        "scale": scale,
        "avg_num_nodes": float(np.mean(num_atoms_array)),
    }

    log.info(
        f"Dataset statistics - Mean: {dataset_stats['shift']:.4f}, Scale ({scale_method}): {dataset_stats['scale']:.4f}, Avg nodes: {dataset_stats['avg_num_nodes']:.1f}"
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        coefficients_data = {
            "per_elm_energy": per_elm_energy,
            "E_isolated": E_isolated,
            "heat_of_formation": heat_of_formation,
            "r2_score": r2_score,
            "mean_abs_error": np.mean(np.abs(E_pred - E_dft)),
            "include_hof": include_hof,
            "dataset_stats": dataset_stats,
        }
        with open(save_path, "wb") as f:
            pickle.dump(coefficients_data, f)
        log.info(f"Saved OMol25 energy per-atom and dataset statistics to {save_path}")

    log.info("OMol25 energy per-atom and stats computed successfully")
    return per_elm_energy, dataset_stats


@typecheck
def normalize_energy(atoms: ase.Atoms, energy_dft: Any, hof_sub_per_atom_energy: dict) -> float:
    """Normalize energy using pre-computed coefficients following HOF reference.

    The formula used is: E_ref = E_DFT - Σ[E_i,DFT - ΔH_f,i]

    Args:
        atoms: ASE Atoms object.
        energy_dft: Original DFT energy.
        hof_sub_per_atom_energy: Dictionary of per-element energy coefficients.
            These contain (E_i,DFT - ΔH_f,i) for each element.

    Returns:
        Normalized energy value (HOF referenced).
    """
    # Get element counts for this molecule (K_i row)
    element_counts = {}
    for symbol in atoms.get_chemical_symbols():
        if symbol in element_counts:
            element_counts[symbol] += 1
        else:
            element_counts[symbol] = 1

    # Calculate sum of per-element contributions: Σ[count_i * (E_i,DFT - ΔH_f,i)]
    # NOTE: energy_coefficients contains the (E_i,DFT - ΔH_f,i) values
    sum_atomic_contributions = 0.0
    for element_symbol, count in element_counts.items():
        if element_symbol in hof_sub_per_atom_energy:
            contribution = count * hof_sub_per_atom_energy[element_symbol]
            sum_atomic_contributions += contribution
        else:
            log.warning(
                f"While normalizing energy, found no coefficient for element {element_symbol}"
            )

    # Apply HOF reference: E_ref = E_DFT - Σ[count_i * (E_i,DFT - ΔH_f,i)]
    normalized_energy = energy_dft - sum_atomic_contributions

    return normalized_energy


@typecheck
def hf_cache_location(subdir: str = "datasets") -> str:
    """Get the Hugging Face cache directory for a specific subdirectory.

    Args:
        subdir: Subdirectory under the main HF cache (default: "datasets").

    Returns:
        Path to the Hugging Face cache subdirectory.
    """
    try:
        from huggingface_hub import constants

        hf_cache_dir = constants.HF_HOME
    except ImportError:
        default_cache = os.path.expanduser("~/.cache/huggingface")
        hf_cache_dir = default_cache

    hf_cache_dir = os.path.join(hf_cache_dir, subdir)  # Add subdirectory

    assert os.path.exists(
        hf_cache_dir
    ), f"Hugging Face cache directory does not exist: {hf_cache_dir}"

    return hf_cache_dir


@typecheck
def hf_datasets_path() -> str:
    """Get the Hugging Face datasets cache directory.

    Returns:
        Path to the Hugging Face datasets cache directory.
    """
    return hf_cache_location(subdir="datasets")


@typecheck
def hf_download_file(
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    local_root: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    decompress: bool = True,
    verbose: bool = False,
) -> Path:
    """
    Download a file from a Hugging Face repo and place it under:

        <local_root>/<repo_id>/<filename>

    By default, local_root is the Hugging Face datasets cache directory.

    If the file looks compressed (.tar.gz, .tgz, .zip, .gz), it is decompressed
    (unless decompress=False) into the same directory and the *local* compressed
    copy is deleted on success.

    The original file in the Hugging Face cache is never modified.

    Args:
        repo_id: E.g. "username/my-dataset"
        filename: Name or path of the file in the repo to download
        repo_type: Type of the HF repo ("dataset" or "model")
        local_root: Root directory to store and decompress the downloaded files
        overwrite: If True, redownload even if the file already exists locally
        decompress: If True, decompress the file if it is compressed
        verbose: If True, print progress and info messages

    Returns:
        Path to the decompressed content:
          - For archives (.tar.gz, .tgz, .zip): directory containing extracted files
          - For .gz: path to the decompressed file
          - For uncompressed files: path to the downloaded file
    """
    if local_root is None:
        local_root = hf_datasets_path()

    local_root = Path(local_root)

    # Make repo_id filesystem-friendly (i.e., replace "/" with "__")
    repo_dir = local_root / repo_id.replace("/", "__")

    target_path = repo_dir / filename

    # Check if Hugging Face file is compressed
    base_name = None
    all_extensions = [".tar.gz", ".tgz", ".tar", ".zip", ".gz"]
    for ext in all_extensions:
        if filename.lower().endswith(ext):
            base_name = filename[: -len(ext)]
            break

    # Check if decompressed file already exists, if so skip download
    if base_name is not None:  # NOTE: File is not compressed, so check for extracted file
        decompressed_path = repo_dir / base_name
        if decompressed_path.exists() and not overwrite:
            if verbose:
                log.info(
                    f"Decompressed path {decompressed_path} already exists and overwrite is False. Skipping download. To re-download, set overwrite=True."
                )
            return decompressed_path
    else:
        # Check if file already exists
        if target_path.exists() and not overwrite:
            if verbose:
                log.info(
                    f"File {target_path} already exists and overwrite is False. Skipping download. To re-download, set overwrite=True."
                )
            return target_path

    # 1) Download into Hugging Face cache
    log.info(f"Downloading {filename} from repo {repo_id} (type={repo_type})...")
    cached_path = Path(
        hf_hub_download(  # nosec
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
        )
    )

    # 2) Copy into local working directory
    target_path = repo_dir / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Copying to local path {target_path}...")
    shutil.copy2(cached_path, target_path)

    # 3) Detect compression and decompress if needed
    name_lower = target_path.name.lower()

    if not decompress:
        log.info(f"Decompression disabled, returning downloaded file at {target_path}")
        return target_path

    # Build helper to extract tar archives
    def _extract_tar(mode: str) -> Path:
        # Strip .tar.gz / .tgz to get folder name
        if name_lower.endswith(".tar.gz"):
            base_name = target_path.name[: -len(".tar.gz")]
        elif name_lower.endswith(".tgz"):
            base_name = target_path.name[: -len(".tgz")]
        else:
            base_name = target_path.stem  # just ".tar"

        extract_dir = target_path.parent / base_name
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(target_path, mode) as tar:
            members = tar.getmembers()
            # Extract with progress bar
            for member in tqdm(members, desc=f"Extracting {target_path.name}", unit="files"):
                tar.extract(member, path=extract_dir)

        # Remove local compressed file after successful extraction
        target_path.unlink()
        return extract_dir

    # .tar.gz / .tgz
    if name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
        log.info(f"Detected tar.gz compression for {target_path.name}, extracting...")
        return _extract_tar("r:gz")

    # .tar (uncompressed tar)
    elif name_lower.endswith(".tar"):
        log.info(f"Detected tar compression for {target_path.name}, extracting...")
        return _extract_tar("r:")

    # .zip
    elif name_lower.endswith(".zip"):
        log.info(f"Detected zip compression for {target_path.name}, extracting...")
        base_name = target_path.stem  # Strip .zip
        extract_dir = target_path.parent / base_name
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(target_path, "r") as zf:
            # Extract with progress bar for zip files
            for file_info in tqdm(
                zf.filelist, desc=f"Extracting {target_path.name}", unit="files"
            ):
                zf.extract(file_info, path=extract_dir)

        target_path.unlink()
        return extract_dir

    # .gz (but not .tar.gz, handled above)
    elif name_lower.endswith(".gz"):
        log.info(f"Detected gz compression for {target_path.name}, extracting...")
        out_path = target_path.with_suffix("")  # Remove .gz

        file_size = target_path.stat().st_size
        with gzip.open(target_path, "rb") as f_in, out_path.open("wb") as f_out:
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Extracting {target_path.name}",
            ) as pbar:
                while True:
                    chunk = f_in.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))

        target_path.unlink()
        return out_path

    # Not compressed (or unsupported): just return the copied file
    return target_path
