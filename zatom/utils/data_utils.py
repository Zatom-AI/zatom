import os
import pickle  # nosec
import tarfile
from typing import Any, Dict

import ase
import numpy as np
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
        log.info(f"Loading energy normalization coefficients from {coef_path}")
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
        log.info(f"Loading energy normalization coefficients from {coef_path}")
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
        log.info(f"Saved dataset statistics to {coef_path}")

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
    log.info("Computing per-element energy for referencing")
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
        log.info(f"Saved energy per-atom and dataset statistics to {save_path}")

    log.info("Energy per-atom and stats computed successfully")
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
