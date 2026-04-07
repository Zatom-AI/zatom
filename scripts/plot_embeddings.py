"""Create all embedding plots from a single shared inference pass.

This script samples structures once, extracts per-atom embeddings once per selected layer,
and then reuses those cached embeddings to create:
1. System/source plot
2. Element-colored plot
3. System-size plot (Optional)
"""

import argparse
import glob
import os
from itertools import combinations
from typing import Dict, List, Tuple

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from torch_geometric.datasets import QM9
from umap import UMAP

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Publication-quality settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Internal defaults
DEFAULT_CKPT_PATH = "checkpoints/zatom_1_joint_paper_weights.ckpt"
DEFAULT_CKPT_TYPE = "joint"
DEFAULT_CONFIG_PATH = "configs"
DEFAULT_NUM_SAMPLES = 100
DEFAULT_LAYERS_TO_EXTRACT = [15]
DEFAULT_RANDOM_SEED = 42
DEFAULT_TIMESTEP = 0.5
DEFAULT_NORMALIZATION = "clip_std"
DEFAULT_PROJECTION = "pca"
DEFAULT_PRE_UMAP_PCA_COMPONENTS = 50
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_METRIC = "euclidean"

ELEMENT_FILTER = {6: "C", 7: "N", 8: "O", 9: "F"}
ELEMENT_COLORS = {"C": "#3498DB", "N": "#E67E22", "O": "#27AE60", "F": "#E74C3C"}

COLOR_MAP_SYSTEM_SOURCE = {
    ("Crystal", "Dataset"): "#1E5AA8",
    ("Crystal", "Generated"): "#7EB6FF",
    ("Molecule", "Dataset"): "#D35400",
    ("Molecule", "Generated"): "#F5B041",
}

CKPT_PATHS = {
    "joint": "checkpoints/zatom_1_joint_paper_weights.ckpt",
    "qm9_only": "checkpoints/zatom_1_qm9_only_pretraining_paper_weights.ckpt",
    "mp20_only": "checkpoints/zatom_1_mp20_only_pretraining_paper_weights.ckpt",
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for sampling and projection reproducibility.",
    )
    runtime_group.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="How many structures to sample from each source.",
    )
    runtime_group.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS_TO_EXTRACT,
        help="Transformer layers to visualize.",
    )
    runtime_group.add_argument(
        "--timestep",
        type=float,
        default=DEFAULT_TIMESTEP,
        help="Timestep passed into the embedding extractor.",
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--ckpt-path",
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint used for embedding extraction. Overrides --ckpt_type unless --ckpt_type none.",
    )
    model_group.add_argument(
        "--ckpt_type",
        choices=["joint", "qm9_only", "mp20_only", "none"],
        default=DEFAULT_CKPT_TYPE,
        help="Which pretrained checkpoint to use. 'none' leaves the model untrained.",
    )
    model_group.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Hydra config directory used to instantiate the model.",
    )

    projection_group = parser.add_argument_group("Projection")
    projection_group.add_argument(
        "--normalization",
        choices=["none", "zscore", "l2", "l2_zscore", "clip_std"],
        default=DEFAULT_NORMALIZATION,
        help="How to normalize embeddings before projection.",
    )
    projection_group.add_argument(
        "--projection",
        choices=["umap", "pca", "pca_umap"],
        default=DEFAULT_PROJECTION,
        help="Dimensionality reduction method to apply after embedding extraction.",
    )
    projection_group.add_argument(
        "--pre-umap-pca-components",
        type=int,
        default=DEFAULT_PRE_UMAP_PCA_COMPONENTS,
        help="Number of PCA components to keep before UMAP when --projection pca_umap.",
    )
    projection_group.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=DEFAULT_UMAP_N_NEIGHBORS,
        help="UMAP neighborhood size.",
    )
    projection_group.add_argument(
        "--umap-min-dist",
        type=float,
        default=DEFAULT_UMAP_MIN_DIST,
        help="UMAP minimum distance.",
    )
    projection_group.add_argument(
        "--umap-metric",
        default=DEFAULT_UMAP_METRIC,
        help="UMAP distance metric.",
    )
    return parser.parse_args()


def normalize_embeddings_for_projection(embeddings: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize embeddings before dimensionality reduction."""
    if normalization == "none":
        return embeddings
    if normalization == "zscore":
        return StandardScaler().fit_transform(embeddings)
    if normalization == "l2":
        return normalize(embeddings, norm="l2")
    if normalization == "l2_zscore":
        return StandardScaler().fit_transform(normalize(embeddings, norm="l2"))
    if normalization == "clip_std":
        mean = np.mean(embeddings, axis=1, keepdims=True)
        std = np.std(embeddings, axis=1, keepdims=True)
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        return np.clip(embeddings, a_min=lower_bound, a_max=upper_bound)
    raise ValueError(f"Unsupported normalization: {normalization}")


def resolve_ckpt_path(ckpt_type: str, ckpt_path: str | None) -> str | None:
    """Resolve the checkpoint path from the selected checkpoint type."""
    if ckpt_type == "none":
        return None
    if ckpt_path is not None and ckpt_path != DEFAULT_CKPT_PATH:
        return ckpt_path
    return CKPT_PATHS[ckpt_type]


def get_ckpt_tag(ckpt_type: str) -> str:
    """Return a filename-safe tag for the selected checkpoint mode."""
    return ckpt_type


def initialize_model(ckpt_path: str | None, config_path: str) -> torch.nn.Module:
    """Load the model and optionally restore checkpoint weights."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.abspath(config_path), version_base="1.3"):
        cfg = compose(
            config_name="train_fm.yaml",
            overrides=[
                "model/architecture=tft_80M",
                "globals.spatial_dim=3",
                "globals.max_num_elements=100",
                "globals.num_dataset_classes=2",
                "globals.num_spacegroup_classes=230",
                "globals.num_global_properties=19",
                "globals.max_num_atoms=2048",
                "globals.rope_base=null",
            ],
        )

    model = hydra.utils.instantiate(cfg.model.architecture).to(device)
    model.eval()

    print(
        f"Model will be loaded from checkpoint: {ckpt_path if ckpt_path is not None else 'None (untrained)'}"
    )

    if ckpt_path is None:
        return model

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)  # nosec
    updated_state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        new_key = key[len("model.") :] if key.startswith("model.") else key
        if new_key in model.state_dict() and value.shape == model.state_dict()[new_key].shape:
            updated_state_dict[new_key] = value

    model.load_state_dict(updated_state_dict, strict=False)
    return model


def load_validation_datasets() -> Tuple[QM9, object]:
    """Load the validation splits used by the plotting scripts."""
    from zatom.data.components.mp20_dataset import MP20

    qm9_dataset = QM9(root="data/qm9/").shuffle()
    mp20_dataset = MP20(root="data/mp_20/")

    qm9_dataset = qm9_dataset[100000:118000]
    mp20_dataset = mp20_dataset[27138 : 27138 + 9046]
    return qm9_dataset, mp20_dataset


def select_samples(
    qm9_dataset: QM9,
    mp20_dataset: object,
    num_samples: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray | List[str]]:
    """Select the structures once so all plots reuse the same inference pass."""
    gen_crystals_dir = "data/gen_samples/zatom_1_materials_mp20"
    gen_molecules_dir = "data/gen_samples/zatom_1_molecules_qm9"

    pdb_files = sorted(glob.glob(os.path.join(gen_molecules_dir, "molecule_*.pdb")))
    cif_files = sorted(glob.glob(os.path.join(gen_crystals_dir, "crystal_*.cif")))

    return {
        "qm9_indices": rng.choice(len(qm9_dataset), num_samples, replace=False),
        "mp20_indices": rng.choice(len(mp20_dataset), num_samples, replace=False),
        "pdb_files": rng.choice(pdb_files, num_samples, replace=False).tolist(),
        "cif_files": rng.choice(cif_files, num_samples, replace=False).tolist(),
    }


def build_atom_metadata(
    atom_type: int,
    system: str,
    source: str,
    system_size: int,
    smallest_angle: float | None,
) -> dict:
    """Build the unified metadata record used by all plots."""
    return {
        "system": system,
        "source": source,
        "system_size": system_size,
        "smallest_angle": smallest_angle if system == "Crystal" else None,
        "element": ELEMENT_FILTER.get(int(atom_type)),
    }


def extract_all_embeddings(
    model: torch.nn.Module,
    qm9_dataset: QM9,
    mp20_dataset: object,
    selected_samples: Dict[str, np.ndarray | List[str]],
    layer: int,
    timestep: float,
) -> Tuple[np.ndarray, List[dict]]:
    """Extract per-atom embeddings once and annotate them for all downstream plots."""
    from zatom.utils.figure_utils import (
        extract_embeddings,
        load_cif_to_tensors,
        load_pdb_to_tensors,
    )

    embeddings_list = []
    metadata_list = []

    for idx in selected_samples["qm9_indices"]:
        sample = qm9_dataset[int(idx)]
        num_atoms = sample.num_nodes
        centered_pos = sample.pos - sample.pos.mean(dim=0, keepdim=True)
        with torch.no_grad():
            emb = extract_embeddings(
                model,
                sample.z.unsqueeze(0),
                centered_pos.unsqueeze(0)
                * 0.5,  # NOTE: Need to change based on pretrained checkpoints
                is_periodic=False,
                dataset_idx=1,
                return_pooled=False,
                layer=layer,
                timestep=timestep,
            )
        atom_types = sample.z.cpu().numpy()
        for i, atom_type in enumerate(atom_types):
            embeddings_list.append(emb[0, i].cpu().numpy())
            metadata_list.append(
                build_atom_metadata(
                    atom_type=atom_type,
                    system="Molecule",
                    source="Dataset",
                    system_size=num_atoms,
                    smallest_angle=None,
                )
            )

    for idx in selected_samples["mp20_indices"]:
        sample = mp20_dataset[int(idx)]
        num_atoms = sample.atom_types.shape[0]
        smallest_angle = float(np.min(sample.angles_radians.cpu().numpy()) * (180.0 / np.pi))
        with torch.no_grad():
            emb = extract_embeddings(
                model,
                sample.atom_types.unsqueeze(0),
                torch.zeros(1, num_atoms, 3),
                is_periodic=True,
                frac_coords=sample.frac_coords.unsqueeze(0),
                lengths_scaled=sample.lengths_scaled.unsqueeze(0),
                angles_radians=sample.angles_radians.unsqueeze(0),
                dataset_idx=0,
                spacegroup=0,
                return_pooled=False,
                layer=layer,
                timestep=timestep,
            )
        atom_types = sample.atom_types.cpu().numpy()
        for i, atom_type in enumerate(atom_types):
            embeddings_list.append(emb[0, i].cpu().numpy())
            metadata_list.append(
                build_atom_metadata(
                    atom_type=atom_type,
                    system="Crystal",
                    source="Dataset",
                    system_size=num_atoms,
                    smallest_angle=smallest_angle,
                )
            )
    for pdb_file in selected_samples["pdb_files"]:
        try:
            atom_types, pos = load_pdb_to_tensors(pdb_file)
            num_atoms = len(atom_types)
            centered_pos = pos - pos.mean(dim=0, keepdim=True)
            with torch.no_grad():
                emb = extract_embeddings(
                    model,
                    atom_types.unsqueeze(0),
                    centered_pos.unsqueeze(0)
                    * 0.5,  # NOTE: Need to change based on pretrained checkpoints
                    is_periodic=False,
                    dataset_idx=1,
                    return_pooled=False,
                    layer=layer,
                    timestep=timestep,
                )
            for i, atom_type in enumerate(atom_types.cpu().numpy()):
                embeddings_list.append(emb[0, i].cpu().numpy())
                metadata_list.append(
                    build_atom_metadata(
                        atom_type=atom_type,
                        system="Molecule",
                        source="Generated",
                        system_size=num_atoms,
                        smallest_angle=None,
                    )
                )
        except Exception as exc:
            print(f"Error processing {pdb_file}, skipping. Exception: {exc}")

    for cif_file in selected_samples["cif_files"]:
        try:
            atom_types, frac_coords, lengths_scaled, angles_radians = load_cif_to_tensors(cif_file)
            num_atoms = len(atom_types)
            smallest_angle = float(np.min(angles_radians.cpu().numpy()) * (180.0 / np.pi))
            with torch.no_grad():
                emb = extract_embeddings(
                    model,
                    atom_types.unsqueeze(0),
                    torch.zeros(1, num_atoms, 3),
                    is_periodic=True,
                    frac_coords=frac_coords.unsqueeze(0),
                    lengths_scaled=lengths_scaled.unsqueeze(0).unsqueeze(0),
                    angles_radians=angles_radians.unsqueeze(0).unsqueeze(0),
                    dataset_idx=0,
                    spacegroup=0,
                    return_pooled=False,
                    layer=layer,
                    timestep=timestep,
                )
            for i, atom_type in enumerate(atom_types.cpu().numpy()):
                embeddings_list.append(emb[0, i].cpu().numpy())
                metadata_list.append(
                    build_atom_metadata(
                        atom_type=atom_type,
                        system="Crystal",
                        source="Generated",
                        system_size=num_atoms,
                        smallest_angle=smallest_angle,
                    )
                )
        except Exception as exc:
            print(f"Error processing {cif_file}, skipping. Exception: {exc}")

    return np.asarray(embeddings_list), metadata_list


def get_projection_axis_labels(projection: str) -> Tuple[str, str]:
    """Return axis labels for the selected projection."""
    if projection == "pca":
        return "Principal Component 1", "Principal Component 2"
    return "UMAP 1", "UMAP 2"


def get_projection_tag(projection: str) -> str:
    """Return a filename-safe tag for the selected projection."""
    return {
        "umap": "UMAP",
        "pca": "PCA",
        "pca_umap": "PCA_UMAP",
    }[projection]


def project_embeddings(
    embeddings: np.ndarray,
    normalization: str,
    projection: str,
    seed: int,
    pre_umap_pca_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
) -> np.ndarray:
    """Project embeddings into 2D once the model forward pass is finished."""
    scaled_embeddings = normalize_embeddings_for_projection(embeddings, normalization)

    if projection == "pca":
        projector = PCA(n_components=2, svd_solver="full", random_state=seed)
        return projector.fit_transform(scaled_embeddings)

    if projection == "umap":
        projector = UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            metric=umap_metric,
            min_dist=umap_min_dist,
            random_state=seed,
        )
        return projector.fit_transform(scaled_embeddings)

    if projection == "pca_umap":
        pca_components = min(
            pre_umap_pca_components,
            scaled_embeddings.shape[1],
        )
        pca_components = max(pca_components, 2)
        pca_projector = PCA(
            n_components=pca_components,
            svd_solver="full",
            random_state=seed,
        )
        pca_embeddings = pca_projector.fit_transform(scaled_embeddings)
        umap_projector = UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            metric=umap_metric,
            min_dist=umap_min_dist,
            random_state=seed,
        )
        return umap_projector.fit_transform(pca_embeddings)

    raise ValueError(f"Unsupported projection: {projection}")


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save the figure next to this script."""
    fig.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _mean_pairwise_distance(points: List[np.ndarray]) -> float:
    """Return the mean pairwise Euclidean distance for a list of 2D points."""
    if len(points) < 2:
        return 0.0
    return float(
        np.mean(
            [np.linalg.norm(point_a - point_b) for point_a, point_b in combinations(points, 2)]
        )
    )


def compute_clustering_and_transfer_learning_metrics(
    embeddings_2d: np.ndarray,
    metadata: List[dict],
) -> Dict[str, float]:
    """Measure element separation while rewarding consistency across systems and sources."""
    points_by_element: Dict[str, List[np.ndarray]] = {element: [] for element in ELEMENT_COLORS}
    points_by_group: Dict[Tuple[str, str, str], List[np.ndarray]] = {}

    for point, meta in zip(embeddings_2d, metadata):
        element = meta["element"]
        if element is None:
            continue
        points_by_element.setdefault(element, []).append(point)
        group_key = (element, meta["system"], meta["source"])
        points_by_group.setdefault(group_key, []).append(point)

    element_centroids: Dict[str, np.ndarray] = {}
    element_spreads: List[float] = []
    for element, points in points_by_element.items():
        if len(points) < 2:
            continue
        point_array = np.asarray(points)
        centroid = point_array.mean(axis=0)
        element_centroids[element] = centroid
        element_spreads.extend(np.linalg.norm(point_array - centroid, axis=1).tolist())

    inter_element_separation = _mean_pairwise_distance(list(element_centroids.values()))
    intra_element_spread = float(np.mean(element_spreads)) if element_spreads else 0.0

    system_alignment_distances: List[float] = []
    for element in ELEMENT_COLORS:
        for source in ["Dataset", "Generated"]:
            group_centroids = {}
            for system in ["Crystal", "Molecule"]:
                points = points_by_group.get((element, system, source))
                if points:
                    group_centroids[system] = np.asarray(points).mean(axis=0)
            if len(group_centroids) == 2:
                system_alignment_distances.append(
                    float(np.linalg.norm(group_centroids["Crystal"] - group_centroids["Molecule"]))
                )

    source_alignment_distances: List[float] = []
    for element in ELEMENT_COLORS:
        for system in ["Crystal", "Molecule"]:
            group_centroids = {}
            for source in ["Dataset", "Generated"]:
                points = points_by_group.get((element, system, source))
                if points:
                    group_centroids[source] = np.asarray(points).mean(axis=0)
            if len(group_centroids) == 2:
                source_alignment_distances.append(
                    float(
                        np.linalg.norm(group_centroids["Dataset"] - group_centroids["Generated"])
                    )
                )

    system_alignment = (
        float(np.mean(system_alignment_distances)) if system_alignment_distances else 0.0
    )
    source_alignment = (
        float(np.mean(source_alignment_distances)) if source_alignment_distances else 0.0
    )
    denominator = intra_element_spread + system_alignment + source_alignment + 1e-8

    return {
        "score": inter_element_separation / denominator,
        "inter_element_separation": inter_element_separation,
        "intra_element_spread": intra_element_spread,
        "system_alignment": system_alignment,
        "source_alignment": source_alignment,
    }


def add_element_metrics_annotation(ax: plt.Axes, element_metrics: Dict[str, float]) -> None:
    """Render a compact, publication-ready summary of clustering and transfer learning metrics."""
    metrics_lines = [
        ("Overall Score (↑)", element_metrics["score"]),
        ("Inter-Element Separation (↑)", element_metrics["inter_element_separation"]),
        ("Intra-Element Spread (↓)", element_metrics["intra_element_spread"]),
        ("Dataset-Generated Shift (↓)", element_metrics["source_alignment"]),
        ("Crystal-Molecule Shift (↓)", element_metrics["system_alignment"]),
    ]
    label_width = max(len(label) for label, _ in metrics_lines)
    annotation_text = "\n".join(
        [r"$\bf{Clustering\ &\ Transfer\ Learning}$"]
        + [f"{label:<{label_width}}  {value:.2f}" for label, value in metrics_lines]
    )
    ax.text(
        0.02,
        0.98,
        annotation_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        linespacing=1.25,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.95,
            "edgecolor": "0.7",
        },
    )


def create_system_source_plot(
    embeddings: np.ndarray,
    metadata: List[dict],
    layer: int,
    timestep: float,
    normalization: str,
    projection: str,
    ckpt_type: str = DEFAULT_CKPT_TYPE,
    embeddings_2d: np.ndarray | None = None,
) -> None:
    """Plot the full embedding set colored by system and source."""
    if embeddings_2d is None:
        embeddings_2d = project_embeddings(
            embeddings,
            normalization,
            projection,
            DEFAULT_RANDOM_SEED,
            DEFAULT_PRE_UMAP_PCA_COMPONENTS,
            DEFAULT_UMAP_N_NEIGHBORS,
            DEFAULT_UMAP_MIN_DIST,
            DEFAULT_UMAP_METRIC,
        )
    fig, ax = plt.subplots(figsize=(10, 8))

    markers = {"Crystal": "o", "Molecule": "x"}
    for system, source in [
        ("Crystal", "Generated"),
        ("Molecule", "Generated"),
        ("Crystal", "Dataset"),
        ("Molecule", "Dataset"),
    ]:
        mask = np.array([m["system"] == system and m["source"] == source for m in metadata])
        if mask.sum() == 0:
            continue
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=COLOR_MAP_SYSTEM_SOURCE[(system, source)],
            s=30,
            marker=markers[system],
            alpha=0.7,
            linewidths=1.5 if system == "Molecule" else 0,
            edgecolors=(
                "none" if system == "Crystal" else COLOR_MAP_SYSTEM_SOURCE[(system, source)]
            ),
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_MAP_SYSTEM_SOURCE[("Crystal", "Dataset")],
            markersize=8,
            linestyle="None",
            label="Crystal (Dataset)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_MAP_SYSTEM_SOURCE[("Crystal", "Generated")],
            markersize=8,
            linestyle="None",
            label="Crystal (Generated)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLOR_MAP_SYSTEM_SOURCE[("Molecule", "Dataset")],
            linestyle="None",
            markersize=8,
            markeredgewidth=2,
            label="Molecule (Dataset)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLOR_MAP_SYSTEM_SOURCE[("Molecule", "Generated")],
            linestyle="None",
            markersize=8,
            markeredgewidth=2,
            label="Molecule (Generated)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95)
    xlabel, ylabel = get_projection_axis_labels(projection)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_figure(
        fig,
        f"{get_projection_tag(projection)}_{get_ckpt_tag(ckpt_type)}_latent_space_layer{layer}_t{int(timestep * 100)}_{normalization}.pdf",
    )


def create_element_plot(
    embeddings: np.ndarray,
    metadata: List[dict],
    layer: int,
    timestep: float,
    normalization: str,
    projection: str,
    ckpt_type: str = DEFAULT_CKPT_TYPE,
    embeddings_2d: np.ndarray | None = None,
) -> None:
    """Plot the element-filtered embedding set."""
    if embeddings_2d is None:
        embeddings_2d = project_embeddings(
            embeddings,
            normalization,
            projection,
            DEFAULT_RANDOM_SEED,
            DEFAULT_PRE_UMAP_PCA_COMPONENTS,
            DEFAULT_UMAP_N_NEIGHBORS,
            DEFAULT_UMAP_MIN_DIST,
            DEFAULT_UMAP_METRIC,
        )

    mask = np.array([m["element"] is not None for m in metadata])
    if mask.sum() == 0:
        print(f"No C/N/O/F atoms found for layer {layer}; skipping element plot.")
        return

    embeddings_2d = embeddings_2d[mask]
    filtered_metadata = [metadata[i] for i in np.where(mask)[0]]
    element_metrics = compute_clustering_and_transfer_learning_metrics(
        embeddings_2d, filtered_metadata
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    markers = {"Crystal": "o", "Molecule": "x"}
    sizes = {"Dataset": 80, "Generated": 20}

    for source in ["Generated", "Dataset"]:
        for element in ["C", "N", "O", "F"]:
            for system in ["Crystal", "Molecule"]:
                point_mask = np.array(
                    [
                        m["element"] == element and m["system"] == system and m["source"] == source
                        for m in filtered_metadata
                    ]
                )
                if point_mask.sum() == 0:
                    continue
                ax.scatter(
                    embeddings_2d[point_mask, 0],
                    embeddings_2d[point_mask, 1],
                    c=ELEMENT_COLORS[element],
                    s=sizes[source],
                    marker=markers[system],
                    alpha=0.7,
                    linewidths=1.5 if system == "Molecule" else 0,
                    edgecolors=("none" if system == "Crystal" else ELEMENT_COLORS[element]),
                )

    legend_elements = [
        Line2D([0], [0], marker="None", linestyle="None", label=r"$\bf{Element}$"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ELEMENT_COLORS["C"],
            markersize=8,
            linestyle="None",
            label="C",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ELEMENT_COLORS["N"],
            markersize=8,
            linestyle="None",
            label="N",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ELEMENT_COLORS["O"],
            markersize=8,
            linestyle="None",
            label="O",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ELEMENT_COLORS["F"],
            markersize=8,
            linestyle="None",
            label="F",
        ),
        Line2D([0], [0], marker="None", linestyle="None", label=r"$\bf{Source}$"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            linestyle="None",
            markersize=10,
            label="Dataset",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            linestyle="None",
            markersize=5,
            label="Generated",
        ),
        Line2D([0], [0], marker="None", linestyle="None", label=r"$\bf{System}$"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            linestyle="None",
            markersize=8,
            label="Crystal",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="gray",
            linestyle="None",
            markersize=8,
            markeredgewidth=2,
            label="Molecule",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95)
    add_element_metrics_annotation(ax, element_metrics)
    xlabel, ylabel = get_projection_axis_labels(projection)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print(
        "Clustering & Transfer Learning Score "
        f"(layer={layer}, projection={projection}, ckpt={ckpt_type}): "
        f"{element_metrics['score']:.3f} = "
        f"{element_metrics['inter_element_separation']:.3f} / "
        f"({element_metrics['intra_element_spread']:.3f} + "
        f"{element_metrics['system_alignment']:.3f} + "
        f"{element_metrics['source_alignment']:.3f})"
    )
    save_figure(
        fig,
        f"{get_projection_tag(projection)}_{get_ckpt_tag(ckpt_type)}_element_embeddings_layer{layer}_t{int(timestep * 100)}_{normalization}.pdf",
    )


def create_system_size_plot(
    embeddings: np.ndarray,
    metadata: List[dict],
    layer: int,
    timestep: float,
    normalization: str,
    projection: str,
    ckpt_type: str = DEFAULT_CKPT_TYPE,
    embeddings_2d: np.ndarray | None = None,
) -> None:
    """Plot crystals and molecules colored by their system size."""
    if embeddings_2d is None:
        embeddings_2d = project_embeddings(
            embeddings,
            normalization,
            projection,
            DEFAULT_RANDOM_SEED,
            DEFAULT_PRE_UMAP_PCA_COMPONENTS,
            DEFAULT_UMAP_N_NEIGHBORS,
            DEFAULT_UMAP_MIN_DIST,
            DEFAULT_UMAP_METRIC,
        )
    fig, ax = plt.subplots(figsize=(10, 8))

    system_types = np.array([m["system"] for m in metadata])
    crystal_mask = system_types == "Crystal"
    molecule_mask = system_types == "Molecule"

    crystal_sizes = np.array([m["system_size"] for m in metadata if m["system"] == "Crystal"])
    molecule_sizes = np.array([m["system_size"] for m in metadata if m["system"] == "Molecule"])

    cmap_crystal = plt.cm.plasma
    cmap_molecule = plt.cm.viridis

    norm_crystal = (
        Normalize(vmin=crystal_sizes.min(), vmax=crystal_sizes.max())
        if len(crystal_sizes) > 0
        else None
    )
    norm_molecule = (
        Normalize(vmin=molecule_sizes.min(), vmax=molecule_sizes.max())
        if len(molecule_sizes) > 0
        else None
    )

    if crystal_mask.sum() > 0:
        colors_crystal = cmap_crystal(norm_crystal(crystal_sizes))
        ax.scatter(
            embeddings_2d[crystal_mask, 0],
            embeddings_2d[crystal_mask, 1],
            c=colors_crystal,
            s=30,
            marker="o",
            alpha=0.7,
            linewidths=0,
            edgecolors="none",
        )

    if molecule_mask.sum() > 0:
        colors_molecule = cmap_molecule(norm_molecule(molecule_sizes))
        ax.scatter(
            embeddings_2d[molecule_mask, 0],
            embeddings_2d[molecule_mask, 1],
            c=colors_molecule,
            s=30,
            marker="x",
            alpha=0.7,
            linewidths=1.5,
        )

    legend_width = 0.25
    legend_height = 0.20
    legend_margin = 0.02
    legend_ax = ax.inset_axes(
        [
            1 - legend_width - legend_margin,
            1 - legend_height - legend_margin,
            legend_width,
            legend_height,
        ]
    )
    legend_ax.set_facecolor("white")
    legend_ax.patch.set_alpha(0.95)
    for spine in legend_ax.spines.values():
        spine.set_edgecolor("0.7")
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    legend_ax.scatter(0.12, 0.83, marker="o", s=34, c="gray")
    legend_ax.text(0.18, 0.83, "Crystal", va="center", fontsize=8.5)
    legend_ax.scatter(0.52, 0.83, marker="x", s=34, c="gray", linewidths=1.4)
    legend_ax.text(0.58, 0.83, "Molecule", va="center", fontsize=8.5)

    if norm_crystal is not None:
        sm_crystal = ScalarMappable(cmap=cmap_crystal, norm=norm_crystal)
        sm_crystal.set_array([])
        legend_ax.text(0.10, 0.58, "Crystal size", va="center", fontsize=7.5)
        cax_crystal = legend_ax.inset_axes([0.10, 0.42, 0.80, 0.12])
        cbar_crystal = fig.colorbar(sm_crystal, cax=cax_crystal, orientation="horizontal")
        cbar_crystal.ax.tick_params(labelsize=6.5, pad=1, length=2)

    if norm_molecule is not None:
        sm_molecule = ScalarMappable(cmap=cmap_molecule, norm=norm_molecule)
        sm_molecule.set_array([])
        legend_ax.text(0.10, 0.26, "Molecule size", va="center", fontsize=7.5)
        cax_molecule = legend_ax.inset_axes([0.10, 0.10, 0.80, 0.12])
        cbar_molecule = fig.colorbar(sm_molecule, cax=cax_molecule, orientation="horizontal")
        cbar_molecule.ax.tick_params(labelsize=6.5, pad=1, length=2)

    xlabel, ylabel = get_projection_axis_labels(projection)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_figure(
        fig,
        f"{get_projection_tag(projection)}_{get_ckpt_tag(ckpt_type)}_system_size_layer{layer}_t{int(timestep * 100)}_{normalization}.pdf",
    )


def main() -> None:
    """Run shared embedding extraction and generate all plots."""
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    rng = np.random.default_rng(args.seed)

    ckpt_path = resolve_ckpt_path(args.ckpt_type, args.ckpt_path)
    model = initialize_model(ckpt_path, args.config_path)
    qm9_dataset, mp20_dataset = load_validation_datasets()
    selected_samples = select_samples(
        qm9_dataset=qm9_dataset,
        mp20_dataset=mp20_dataset,
        num_samples=args.num_samples,
        rng=rng,
    )

    for layer in args.layers:
        embeddings, metadata = extract_all_embeddings(
            model=model,
            qm9_dataset=qm9_dataset,
            mp20_dataset=mp20_dataset,
            selected_samples=selected_samples,
            layer=layer,
            timestep=args.timestep,
        )
        embeddings_2d = project_embeddings(
            embeddings=embeddings,
            normalization=args.normalization,
            projection=args.projection,
            seed=args.seed,
            pre_umap_pca_components=args.pre_umap_pca_components,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
        )
        create_system_source_plot(
            embeddings,
            metadata,
            layer,
            args.timestep,
            args.normalization,
            args.projection,
            ckpt_type=args.ckpt_type,
            embeddings_2d=embeddings_2d,
        )
        create_element_plot(
            embeddings,
            metadata,
            layer,
            args.timestep,
            args.normalization,
            args.projection,
            ckpt_type=args.ckpt_type,
            embeddings_2d=embeddings_2d,
        )
        # create_system_size_plot(
        #     embeddings,
        #     metadata,
        #     layer,
        #     args.timestep,
        #     args.normalization,
        #     args.projection,
        #     ckpt_type=args.ckpt_type,
        #     embeddings_2d=embeddings_2d,
        # )


if __name__ == "__main__":
    main()
