"""Copyright (c) Meta Platforms, Inc. and affiliates.

Adapted from:
- CDVAE: https://github.com/txie-93/cdvae
- DiffCSP: https://github.com/jiaor17/DiffCSP
"""

import copy
import faulthandler
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_umap
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from torch_geometric.data import Data
from torch_scatter import segment_coo, segment_csr

from zatom.utils import pylogger
from zatom.utils.typing_utils import typecheck

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

faulthandler.enable()

FLOAT_TYPE = float | np.float32 | np.float64 | torch.FloatTensor

# Tensor of unit cells. Assumes 27 cells in -1, 0, 1 offsets in the x and y dimensions.
# Note that differing from OCP, we have 27 offsets here because we are in 3D.
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

EPSILON = 1e-5

chemical_symbols = [
    # 0
    "X",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    # 5
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    # 6
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    # 7
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)


@typecheck
def build_crystal(crystal_str: str, niggli: bool = True, primitive: bool = False) -> Structure:
    """Build crystal from CIF string.

    Args:
        crystal_str: CIF string representation of the crystal structure.
        niggli: Whether to apply Niggli reduction.
        primitive: Whether to convert to primitive cell.

    Returns:
        The constructed crystal structure.
    """
    crystal = Structure.from_str(crystal_str, fmt="cif")

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    # NOTE: Match is guaranteed because CIF only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)
    return canonical_crystal


@typecheck
def refine_spacegroup(crystal: Structure, tol: float = 0.01) -> Tuple[Structure, int]:
    """Refine the space group of a crystal structure.

    Args:
        crystal: The crystal structure to refine.
        tol: The tolerance for symmetry detection.

    Returns:
        The refined crystal structure and its space group number.
    """
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_conventional_standard_structure()
    space_group = spga.get_space_group_number()
    crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return crystal, space_group


@typecheck
def get_symmetry_info(crystal: Structure, tol: float = 0.01) -> Tuple[Structure, dict]:
    """Get the symmetry information of a crystal structure.

    Args:
        crystal: The crystal structure to analyze.
        tol: The tolerance for symmetry detection.

    Returns:
        The refined crystal structure and its symmetry information.
    """
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except Exception as e:
        log.warning(f"Initial symmetry detection failed due to: {e}. Using lower tolerance...")
        c.from_seed(crystal, tol=0.0001)
    space_group = c.group.number
    species = []
    anchors = []
    matrices = []
    coords = []
    for site in c.atom_sites:
        specie = site.specie
        anchor = len(matrices)
        coord = site.position
        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)
    anchors = np.array(anchors)
    matrices = np.array(matrices)
    coords = np.array(coords) % 1.0
    sym_info = {"anchors": anchors, "wyckoff_ops": matrices, "spacegroup": space_group}
    crystal = Structure(
        lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    return crystal, sym_info


@typecheck
def build_crystal_graph(crystal: Structure, graph_method: str = "crystalnn") -> Dict[str, Any]:
    """Build a crystal graph from a crystal structure.

    Args:
        crystal: The crystal structure to convert.
        graph_method: The method to use for graph construction.

    Returns:
        The constructed crystal graph.
    """
    if graph_method == "crystalnn":
        try:
            crystal_graph = StructureGraph.from_local_env_strategy(crystal, CrystalNN)
        except Exception as e:
            log.warning(f"CrystalNN graph construction failed due to: {e}")
            crystalNN_tmp = local_env.CrystalNN(
                distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=10
            )
            crystal_graph = StructureGraph.from_local_env_strategy(crystal, crystalNN_tmp)
    elif graph_method == "none":
        pass
    else:
        raise NotImplementedError

    cell = crystal.lattice.matrix
    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers

    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]
    assert np.allclose(crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return {
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "cell": cell,
        "lattices": lattice_parameters,
        "lengths": lengths,
        "angles": angles,
        "edge_indices": edge_indices,
        "to_jimages": to_jimages,
        "num_atoms": num_atoms,
    }


@typecheck
def abs_cap(val: FLOAT_TYPE, max_abs_val: float = 1) -> FLOAT_TYPE:
    """Return the value with its absolute value capped at `max_abs_val`. Particularly useful in
    passing values to trigonometric functions where numerical errors may result in an argument > 1
    being passed in.

    Reference:
        https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15

    Args:
        val: Input value.
        max_abs_val: The maximum absolute value for val. Defaults to 1.

    Returns:
        `val` if `abs(val) < 1` else sign of `val * max_abs_val`.
    """
    return max(min(val, max_abs_val), -max_abs_val)


@typecheck
def lattice_params_to_matrix(
    a: FLOAT_TYPE,
    b: FLOAT_TYPE,
    c: FLOAT_TYPE,
    alpha: FLOAT_TYPE,
    beta: FLOAT_TYPE,
    gamma: FLOAT_TYPE,
) -> np.ndarray:
    """Convert lattice from abc, angles to matrix.

    Reference:
        https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311

    Args:
        a: Lattice parameter a.
        b: Lattice parameter b.
        c: Lattice parameter c.
        alpha: Lattice angle alpha (in degrees).
        beta: Lattice angle beta (in degrees).
        gamma: Lattice angle gamma (in degrees).

    Returns:
        Lattice matrix as a 3x3 numpy array.
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # NOTE: Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


@typecheck
def lattice_params_to_matrix_torch(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Run a batched PyTorch version of a lattice matrix from params conversion.

    Args:
        lengths: Tensor of shape (N, 3), units in A.
        angles: Tensor of shape (N, 3), units in degrees.

    Returns:
        Tensor of shape (N, 3, 3) representing the lattice matrix.
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # NOTE: Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


@typecheck
def compute_volume(batch_lattice: torch.Tensor) -> torch.Tensor:
    """Compute volume from batched lattice matrix.

    Args:
        batch_lattice: Tensor of shape (N, 3, 3).

    Returns:
        Tensor of shape (N,) representing the volume.
    """
    vector_a, vector_b, vector_c = torch.unbind(batch_lattice, dim=1)
    return torch.abs(torch.einsum("bi,bi->b", vector_a, torch.cross(vector_b, vector_c, dim=1)))


@typecheck
def lengths_angles_to_volume(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Convert lattice lengths and angles to volume.

    Args:
        lengths: Tensor of shape (N, 3), with units in A.
        angles: Tensor of shape (N, 3), with units in degrees.

    Returns:
        Tensor of shape (N,) representing the volume.
    """
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return compute_volume(lattice)


@typecheck
def lattice_matrix_to_params(
    matrix: np.ndarray,
) -> Tuple[FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE]:
    """Convert a lattice matrix to its parameters.

    Args:
        matrix: Array of shape (3, 3) representing the lattice matrix.

    Returns:
        A tuple (a, b, c, alpha, beta, gamma) representing the lattice parameters.
    """
    lengths = np.sqrt(np.sum(matrix**2, axis=1)).tolist()

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) / (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma


@typecheck
def lattices_to_params_shape(lattices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a batch of lattice matrices to their parameters.

    Args:
        lattices: Tensor of shape (N, 3, 3) representing the lattice matrices.

    Returns:
        A tuple (lengths, angles) where lengths is a tensor of shape (N, 3) and angles is a tensor of shape (N, 3).
    """
    lengths = torch.sqrt(torch.sum(lattices**2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = torch.clamp(
            torch.sum(lattices[..., j, :] * lattices[..., k, :], dim=-1)
            / (lengths[..., j] * lengths[..., k]),
            -1.0,
            1.0,
        )
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles


@typecheck
def frac_to_cart_coords(
    frac_coords: torch.Tensor,
    num_atoms: int,
    lengths: torch.Tensor | None = None,
    angles: torch.Tensor | None = None,
    lattices: torch.Tensor | None = None,
    regularized: bool = True,
) -> torch.Tensor:
    """Convert fractional coordinates to Cartesian coordinates.

    Args:
        frac_coords: Tensor of shape (N, 3) representing the fractional coordinates.
        num_atoms: Number of atoms in the system.
        lengths: Optional tensor of shape (N, 3) representing the lattice lengths.
        angles: Optional tensor of shape (N, 3) representing the lattice angles.
        lattices: Optional tensor of shape (N, 3, 3) directly representing the lattice matrices.
        regularized: Whether to regularize the fractional coordinates.

    Returns:
        Tensor of shape (N, 3) representing the Cartesian coordinates.
    """
    if regularized:
        frac_coords = frac_coords % 1.0
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
    pos = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)  # Cartesian coords

    return pos


@typecheck
def cart_to_frac_coords(
    cart_coords: torch.Tensor,
    num_atoms: int,
    lengths: torch.Tensor | None = None,
    angles: torch.Tensor | None = None,
    lattices: torch.Tensor | None = None,
    regularized: bool = True,
) -> torch.Tensor:
    """Convert Cartesian coordinates to fractional coordinates.

    Args:
        cart_coords: Tensor of shape (N, 3) representing the Cartesian coordinates.
        num_atoms: Number of atoms in the system.
        lengths: Optional tensor of shape (1, 3) representing the lattice lengths.
        angles: Optional tensor of shape (1, 3) representing the lattice angles.
        lattices: Optional tensor of shape (1, 3, 3) directly representing the lattice matrices.
        regularized: Whether to regularize the fractional coordinates.

    Returns:
        Tensor of shape (N, 3) representing the fractional coordinates.
    """
    if lattices is None:
        if lengths is None or angles is None:
            raise ValueError("Either 'lattices' or both 'lengths' and 'angles' must be provided.")
        lattices = lattice_params_to_matrix_torch(lengths, angles)

    # Use `pinv` in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattices)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    if regularized:
        frac_coords = frac_coords % 1.0
    return frac_coords


@typecheck
def get_pbc_distances(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    to_jimages: torch.Tensor,
    num_atoms: int,
    num_bonds: int,
    coord_is_cart: bool = False,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
    lattices: torch.Tensor | None = None,
):
    """Get periodic boundary conditions (PBC) distances for a set of coordinates.

    Args:
        coords: Tensor of shape (N, 3) representing the coordinates.
        edge_index: Tensor of shape (2, M) representing the edges.
        lengths: Tensor of shape (N, 3) representing the lattice lengths.
        angles: Tensor of shape (N, 3) representing the lattice angles.
        to_jimages: Tensor of shape (M, 3) representing the image offsets.
        num_atoms: Number of atoms in the system.
        num_bonds: Number of bonds in the system.
        coord_is_cart: Whether the input coordinates are in Cartesian form.
        return_offsets: Whether to return the offset vectors.
        return_distance_vec: Whether to return the distance vectors.
        lattices: Optional tensor of shape (N, 3, 3) representing the lattice matrices.

    Returns:
        A dictionary containing the PBC distances and optionally the offsets and distance vectors.
    """
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
        pos = torch.einsum("bi,bij->bj", coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # Correct for PBC
    lattice_edges = torch.repeat_interleave(lattices, num_bonds, dim=0)
    offsets = torch.einsum("bi,bij->bj", to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # Compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out


@typecheck
def radius_graph_pbc_wrapper(
    data: Data, radius: float, max_num_neighbors_threshold: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrapper for the radius_graph_pbc function.

    Args:
        data: The input data containing atomic coordinates and other information.
        radius: The cutoff radius for the graph.
        max_num_neighbors_threshold: The maximum number of neighbors to consider.
        device: The device to perform computations on.

    Returns:
        A tuple of three tensors:
            - edge_index: Tensor of shape (2, num_edges) containing the indices of the edges.
            - unit_cell: Tensor of shape (batch_size, 3, 3) containing the unit cell vectors.
            - num_neighbors_image: Tensor of shape (batch_size,) containing the number of neighbors per image.
    """
    cart_coords = frac_to_cart_coords(
        data.frac_coords, lengths=data.lengths, angles=data.angles, num_atoms=data.num_atoms
    )
    return radius_graph_pbc(
        cart_coords,
        data.lengths,
        data.angles,
        data.num_atoms,
        radius,
        max_num_neighbors_threshold,
        device,
    )


@typecheck
def repeat_blocks(
    sizes: torch.Tensor,
    repeats: torch.Tensor,
    continuous_indexing: bool = True,
    start_idx: int = 0,
    block_inc: int = 0,
    repeat_inc: int = 0,
):
    """Repeat blocks of indices.

    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements.

    Args:
        sizes: A tensor of shape (N,) representing the sizes of each block.
        repeats: A tensor of shape (N,) representing the number of repetitions for each block.
        continuous_indexing: Whether to keep increasing the index after each block.
        start_idx: Starting index.
        block_inc: Number to increment by after each block, either global or per block. Shape: len(sizes) - 1.
        repeat_inc: Number to increment by after each repetition, either global or per block.

    Examples:
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False

        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]

        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True

        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]

        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4

        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]

        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5

        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]

        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1

        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]

        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True

        Return: [0 1 2 0 1 2  3 4 3 4 3 4]

        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True

        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # NOTE: Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), repeats)

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(block_inc[: r1[-1]], indptr, reduce="sum")
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res


# Utilities from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/common/utils.py#L566:


@typecheck
def radius_graph_pbc(
    pos: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    natoms: int,
    radius: float,
    max_num_neighbors_threshold: int,
    device: torch.device,
    lattices: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the radius graph with periodic boundary conditions.

    NOTE: It seems like radius param is not even being used, and only max threshold is used.
    Also, there are two such methods...

    Args:
        pos: Atom positions tensor of shape (batch_size, natoms, 3).
        lengths: Cell lengths tensor of shape (batch_size, 3).
        angles: Cell angles tensor of shape (batch_size, 3).
        natoms: Number of atoms per image.
        radius: Cutoff radius for neighbor search.
        max_num_neighbors_threshold: Maximum number of neighbors to consider.
        device: Device to perform computations on.
        lattices: Optional lattice tensor of shape (batch_size, 3, 3).

    Returns:
        A tuple of three tensors:
            - edge_index: Tensor of shape (2, num_edges) containing the indices of the edges.
            - unit_cell: Tensor of shape (batch_size, 3, 3) containing the unit cell vectors.
            - num_neighbors_image: Tensor of shape (batch_size,) containing the number of neighbors per image.
    """

    batch_size = len(natoms)
    if lattices is None:
        cell = lattice_params_to_matrix_torch(lengths, angles)
    else:
        cell = lattices

    # Position of the atoms
    atom_pos = pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # Index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this approach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    min_dist_a1 = (1 / inv_min_dist_a1).reshape(-1, 1)

    cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    min_dist_a2 = (1 / inv_min_dist_a2).reshape(-1, 1)

    cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
    inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
    min_dist_a3 = (1 / inv_min_dist_a3).reshape(-1, 1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = torch.ones(3, dtype=torch.long, device=device)
    min_dist = torch.cat([min_dist_a1, min_dist_a2, min_dist_a3], dim=-1)  # N_graphs * 3

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float32) for rep in max_rep
    ]

    unit_cell = torch.cat([_.reshape(-1, 1) for _ in torch.meshgrid(cells_per_dim)], dim=-1)

    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, num_atoms_per_image_sqr, dim=0)

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    radius_real = min_dist.min(dim=-1)[0] + 0.01  # .clamp(max=radius)
    radius_real = torch.repeat_interleave(radius_real, num_atoms_per_image_sqr * num_cells)
    mask_within_radius = torch.le(atom_distance_sqr, radius_real * radius_real)

    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3))
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    if max_num_neighbors_threshold is not None:
        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=natoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not torch.all(mask_num_neighbors):
            # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)
            unit_cell = torch.masked_select(
                unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
            )
            unit_cell = unit_cell.view(-1, 3)

    else:
        ones = index1.new_ones(1).expand_as(index1)
        num_neighbors = segment_coo(ones, index1, dim_size=natoms.sum())

        # Get number of (thresholded) neighbors per image
        image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
        image_indptr[1:] = torch.cumsum(natoms, dim=0)
        num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


@typecheck
def get_max_neighbors_mask(
    natoms: torch.Tensor,
    index: torch.Tensor,
    atom_distance: torch.Tensor,
    max_num_neighbors_threshold: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.

    NOTE: Assumes that `index` is sorted.

    Args:
        natoms: A tensor of shape (batch_size,) containing the number of atoms per image.
        index: A tensor of shape (num_edges,) containing the indices of the edges.
        atom_distance: A tensor of shape (num_edges,) containing the distances of the edges.
        max_num_neighbors_threshold: The maximum number of neighbors allowed per atom.

    Returns:
        A tuple containing:
            - mask_num_neighbors: A tensor of shape (num_edges,) containing a boolean mask indicating which edges are valid.
            - num_neighbors_image: A tensor of shape (batch_size,) containing the number of neighbors per image.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # NOTE: segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if max_num_neighbors <= max_num_neighbors_threshold or max_num_neighbors_threshold <= 0:
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    # NOTE: index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(index_neighbor_offset, num_neighbors)
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_real_cutoff = (
        distance_sort[:, max_num_neighbors_threshold].reshape(-1, 1).expand(-1, max_num_neighbors)
        + 0.01
    )

    mask_distance = distance_sort < distance_real_cutoff

    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(-1, max_num_neighbors)

    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite & mask_distance)

    num_neighbor_per_node = (mask_finite & mask_distance).sum(dim=-1)
    num_neighbors_image = segment_csr(num_neighbor_per_node, image_indptr)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


@typecheck
def radius_graph_pbc_(
    cart_coords: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    num_atoms: torch.Tensor,
    radius: float,
    max_num_neighbors_threshold: int,
    device: torch.device,
    topk_per_pair: torch.Tensor | None = None,
) -> (
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Compute graph edges under PBC.

    NOTE: The top-k should take into account the self-self edge for (i, i).

    Args:
        cart_coords: Atomic Cartesian coordinates of shape (num_atoms, 3).
        lengths: Lattice parameters (a, b, c) of shape (3,).
        angles: Lattice angles (alpha, beta, gamma) of shape (3,).
        num_atoms: Number of atoms in each image of shape (batch_size,).
        radius: Cutoff radius for neighbor search.
        max_num_neighbors_threshold: Maximum number of neighbors to consider.
        device: The device to perform computations on.
        topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Returns:
        A tuple of three or four tensors:
            - edge_index: Tensor of shape (2, num_edges) containing the indices of the edges.
            - unit_cell: Tensor of shape (batch_size, 3, 3) containing the unit cell vectors.
            - num_neighbors_image: Tensor of shape (batch_size,) containing the number of neighbors per image.
            - (optional) topk_mask: Tensor of shape (num_edges,) containing a mask for the top-k edges.
    """
    batch_size = len(num_atoms)

    # Position of the atoms
    atom_pos = cart_coords

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # Index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this approach could run into numerical precision issues
    index1 = (atom_count_sqr // num_atoms_per_image_expand).long() + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand).long() + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, num_atoms_per_image_sqr, dim=0)

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)

    if topk_per_pair is not None:
        assert topk_per_pair.size(0) == num_atom_pairs
        atom_distance_sqr_sort_index = torch.argsort(atom_distance_sqr, dim=1)
        assert atom_distance_sqr_sort_index.size() == (num_atom_pairs, num_cells)
        atom_distance_sqr_sort_index = (
            atom_distance_sqr_sort_index
            + torch.arange(num_atom_pairs, device=device)[:, None] * num_cells
        ).view(-1)
        topk_mask = torch.arange(num_cells, device=device)[None, :] < topk_per_pair[:, None]
        topk_mask = topk_mask.view(-1)
        topk_indices = atom_distance_sqr_sort_index.masked_select(topk_mask)

        topk_mask = torch.zeros(num_atom_pairs * num_cells, device=device)
        topk_mask.scatter_(0, topk_indices, 1.0)
        topk_mask = topk_mask.bool()

    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3))
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask)

    num_neighbors = torch.zeros(len(cart_coords), device=device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
    num_neighbors = num_neighbors.long()
    max_num_neighbors = torch.max(num_neighbors).long()

    # Compute neighbors per image
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[_max_neighbors > max_num_neighbors_threshold] = max_num_neighbors_threshold
    _num_neighbors = torch.zeros(len(cart_coords) + 1, device=device).long()
    _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long()
    _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
    _natoms[1:] = torch.cumsum(num_atoms, dim=0)
    num_neighbors_image = _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]

    # If max_num_neighbors is below the threshold, return early
    if max_num_neighbors <= max_num_neighbors_threshold or max_num_neighbors_threshold <= 0:
        if topk_per_pair is None:
            return torch.stack((index2, index1)), unit_cell, num_neighbors_image
        else:
            return torch.stack((index2, index1)), unit_cell, num_neighbors_image, topk_mask

    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
    distance_sort = torch.zeros(len(cart_coords) * max_num_neighbors, device=device).fill_(
        radius * radius + 1.0
    )

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(index_neighbor_offset, num_neighbors)
    index_sort_map = (
        index1 * max_num_neighbors
        + torch.arange(len(index1), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(cart_coords), max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
    mask_within_radius = torch.le(distance_sort, radius * radius)
    index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
    mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = torch.masked_select(index1, mask_num_neighbors)
    index2 = torch.masked_select(index2, mask_num_neighbors)
    unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)

    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))

    if topk_per_pair is None:
        return edge_index, unit_cell, num_neighbors_image
    else:
        return edge_index, unit_cell, num_neighbors_image, topk_mask


@typecheck
def min_distance_sqr_pbc(
    cart_coords1: torch.Tensor,
    cart_coords2: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    num_atoms: torch.Tensor,
    device: torch.device,
    return_vector: bool = False,
    return_to_jimages: bool = False,
) -> torch.Tensor | List[torch.Tensor]:
    """Compute the PBC distance between atoms in `cart_coords1` and `cart_coords2`.

    NOTE: This function assumes that `cart_coords1` and `cart_coords2` have the same number of atoms
    in each data point.

    Args:
        cart_coords1: (N_atoms, 3) tensor of Cartesian coordinates for the first set of atoms.
        cart_coords2: (N_atoms, 3) tensor of Cartesian coordinates for the second set of atoms.
        lengths: (3,) tensor of lattice parameters (a, b, c).
        angles: (3,) tensor of lattice angles (alpha, beta, gamma).
        num_atoms: (N_atoms,) tensor of atom counts for each data point.
        device: the device to perform computations on (CPU or GPU).
        return_vector: whether to return the distance vector.
        return_to_jimages: whether to return the image indices.

    Returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from `cart_coords1` to `cart_coords2`, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of `cart_coord2` relative to `cart_coord1` in PBC
    """
    batch_size = len(num_atoms)

    # Get the positions for each atom
    pos1 = cart_coords1
    pos2 = cart_coords2

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, num_atoms, dim=0)

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # Shape: (num_atom_squared_sum, 3, 27)
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = torch.sum(atom_distance_vector**2, dim=1)

    min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)

    return_list = [min_atom_distance_sqr]

    if return_vector:
        min_indices = min_indices[:, None, None].repeat([1, 3, 1])

        min_atom_distance_vector = torch.gather(atom_distance_vector, 2, min_indices).squeeze(-1)

        return_list.append(min_atom_distance_vector)

    if return_to_jimages:
        to_jimages = unit_cell.T[min_indices].long()
        return_list.append(to_jimages)

    return return_list[0] if len(return_list) == 1 else return_list


@typecheck
def process_one(
    row: pd.Series,
    niggli: bool,
    primitive: bool,
    graph_method: str,
    prop_list: List[str],
    use_space_group: bool = False,
    tol: float = 0.01,
) -> Dict[str, Any]:
    """Process a single row (i.e., series) of the DataFrame.

    Args:
        row: A Pandas Series representing a single row of the DataFrame.
        niggli: Whether to use Niggli reduction.
        primitive: Whether to use primitive cell.
        graph_method: The method to use for graph construction.
        prop_list: The list of properties to extract from the row.
        use_space_group: Whether to use space group information.
        tol: Tolerance for symmetry detection.

    Returns:
        A dictionary containing the processed information for the crystal.
    """
    crystal_str = row["cif"]
    crystal = build_crystal(crystal_str, niggli=niggli, primitive=primitive)
    result_dict = {}
    if use_space_group:
        _, sym_info = get_symmetry_info(crystal, tol=tol)  # Do not modify crystal
        result_dict.update(sym_info)
    else:
        result_dict["spacegroup"] = 1
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: row[k] for k in prop_list if k in row.keys()}
    result_dict.update(
        {"mp_id": row["material_id"], "cif": crystal_str, "graph_arrays": graph_arrays}
    )
    result_dict.update(properties)
    return result_dict


@typecheck
def process_one_parquet(
    row: pd.Series,
    prop_list: List[str],
) -> Dict[str, Any]:
    """Process a single row (i.e., series) of the DataFrame.

    Args:
        row: A Pandas Series representing a single row of the DataFrame.
        prop_list: The list of properties to extract from the row.

    Returns:
        A dictionary containing the processed information for the crystal.
    """
    result_dict = {"graph_arrays": {}, "spacegroup": 1, "mp_id": row["mp_id"]}

    # --- Direct use of the original lattice matrix ---
    original_lattice_matrix = np.stack(row["cell"])
    original_cart_coords = torch.from_numpy(np.stack(row["positions"])).float()
    lattice_torch = torch.from_numpy(original_lattice_matrix).float().unsqueeze(0)  # Add batch dim

    result_dict["graph_arrays"]["num_atoms"] = int(row["num_atoms"])
    result_dict["graph_arrays"]["atom_types"] = row["numbers"]
    result_dict["graph_arrays"]["cell"] = original_lattice_matrix

    # Calculate and store the lattice params for use elsewhere
    a, b, c, alpha, beta, gamma = lattice_matrix_to_params(original_lattice_matrix)
    result_dict["graph_arrays"]["lengths"] = np.array([a, b, c])
    result_dict["graph_arrays"]["angles"] = np.array([alpha, beta, gamma])
    result_dict["graph_arrays"]["lattices"] = np.concatenate(
        [result_dict["graph_arrays"]["lengths"], result_dict["graph_arrays"]["angles"]]
    )

    # --- Perform conversions using the ORIGINAL lattice matrix ---
    result_dict["graph_arrays"]["frac_coords"] = cart_to_frac_coords(
        cart_coords=original_cart_coords,
        lattices=lattice_torch,
        num_atoms=result_dict["graph_arrays"]["num_atoms"],
    ).numpy()

    result_dict.update({k: row[k] for k in prop_list if k in row.keys()})

    return result_dict


@typecheck
def preprocess(
    input_file: str,
    num_workers: int,
    niggli: bool,
    primitive: bool,
    graph_method: str,
    prop_list: List[str],
    use_space_group: bool = False,
    tol: float = 0.01,
) -> List[Dict[str, Any]]:
    """Preprocess the input data.

    Args:
        input_file: The path to the input CSV file.
        num_workers: The number of worker processes to use.
        niggli: Whether to use Niggli reduction.
        primitive: Whether to use primitive cell.
        graph_method: The method to use for graph construction.
        prop_list: The list of properties to extract from the DataFrame.
        use_space_group: Whether to use space group information.
        tol: Tolerance for symmetry detection.

    Returns:
        A list of dictionaries containing the processed information for each crystal.
    """
    df = pd.read_csv(input_file)

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [niggli] * len(df),
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list] * len(df),
        [use_space_group] * len(df),
        [tol] * len(df),
        num_cpus=num_workers,
    )

    mpid_to_results = {result["mp_id"]: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]["material_id"]] for idx in range(len(df))]

    return ordered_results


@typecheck
def preprocess_parquet(
    df: pd.DataFrame,
    num_workers: int,
    prop_list: List[str],
) -> List[Dict[str, Any]]:
    """Preprocess the input data.

    Args:
        df: The path to the input Parquet DataFrame
        num_workers: The number of worker processes to use.
        prop_list: The list of properties to extract from the DataFrame.

    Returns:
        A list of dictionaries containing the processed information for each crystal.
    """
    unordered_results = p_umap(
        process_one_parquet,
        [df.iloc[idx] for idx in range(len(df))],
        [prop_list] * len(df),
        num_cpus=num_workers,
    )

    mpid_to_results = {result["mp_id"]: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]["mp_id"]] for idx in range(len(df))]

    return ordered_results
