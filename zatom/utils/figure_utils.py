from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pymatgen.core import Structure
from rdkit import Chem
from tensordict import TensorDict
from torch import nn

from zatom.utils.typing_utils import typecheck

ELEMENT_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    "P": 15,
    "B": 5,
    "Si": 14,
}


@typecheck
def load_cif_to_tensors(
    cif_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a CIF file and extract atom types, fractional coordinates, and lattice parameters.

    Args:
        cif_path: Path to the CIF file

    Returns:
        atom_types: Tensor of atomic numbers (N,)
        frac_coords: Tensor of fractional coordinates (N, 3)
        lengths: Tensor of lattice lengths in Angstroms (3,)
        angles: Tensor of lattice angles in degrees (3,)
    """
    structure = Structure.from_file(cif_path)

    # Get atomic numbers
    atom_types = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)

    # Get fractional coordinates
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)

    # Get lattice parameters
    lattice = structure.lattice
    lengths = torch.tensor([lattice.a, lattice.b, lattice.c], dtype=torch.float32)
    angles = torch.tensor([lattice.alpha, lattice.beta, lattice.gamma], dtype=torch.float32)

    # Convert angles to radians
    angles_radians = angles * (np.pi / 180.0)

    # Normalize the lengths of lattice vectors, which makes
    # lengths for materials of different sizes at same scale
    num_atoms = len(atom_types)
    lengths_scaled = lengths / float(num_atoms) ** (1 / 3)

    return atom_types, frac_coords, lengths_scaled, angles_radians


@typecheck
def load_pdb_to_tensors(pdb_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a PDB file and extract atom types and 3D coordinates. Skips RDKit sanitization to
    handle chemically invalid generated molecules.

    Args:
        pdb_path: Path to the PDB file

    Returns:
        atom_types: Tensor of atomic numbers (N,)
        pos: Tensor of 3D coordinates (N, 3)
    """
    from rdkit import RDLogger

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")

    # Try loading without sanitization first (for invalid molecules)
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        if mol is None:
            mol = parse_pdb_simple(pdb_path)
            if mol is None:
                raise ValueError(f"Could not load PDB file: {pdb_path}")

    # Get atomic numbers
    atom_types = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

    # Get 3D coordinates
    conf = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    pos = torch.tensor(positions, dtype=torch.float32)

    # Re-enable logging
    RDLogger.EnableLog("rdApp.*")

    return atom_types, pos


@typecheck
def parse_pdb_simple(pdb_path: str) -> Chem.Mol | None:
    """Simple PDB parser that extracts atom positions without chemistry validation. Fallback for
    when RDKit fails.

    Args:
        pdb_path: Path to the PDB file

    Returns:
        RDKit Mol object with atoms and coordinates, or None if parsing fails.
    """
    atoms = []
    coords = []

    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # PDB format: columns 13-16 atom name, 31-38 x, 39-46 y, 47-54 z, 77-78 element
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    # Try to get element from columns 77-78, fallback to atom name
                    if len(line) >= 78:
                        element = line[76:78].strip()
                    else:
                        element = line[12:16].strip()[0]  # First char of atom name

                    # Clean element name
                    element = element.capitalize()
                    if element and element[0].isdigit():
                        element = element[1:] if len(element) > 1 else "C"

                    if element in ELEMENT_TO_Z:
                        atoms.append(ELEMENT_TO_Z[element])
                        coords.append([x, y, z])

        if not atoms:
            return None

        # Create an editable RDKit mol
        mol = Chem.RWMol()
        conf = Chem.Conformer(len(atoms))

        for i, (atomic_num, (x, y, z)) in enumerate(zip(atoms, coords)):
            atom = Chem.Atom(atomic_num)
            mol.AddAtom(atom)
            conf.SetAtomPosition(i, (x, y, z))

        mol.AddConformer(conf)
        return mol.GetMol()

    except Exception as e:
        return None


@typecheck
def extract_embeddings(
    model: nn.Module,
    atom_types: torch.Tensor,  # (B, N) - atomic numbers
    pos: torch.Tensor,  # (B, N, 3) - 3D coordinates
    is_periodic: bool = False,  # Whether the sample is periodic (crystal) or not (molecule)
    frac_coords: torch.Tensor = None,  # (B, N, 3) - fractional coordinates for crystals
    lengths_scaled: torch.Tensor = None,  # (B, 1, 3) - scaled lattice lengths
    angles_radians: torch.Tensor = None,  # (B, 1, 3) - lattice angles in radians
    dataset_idx: int = 1,  # 0=periodic, 1=non-periodic
    spacegroup: int = 0,  # spacegroup index (0 for molecules)
    return_pooled: bool = True,  # Whether to return mean-pooled embedding
    layer: int | None = None,  # Which layer to extract from (None = final layer)
    max_num_nodes: int | None = 29,  # Maximum number of nodes (for padding)
    timestep: float = 1.0,  # Time step for conditioning (1.0 for generated samples)
) -> torch.Tensor:
    """Extract embeddings from the TFT model for clean molecules (e.g., t=1.0).

    Args:
        model: The TFT model
        atom_types: Tensor of atomic numbers (B, N)
        pos: Tensor of 3D coordinates (B, N, 3)
        is_periodic: Whether this is a periodic crystal
        frac_coords: Fractional coordinates for crystals
        lengths_scaled: Scaled lattice lengths for crystals
        angles_radians: Lattice angles for crystals
        dataset_idx: Dataset index (1 for molecules like QM9)
        spacegroup: Spacegroup index
        return_pooled: If True, return mean-pooled embedding, else per-atom embeddings
        layer: Which transformer layer to extract from (0-indexed).
            None means final layer output. For a 16-layer model, valid values are 0-15.
        max_num_nodes: Maximum number of nodes for padding (if needed). May need to change
            if the model was not pretrained on QM9 (which has max 29 atoms). Set to None
            to disable padding.
        timestep: Time step for conditioning. Use 1.0 for generated samples to get embeddings
            close to final.

    Returns:
        embeddings: Tensor of shape (B, hidden_dim) if pooled, else (B, N, hidden_dim)
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size, num_atoms = atom_types.shape
    hidden_dim = model.model.hidden_dim

    if max_num_nodes is not None:
        if num_atoms > max_num_nodes:
            raise ValueError(
                f"Number of atoms ({num_atoms}) exceeds max_num_nodes ({max_num_nodes})."
            )

        if hasattr(model, "jvp_attn") and model.jvp_attn:
            # Find the smallest power of 2 >= max(max_num_nodes, 32)
            min_num_nodes = max(max_num_nodes, 32)
            closest_power_of_2 = 1 << (min_num_nodes - 1).bit_length()
            max_num_nodes = int(closest_power_of_2)

        if hasattr(model, "context_length") and model.context_length < max_num_nodes:
            raise ValueError(
                f"Model context length ({model.context_length}) is smaller than max_num_nodes ({max_num_nodes})."
            )
    else:
        max_num_nodes = num_atoms

    # Move tensors to device
    atom_types = atom_types.to(device)
    pos = pos.to(device)

    # Set default values for crystals/molecules
    if frac_coords is None:
        frac_coords = torch.zeros_like(pos)
    else:
        frac_coords = frac_coords.to(device)

    if lengths_scaled is None:
        lengths_scaled = torch.zeros(batch_size, 1, 3, device=device)
    else:
        lengths_scaled = lengths_scaled.to(device)

    if angles_radians is None:
        angles_radians = torch.zeros(batch_size, 1, 3, device=device)
    else:
        angles_radians = angles_radians.to(device)

    # Create padding mask (True for real atoms, False for padding)
    token_mask = torch.zeros(
        batch_size,
        max_num_nodes,
        dtype=torch.bool,
        device=device,
    )
    for i in range(num_atoms):
        token_mask[:, i] = True

    # Pad inputs to max_num_nodes if necessary
    if num_atoms < max_num_nodes:
        pad_size = max_num_nodes - num_atoms
        atom_types = F.pad(atom_types, (0, pad_size), value=0)  # Pad with atomic number 0
        pos = F.pad(pos, (0, 0, 0, pad_size), value=0.0)  # Pad positions with zeros
        frac_coords = F.pad(
            frac_coords, (0, 0, 0, pad_size), value=0.0
        )  # Pad fractional coords with zeros
        # lengths_scaled and angles_radians are already (B, 1, 3) so no need to pad them

    # Token periodicity
    token_is_periodic = torch.full(
        (batch_size, max_num_nodes), is_periodic, dtype=torch.bool, device=device
    )

    # Conditioning features
    use_cfg = model.class_dropout_prob > 0
    dataset_idx_tensor = torch.full(
        # NOTE 0 -> null class within model, while 0 -> MP20 elsewhere, so increment by 1 (for classifier-free guidance or CFG)
        (batch_size,),
        dataset_idx + int(use_cfg),
        dtype=torch.int64,
        device=device,
    )
    spacegroup_tensor = torch.full((batch_size,), spacegroup, dtype=torch.long, device=device)
    # NOTE: For now, we do not condition on spacegroups
    spacegroup_tensor = torch.zeros_like(spacegroup_tensor)

    # Set timestep for generated sample embeddings
    t = torch.full((batch_size,), timestep, device=device)

    # Create batch to sample noise
    batch = TensorDict(
        {
            "atom_types": F.one_hot(
                atom_types, num_classes=model.model.atom_type_embed.num_embeddings
            ).float(),
            "pos": pos,
            "frac_coords": frac_coords,
            "lengths_scaled": lengths_scaled,
            "angles_radians": angles_radians,
            "dataset_idx": dataset_idx_tensor,
            "spacegroup": spacegroup_tensor,
            "padding_mask": ~token_mask,
            "token_is_periodic": token_is_periodic,
        },
        device=device,
    )
    path = model._create_path(batch, t=t)

    with torch.no_grad():
        # Call the inner transformer module to get embeddings
        # We need to hook into the transformer to extract h_out
        transformer_module = model.model

        sample_is_periodic = token_is_periodic.any(-1, keepdim=True).unsqueeze(-1)
        token_is_periodic_expanded = token_is_periodic.unsqueeze(-1)

        # Mask coordinates based on periodicity
        if transformer_module.mask_material_coords:
            pos_input = path.x_t["pos"] * ~token_is_periodic_expanded
        else:
            pos_input = path.x_t["pos"]
        atom_types_input = path.x_t["atom_types"].argmax(-1)
        frac_coords_input = path.x_t["frac_coords"] * token_is_periodic_expanded
        lengths_scaled_input = path.x_t["lengths_scaled"] * sample_is_periodic
        angles_radians_input = path.x_t["angles_radians"] * sample_is_periodic

        real_mask = token_mask.int()

        # Compute embeddings
        embed_atom_types = transformer_module.atom_type_embed(atom_types_input)
        embed_pos = transformer_module.pos_embed(pos_input)
        embed_frac_coords = transformer_module.frac_coords_embed(frac_coords_input)
        embed_lengths_scaled = transformer_module.lengths_scaled_embed(lengths_scaled_input)
        embed_angles_radians = transformer_module.angles_radians_embed(angles_radians_input)

        if transformer_module.add_sinusoid_posenc:
            embed_posenc = transformer_module.positional_encoding(
                batch_size=batch_size, seq_len=max_num_nodes
            )
        else:
            embed_posenc = torch.zeros(batch_size, max_num_nodes, hidden_dim, device=device)

        # Time encoding
        modals_t = torch.stack([t, t, t, t, t], dim=-1)
        embed_time = (
            transformer_module.time_encoding(modals_t.reshape(-1))
            .reshape(batch_size, modals_t.shape[1], -1)
            .mean(-2)
        )

        is_training = False  # Set to False since we're extracting embeddings, not training
        embed_dataset = transformer_module.dataset_embedder(dataset_idx_tensor, is_training)
        embed_spacegroup = transformer_module.spacegroup_embedder(spacegroup_tensor, is_training)
        embed_conditions = (embed_time + embed_dataset + embed_spacegroup).unsqueeze(-2)

        # Combine all embeddings based on concat_combine_input mode
        if transformer_module.concat_combine_input:
            # Repeat embeddings that have shape (B, 1, hidden_dim) to (B, N, hidden_dim)
            embed_lengths_scaled = embed_lengths_scaled.repeat(1, max_num_nodes, 1)
            embed_angles_radians = embed_angles_radians.repeat(1, max_num_nodes, 1)
            embed_conditions = embed_conditions.repeat(1, max_num_nodes, 1)
            h_in = torch.cat(
                [
                    embed_atom_types,
                    embed_pos,
                    embed_frac_coords,
                    embed_lengths_scaled,
                    embed_angles_radians,
                    embed_posenc,
                    embed_conditions,
                ],
                dim=-1,
            )
            h_in = transformer_module.combine_input(h_in)
        else:
            # Use addition with broadcasting
            h_in = (
                embed_atom_types
                + embed_pos
                + embed_frac_coords
                + embed_lengths_scaled
                + embed_angles_radians
                + embed_posenc
                + embed_conditions
            )
        h_in = h_in * real_mask.unsqueeze(-1)

        # Pass through transformer
        pos_ids = torch.arange(max_num_nodes, device=device).unsqueeze(0).repeat(batch_size, 1)
        attention_kwargs = {"pos_ids": pos_ids}

        h_in = transformer_module.transformer_norm(h_in)

        # If extracting from a specific layer, manually iterate through layers
        if layer is not None:
            transformer = transformer_module.transformer
            h = h_in
            for i, layer_module in enumerate(transformer.layers):
                h = layer_module(h, pos_ids=pos_ids, padding_mask=~token_mask)
                if i == layer:
                    # Apply layer norm to the intermediate representation
                    h_out = transformer.norm(h)
                    break
            else:
                raise ValueError(
                    f"Layer {layer} not found. Model has {len(transformer.layers)} layers (0-indexed)."
                )
        else:
            # Use full forward pass for final layer
            h_out, h_aux = transformer_module.transformer(
                h_in, padding_mask=~token_mask, **attention_kwargs
            )

        h_out = h_out * real_mask.unsqueeze(-1)

        # Return embeddings
        if return_pooled:
            # Mean pool over atoms (excluding padding)
            embeddings = h_out.sum(dim=1) / real_mask.sum(dim=1, keepdim=True).float()
        else:
            # Return per-atom embeddings, excluding padding
            embeddings = h_out[:, :num_atoms, :]

    return embeddings
