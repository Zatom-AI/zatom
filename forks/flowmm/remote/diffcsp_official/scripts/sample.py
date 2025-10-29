import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from p_tqdm import p_map
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from diffcsp.eval_utils import get_crystals_list, lattices_to_params_shape, load_model
from diffcsp.script_utils import SampleDataset


def diffusion(loader, model, step_lr):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr=step_lr)
        frac_coords.append(outputs["frac_coords"].detach().cpu())
        num_atoms.append(outputs["num_atoms"].detach().cpu())
        atom_types.append(outputs["atom_types"].detach().cpu())
        lattices.append(outputs["lattices"].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (frac_coords, atom_types, lattices, lengths, angles, num_atoms)


def get_pymatgen(crystal_array):
    frac_coords = crystal_array["frac_coords"]
    atom_types = crystal_array["atom_types"]
    lengths = crystal_array["lengths"]
    angles = crystal_array["angles"]
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
        return structure
    except:
        return None


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(model_path, load_data=False)

    if torch.cuda.is_available():
        model.to("cuda")

    tar_dir = os.path.join(args.save_path, args.formula)
    os.makedirs(tar_dir, exist_ok=True)

    print("Evaluate the diffusion model.")

    test_set = SampleDataset(args.formula, args.num_evals)
    test_loader = DataLoader(test_set, batch_size=min(args.batch_size, args.num_evals))

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(
        test_loader, model, args.step_lr
    )

    crystal_list = get_crystals_list(
        frac_coords, atom_types, lengths, angles, num_atoms
    )

    strcuture_list = p_map(get_pymatgen, crystal_list)

    for i, structure in enumerate(strcuture_list):
        tar_file = os.path.join(tar_dir, f"{args.formula}_{i+1}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i+1} Error Structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--formula", required=True)
    parser.add_argument("--num_evals", default=1, type=int)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--step_lr", default=1e-5, type=float)

    args = parser.parse_args()

    main(args)
