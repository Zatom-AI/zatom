import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import chemparse
import torch
from p_tqdm import p_map
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group
from torch.optim import Adam
from torch_geometric.data import Batch, DataLoader
from tqdm import tqdm

from diffcsp.eval_utils import (
    get_crystals_list,
    lattices_to_params_shape,
    load_model,
    recommand_step_lr,
)
from diffcsp.script_utils import GenDataset


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


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(
        model_path,
        load_data=False,
    )
    if torch.cuda.is_available():
        model.to("cuda")

    print("Evaluate the diffusion model.")

    test_set = GenDataset(args.dataset, args.batch_size * args.num_batches_to_samples)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    step_lr = (
        args.step_lr if args.step_lr >= 0 else recommand_step_lr["gen"][args.dataset]
    )

    print(step_lr)

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(
        test_loader, model, step_lr
    )

    if args.label == "":
        gen_out_name = "eval_gen.pt"
    else:
        gen_out_name = f"eval_gen_{args.label}.pt"

    torch.save(
        {
            "eval_setting": args,
            "frac_coords": frac_coords,
            "num_atoms": num_atoms,
            "atom_types": atom_types,
            "lengths": lengths,
            "angles": angles,
        },
        model_path / gen_out_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--step_lr", default=-1, type=float)
    parser.add_argument("--num_batches_to_samples", default=20, type=int)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    main(args)
