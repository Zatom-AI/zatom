"""Copyright (c) Meta Platforms, Inc.

and affiliates.
"""

import os
import warnings
from functools import partial
from typing import Any, Dict

import numpy as np
import torch
import wandb
from pymatgen.analysis.molecule_matcher import MoleculeMatcher
from pymatgen.core import Molecule
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from tqdm import tqdm

from zatom.utils import joblib_map

RDLogger.DisableLog("rdApp.*")

from openbabel import openbabel

openbabel.obErrorLog.StopLogging()

warnings.filterwarnings("ignore", category=UserWarning)


class MoleculeReconstructionEvaluator:
    """Evaluator for molecule reconstruction tasks. Can be used within a Lightning module,
    appending predictions and ground truths during training and computing metrics at the end of an
    epoch, or can be used as a standalone object to evaluate predictions on a dataset.

    Args:
        tolerance (float): MoleculeMatcher tolerance for whether two molecules are the same.
    """

    def __init__(self, tolerance=0.01):
        self.matcher = MoleculeMatcher(tolerance=tolerance)
        self.pred_arrays_list = []  # list of Dict[str, np.array] predictions
        self.gt_arrays_list = []  # list of Dict[str, np.array] ground truths
        self.pred_mol_list = []  # list of Molecule predictions
        self.gt_mol_list = []  # list of Molecule ground truths

    def append_pred_array(self, pred: Dict[str, np.array]):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def append_gt_array(self, gt: Dict[str, np.array]):
        """Append a ground truth to the evaluator."""
        self.gt_arrays_list.append(gt)

    def clear(self):
        """Clear the stored predictions and ground truths, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.gt_arrays_list = []
        self.pred_mol_list = []
        self.gt_mol_list = []

    def _arrays_to_molecules(self, save: bool = False, save_dir: str = "", n_jobs: int = -4):
        """Convert stored predictions and ground truths to Molecule objects for evaluation."""
        self.pred_mol_list = joblib_map(
            partial(
                array_dict_to_molecule,
                save=save,
                save_dir_name=f"{save_dir}/pred",
            ),
            self.pred_arrays_list,
            n_jobs=n_jobs,
            inner_max_num_threads=1,
            desc="    Pred to Molecule",
            total=len(self.pred_arrays_list),
        )
        self.gt_mol_list = joblib_map(
            partial(
                array_dict_to_molecule,
                save=save,
                save_dir_name=f"{save_dir}/gt",
            ),
            self.gt_arrays_list,
            n_jobs=n_jobs,
            inner_max_num_threads=1,
            desc="    G.T. to Molecule",
            total=len(self.gt_arrays_list),
        )

    def _get_metrics(self, pred, gt):
        """Compute the RMSD between two molecules."""
        try:
            rms_dist = self.matcher.get_rmsd(pred, gt)
            rms_dist = float("inf") if rms_dist == np.inf else rms_dist
            return rms_dist
        except Exception:
            return float("inf")

    def get_metrics(
        self, current_epoch: int = 0, save: bool = False, save_dir: str = "", n_jobs: int = -4
    ) -> Dict[str, Any]:
        """Compute the match rate and avg. RMS distance between predictions and ground truths.

        Note: self.rms_dists can be used to access RMSD per sample but is not returned.

        Returns:
            Dict: Dictionary of metrics, including match rate and avg. RMSD.
        """
        assert len(self.pred_arrays_list) == len(
            self.gt_arrays_list
        ), "Number of predictions and ground truths must match."

        # Convert predictions and ground truths to Molecule objects
        self._arrays_to_molecules(save, save_dir, n_jobs=n_jobs)

        self.rms_dists = []
        for i in tqdm(
            range(len(self.pred_mol_list)), desc=f"Epoch {current_epoch}, reconstruction eval"
        ):
            self.rms_dists.append(self._get_metrics(self.pred_mol_list[i], self.gt_mol_list[i]))
        self.rms_dists = torch.tensor(self.rms_dists, device=self.device)
        match_rate = (~torch.isinf(self.rms_dists)).long()
        if match_rate.sum() == 0:
            # No valid predictions --> return large RMSD for logging purposes
            return {
                "match_rate": match_rate,
                "rms_dist": torch.tensor([10.0] * len(match_rate), device=self.device),
            }
        else:
            return {
                "match_rate": match_rate,
                "rms_dist": self.rms_dists[~torch.isinf(self.rms_dists)],
            }

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = "") -> wandb.Table:
        """Create a wandb.Table object with the results of the evaluation."""
        pred_table = wandb.Table(
            columns=[
                "Epoch",
                "Sample idx",
                "Num atoms",
                "RMSD",
                "Match?",
                "True atom types",
                "Pred atom types",
                "True 2D",
                "Pred 2D",
                "True 3D",
                "Pred 3D",
            ]
        )
        for idx in range(len(self.pred_mol_list)):
            sample_idx = self.gt_arrays_list[idx]["sample_idx"]
            assert sample_idx == self.pred_arrays_list[idx]["sample_idx"]

            num_atoms = len(self.gt_mol_list[idx].atomic_numbers)

            rmsd = self.rms_dists[idx]

            match = rmsd != float("inf")

            true_atom_types = " ".join([str(int(t)) for t in self.gt_mol_list[idx].atomic_numbers])

            pred_atom_types = " ".join(
                [str(int(t)) for t in self.pred_mol_list[idx].atomic_numbers]
            )

            # 2D structures
            try:
                true_2d = wandb.Image(
                    Draw.MolToImage(
                        Chem.MolFromPDBFile(
                            f"{save_dir}/gt/molecule_{sample_idx}.pdb",
                            removeHs=False,
                        )
                    )
                )
            except Exception as e:
                # log.error(f"Failed to load 2D structure for true sample {sample_idx}.")
                true_2d = None
            try:
                pred_2d = wandb.Image(
                    Draw.MolToImage(
                        Chem.MolFromPDBFile(
                            f"{save_dir}/pred/molecule_{sample_idx}.pdb",
                            removeHs=False,
                        )
                    )
                )
            except Exception as e:
                # log.error(f"Failed to load 2D structure for predicted sample {sample_idx}.")
                pred_2d = None

            # 3D structures
            true_3d = wandb.Molecule(f"{save_dir}/gt/molecule_{sample_idx}.pdb")
            pred_3d = wandb.Molecule(f"{save_dir}/pred/molecule_{sample_idx}.pdb")

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                rmsd,
                match,
                true_atom_types,
                pred_atom_types,
                true_2d,
                pred_2d,
                true_3d,
                pred_3d,
            )

        return pred_table


def array_dict_to_molecule(
    x: dict[str, np.ndarray],
    save: bool = False,
    save_dir_name: str = "",
) -> Molecule:
    """Method to convert a dictionary of numpy arrays to a Molecule object which is compatible with
    MoleculeMatcher (used for evaluations).

    Args:
        x: Dictionary of numpy arrays with keys:
            - 'atom_types': Atomic numbers of atoms.
            - 'pos': 3D coordinates of atoms.
            - 'sample_idx': Index of the sample in the dataset.
        save: Whether to save the molecule as a pdb file.
        save_dir_name: Directory to save the pdb file.

    Returns:
        Molecule: Molecule object, optionally saved as a pdb file.
    """
    mol = Molecule(
        species=x["atom_types"], coords=x["pos"], properties={"sample_idx": x["sample_idx"]}
    )
    if save:
        os.makedirs(save_dir_name, exist_ok=True)
        mol.to(os.path.join(save_dir_name, f"molecule_{x['sample_idx']}.pdb"), fmt="pdb")
    return mol
