import chemparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

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


class SampleDataset(Dataset):
    def __init__(self, formula, num_evals):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.get_structure()

    def get_structure(self):
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
        )


train_dist = {
    "perov_5": [0, 0, 0, 0, 0, 1],
    "carbon_24": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3250697750779839,
        0.0,
        0.27795107535708424,
        0.0,
        0.15383352487276308,
        0.0,
        0.11246100804465604,
        0.0,
        0.04958134953209654,
        0.0,
        0.038745690362830404,
        0.0,
        0.019044491873255624,
        0.0,
        0.010178952552946971,
        0.0,
        0.007059596125430964,
        0.0,
        0.006074536200952225,
    ],
    "perov": [0, 0, 0, 0, 0, 1],
    "carbon": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3250697750779839,
        0.0,
        0.27795107535708424,
        0.0,
        0.15383352487276308,
        0.0,
        0.11246100804465604,
        0.0,
        0.04958134953209654,
        0.0,
        0.038745690362830404,
        0.0,
        0.019044491873255624,
        0.0,
        0.010178952552946971,
        0.0,
        0.007059596125430964,
        0.0,
        0.006074536200952225,
    ],
    "mp_20": [
        0.0,
        0.0021742334905660377,
        0.021079009433962265,
        0.019826061320754717,
        0.15271226415094338,
        0.047132959905660375,
        0.08464770047169812,
        0.021079009433962265,
        0.07808814858490566,
        0.03434551886792453,
        0.0972877358490566,
        0.013303360849056603,
        0.09669811320754718,
        0.02155807783018868,
        0.06522700471698113,
        0.014372051886792452,
        0.06703272405660378,
        0.00972877358490566,
        0.053176591981132074,
        0.010576356132075472,
        0.08995430424528301,
    ],
}


class GenDataset(Dataset):
    def __init__(self, dataset, total_num):
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.num_atoms = np.random.choice(
            len(self.distribution), total_num, p=self.distribution
        )
        self.is_carbon = dataset == "carbon_24"

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data
