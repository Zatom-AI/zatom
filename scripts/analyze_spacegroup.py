from pathlib import Path

from pymatgen.io.cif import CifParser

cif_dir = Path("viz/")

for file in cif_dir.iterdir():
    # Step 1: Load the CIF file
    parser = CifParser(file)
    structure = parser.parse_structures()[0]  # Get the first structure from the CIF file

    # Step 2: Write to a new CIF file with symmetry information
    output_cif_path = cif_dir / f"{file.stem}_sym.cif"
    structure.to(filename=output_cif_path, symprec=0.1)
