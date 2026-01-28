from pymatgen.io.cif import CifParser

# Step 1: Load the CIF file
cif_file_path = "crystal.cif"
parser = CifParser(cif_file_path)
structure = parser.get_structures()[0]  # Get the first structure from the CIF file

# Step 2: Write to a new CIF file with symmetry information
output_cif_path = "crystal_sym.cif"
structure.to(filename=output_cif_path, symprec=0.01)

print(f"New CIF file with symmetry information written to: {output_cif_path}")
