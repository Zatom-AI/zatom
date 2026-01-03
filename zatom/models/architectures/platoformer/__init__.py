from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS

# Keep only those groups acting on R^3 (not R^2)
PLATONIC_GROUPS_3D = {name: group for name, group in PLATONIC_GROUPS.items() if group.dim == 3}


def get_platonic_group(solid_name: str) -> None:
    """Return Platonic group for the given solid_name."""
    if solid_name not in PLATONIC_GROUPS_3D:
        raise ValueError(
            f"Invalid solid_name '{solid_name}'. Must be one of: {list(PLATONIC_GROUPS_3D)}."
        )
    return PLATONIC_GROUPS_3D[solid_name]
