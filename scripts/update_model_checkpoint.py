import argparse
import pickle  # nosec
import types
from collections import OrderedDict

import torch

from zatom.utils.typing_utils import typecheck


class ModulePathRemapperUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps old module paths to new ones.

    Inherits from the standard library pickle.Unpickler.
    """

    @typecheck
    def find_class(self, module: str, name: str) -> object:
        """Overrides the find_class method to remap module paths.

        Args:
            module: The original module path.
            name: The class name.

        Returns:
            The class object from the (possibly remapped) module.
        """
        # Define the mapping from the old path to the new path
        remap = {
            "zatom.models.architectures.dit.layers": "zatom.models.architectures.transformer.common"
        }

        # If the module path being requested is in our remap dictionary,
        # replace it with the new path.
        if module in remap:
            print(f"Remapping module path: '{module}' -> '{remap[module]}'")
            module = remap[module]

        # Proceed with the default find_class behavior from the parent class.
        return super().find_class(module, name)


@typecheck
def _create_custom_pickle_module() -> types.ModuleType:
    """Create a module-like object that wraps the standard pickle module but uses our custom
    Unpickler class.

    Returns:
        A module-like object with the custom Unpickler.
    """
    custom_pickle = types.ModuleType("custom_pickle")
    custom_pickle.Unpickler = ModulePathRemapperUnpickler

    # Copy all other attributes from the standard pickle module
    for attr in dir(pickle):
        if not attr.startswith("_") and attr != "Unpickler":
            try:
                setattr(custom_pickle, attr, getattr(pickle, attr))
            except (AttributeError, TypeError):
                pass

    return custom_pickle


@typecheck
def load_checkpoint_with_remap(path: str) -> dict:
    """Loads a checkpoint file, using a custom pickle module that handles module path remapping
    during unpickling."""
    print(f"Loading checkpoint with module path remapping from: {path}")

    custom_pickle = _create_custom_pickle_module()

    with open(path, "rb") as f:
        # Use pickle_module parameter (not unpickler) and set weights_only=False
        # to allow use of the custom pickle module
        checkpoint = torch.load(  # nosec
            f,
            map_location="cpu",
            pickle_module=custom_pickle,
            weights_only=False,
        )
    return checkpoint


# --- Main Checkpoint Merging Logic ---


@typecheck
def merge_checkpoints(source_ckpt_path: str, base_ckpt_path: str, output_path: str):
    """Loads two PyTorch Lightning checkpoints, merges them, and fixes legacy module paths."""
    print(f"Source checkpoint: {source_ckpt_path}")
    print(f"Base checkpoint: {base_ckpt_path}")

    source_ckpt = load_checkpoint_with_remap(source_ckpt_path)
    source_state_dict = source_ckpt["state_dict"]

    base_ckpt = load_checkpoint_with_remap(base_ckpt_path)
    base_state_dict = base_ckpt["state_dict"]

    new_state_dict = OrderedDict()
    replaced_keys_count = 0

    print("\nStarting merge process...")
    for key in base_state_dict.keys():
        if "global_property_head" in key:
            if key in source_state_dict:
                print(f"  - Replacing key: '{key}'")
                new_state_dict[key] = source_state_dict[key]
                replaced_keys_count += 1
            else:
                print(
                    f"  - WARNING: Key '{key}' found in base but not in source. Keeping original."
                )
                new_state_dict[key] = base_state_dict[key]
        else:
            new_state_dict[key] = base_state_dict[key]

    base_ckpt["state_dict"] = new_state_dict

    print(f"\nMerge complete. Replaced {replaced_keys_count} key(s).")

    try:
        torch.save(base_ckpt, output_path)
        print(f"Successfully saved merged and fixed checkpoint to: {output_path}")
        print("This new checkpoint can now be loaded directly without modifying sys.modules.")
    except Exception as e:
        print(f"An error occurred while saving the checkpoint: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge two PyTorch Lightning checkpoints, replacing specific keys and fixing legacy module paths.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "source_checkpoint",
        type=str,
        help="Path to the source checkpoint (.ckpt). Weights for 'global_property_head' will be taken from here.",
    )
    parser.add_argument(
        "base_checkpoint",
        type=str,
        help="Path to the base checkpoint (.ckpt). This checkpoint will be modified.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the new, merged, and path-corrected checkpoint file (.ckpt).",
    )

    args = parser.parse_args()

    try:
        import zatom.models.architectures.transformer.common
    except ImportError:
        print(
            "Error: The new module 'zatom.models.architectures.transformer.common' could not be imported."
        )
        print("Please ensure your current environment and PYTHONPATH are set up correctly.")
        exit(1)

    merge_checkpoints(
        source_ckpt_path=args.source_checkpoint,
        base_ckpt_path=args.base_checkpoint,
        output_path=args.output_file,
    )
