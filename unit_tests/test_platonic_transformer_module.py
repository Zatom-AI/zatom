import unittest
from types import SimpleNamespace

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from zatom.models.architectures.platoformer import (
    PLATONIC_GROUPS_3D,
    get_platonic_group,
)


class TestTransformerModulePlatonic(unittest.TestCase):
    """Unit tests for TransformerModulePlatonic."""

    def _get_input_data(self):
        """Helper method to prepare input data, shared by other unit tests."""
        cfg = OmegaConf.load("transformer_module_platonic_cfg.yaml")
        B, N = 6, 17
        V = cfg.multimodal_model.num_atom_types
        N_datasets = cfg.multimodal_model.dataset_embedder.num_classes
        N_spacegroups = cfg.multimodal_model.spacegroup_embedder.num_classes
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        # Generate input data
        atom_types = torch.randn(
            (B, N, V), device=device, dtype=dtype
        ).int()  # somehow expected to be int, but applies argmax(-1) to get index
        coords = torch.randn((B, N, 3), device=device, dtype=dtype)
        frac_coords = torch.randn((B, N, 3), device=device, dtype=dtype)
        lengths_scaled = torch.randn((B, 1, 3), device=device, dtype=dtype)
        angles_radians = torch.randn((B, 1, 3), device=device, dtype=dtype)

        x = (atom_types, coords, frac_coords, lengths_scaled, angles_radians)
        t = tuple(ti for ti in torch.rand((5, B), device=device, dtype=dtype))

        feats = {
            "dataset_idx": torch.randint(0, N_datasets, (B,), device=device),
            "spacegroup": torch.randint(0, N_spacegroups, (B,), device=device),
            "token_is_periodic": torch.arange(B, device=device).unsqueeze(-1).expand(B, N)
            > B // 2,
            "charge": torch.randint(-10, 10, (B,), device=device),
            "spin": torch.randint(-10, 10, (B,), device=device),
        }
        token_is_periodic = feats["token_is_periodic"].unsqueeze(-1)
        sample_is_periodic = token_is_periodic.any(-2, keepdim=True)

        ptcloud_lens = torch.randint(N // 4, N, size=(B, 1), device=device, dtype=dtype)
        padding_mask = torch.arange(N, device=device, dtype=dtype).view(1, N) < ptcloud_lens
        padding_mask[B // 2] = 1

        input_kwargs = vars(
            SimpleNamespace(
                t=t,
                feats=feats,
                padding_mask=padding_mask,
                token_is_periodic=token_is_periodic,
                sample_is_periodic=sample_is_periodic,
            )
        )

        return cfg, x, input_kwargs, B, N, device, dtype

    def test_equivariance_embeddings(self):
        """Isolated equivariance unit test for TransformerModulePlatonic._get_embedding."""
        cfg, x, input_kwargs, B, N, device, dtype = self._get_input_data()

        # Loop over some embedding options to unit test
        for coords_embed in ("coords_embed_none", "coords_embed_linear", "coords_embed_sinusoid"):
            for use_sequence_sin_ape in (True, False):
                for concat_combine_input in (True, False):
                    # Dynamically update model config
                    cfg.multimodal_model.coords_embed = cfg.coords_embed_options[coords_embed]
                    cfg.multimodal_model.use_sequence_sin_ape = use_sequence_sin_ape
                    cfg.multimodal_model.concat_combine_input = concat_combine_input

                    for solid_name in PLATONIC_GROUPS_3D:
                        group = get_platonic_group(solid_name)
                        G = group.G
                        cfg.multimodal_model.solid_name = solid_name

                        net = hydra.utils.instantiate(cfg.multimodal_model)
                        net = net.to(device=device, dtype=dtype)

                        # Original output, for non-transformed inputs
                        embed_x = net._get_embedding(x, **input_kwargs)

                        for g in range(G):
                            g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep
                            g_element = group.elements[g].to(device, dtype)

                            # Transformed inputs (only coords x[1] transform non-trivially)
                            g_coords = torch.einsum("ji,...j ->...i", g_element, x[1])
                            g_x = (x[0], g_coords, x[2], x[3], x[4])

                            # Output for transformed inputs
                            embed_g_x = net._get_embedding(g_x, **input_kwargs)

                            # Transformed output from non-transformed inputs
                            g_embed_x = embed_x.view(B, N, G, -1)[:, :, g_indices, :].flatten(-2)

                            self.assertTrue(
                                torch.allclose(embed_g_x, g_embed_x, atol=1e-5),
                                "\nEquivariance test failed for:\n"
                                + f"  G:              {solid_name}\n"
                                + f"  max difference: {torch.max(torch.abs(embed_g_x - g_embed_x))}",
                            )

    def test_equivariance_full_model(self):
        """Full equivariance unit test for TransformerModulePlatonic.forward."""
        modal_names = ("atom_types", "coords", "frac_coords", "lengths_scaled", "angles_radians")
        aux_names = ("property", "energy", "forces")
        cfg, x, input_kwargs, B, N, device, dtype = self._get_input_data()

        for use_cross_attn in (True, False):

            if not use_cross_attn:
                cfg.multimodal_model.cross_attn_factory = None
            cfg.multimodal_model.coords_embed = cfg.coords_embed_options.coords_embed_sinusoid

            for solid_name in tqdm(PLATONIC_GROUPS_3D):
                group = get_platonic_group(solid_name)
                G = group.G
                cfg.multimodal_model.solid_name = solid_name

                net = hydra.utils.instantiate(cfg.multimodal_model)
                net = net.to(device=device, dtype=dtype)

                # Original output, for non-transformed inputs
                pred_modals, pred_aux = net(x, **input_kwargs)

                for g in range(G):
                    g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep
                    g_element = group.elements[g].to(device, dtype)

                    # Transformed inputs
                    g_coords = torch.einsum("ji,...j ->...i", g_element, x[1])
                    g_x = (x[0], g_coords, x[2], x[3], x[4])

                    # Output for transformed inputs
                    pred_g_modals, pred_g_aux = net(g_x, **input_kwargs)

                    # Transformed output from non-transformed inputs
                    # Only coords and forces transform non-trivially
                    # Coordinate predictions:
                    g_pred_modals = [p.clone() for p in pred_modals]
                    g_pred_modals[1] = torch.einsum("ji,...j ->...i", g_element, g_pred_modals[1])
                    # Force predictions:
                    g_pred_aux = [p.clone() for p in pred_aux]
                    g_pred_aux[2] = torch.einsum("ji,...j ->...i", g_element, g_pred_aux[2])

                    for name, pred_g, g_pred in zip(
                        modal_names + aux_names,
                        pred_g_modals + pred_g_aux,
                        g_pred_modals + g_pred_aux,
                    ):
                        self.assertTrue(
                            torch.allclose(pred_g, g_pred, atol=5e-5),
                            "\nEquivariance test failed for:\n"
                            + f"  G:              {solid_name}\n"
                            + f"  max difference: {torch.max(torch.abs(pred_g - g_pred))}",
                        )


if __name__ == "__main__":
    unittest.main()
