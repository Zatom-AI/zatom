import time
from collections import defaultdict
from typing import DefaultDict, Dict, List, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# constants

INT_TYPE = int | np.int32 | np.int64
PACKING_FUNCTION = Literal["spfhp", "lpfhp"]


# helper functions


def spfhp_add_pack(
    pack: List[INT_TYPE],
    count: INT_TYPE,
    tmp: DefaultDict[INT_TYPE, List[Tuple[INT_TYPE, List[INT_TYPE]]]],
    final: DefaultDict[INT_TYPE, List[Tuple[INT_TYPE, List[INT_TYPE]]]],
    limit: INT_TYPE,
    offset: INT_TYPE = 0,
):
    """Filter out packs that reached maximum length or number of sequences for the SPFHP algorithm.

    :param pack: List of sequence lengths in the pack. :param count: Number of sequences in the
    pack. :param tmp: Temporary dictionary to hold packs that are not yet     finalized. :param
    final: Final dictionary to hold completed packs. :param limit: Maximum number of sequences per
    pack. :param offset: Offset to adjust the pack length.
    """
    if len(pack) == limit or offset == 0:
        final[offset].append((count, pack))
    else:
        tmp[offset].append((count, pack))


def lpfhp_add_pack(
    pack: List[INT_TYPE],
    count: INT_TYPE,
    tmp: DefaultDict[INT_TYPE, List[Tuple[INT_TYPE, List[INT_TYPE]]]],
    final: DefaultDict[INT_TYPE, List[Tuple[INT_TYPE, List[INT_TYPE]]]],
    limit: INT_TYPE,
    offset: INT_TYPE = 0,
    max_sequence_length: INT_TYPE = 512,
):
    """Filter out packs that have reached the maximum length or number of components for the LPFHP
    algorithm.

    Adapted from
    https://github.com/graphcore/examples/blob/v3.2.0/tutorials/blogs_code/packedBERT/lpfhp.py.

    :param pack: List of sequence lengths in the pack.
    :param count: Number of sequences in the pack.
    :param tmp: Temporary dictionary to hold packs that are not yet
    finalized.
    :param final: Final dictionary to hold completed packs.
    :param limit: Maximum number of sequences per pack.
    :param offset: Offset to adjust the pack length.
    :param max_sequence_length: Maximum sequence length allowed.
    """
    assert max_sequence_length - sum(pack) == offset, "Incorrect offset."
    assert offset >= 0, "Too small offset."
    assert offset < max_sequence_length, "Too large offset."
    if len(pack) == limit or offset == 0:
        final[offset].append((count, pack))
    else:
        tmp[offset].append((count, pack))


def pack_using_spfhp(
    histogram: np.ndarray,
    max_sequence_length: int,
    max_sequences_per_pack: int | str = "max",
    verbose: bool = True,
) -> Tuple[List[List[int]], np.ndarray]:
    """Execute shortest-pack-first histogram-packing (SPFHP).

    Adapted from
    https://github.com/graphcore/examples/blob/v3.2.0/tutorials/blogs_code/packedBERT/spfhp.py.

    :param histogram: Histogram (NumPy array) of sequence lengths.
    :param max_sequence_length: Maximum sequence length.
    :param max_sequences_per_pack: Maximum number of sequences per pack,
    or "max" for no limit.
    :param verbose: Whether to print detailed information about the
    packing process.
    :return: A tuple containing a list of packed sequences and a NumPy
    array of their counts.
    """
    start = time.time()
    reversed_histogram = np.flip(histogram)

    # Initialize main strategy data dictionary.
    # The key indicates how many tokens are left for full length.
    # The value is a list of tuples, consisting of counts and respective packs.
    # A pack is a (sorted) list of sequence length values that get concatenated.
    tmp_strategies_per_length = defaultdict(list)
    strategies_per_length = defaultdict(list)

    # Index i indicates here, how much space is left, due to reversed histogram.
    for i in range(max_sequence_length):
        n_sequences_to_bin = reversed_histogram[i]
        length_to_bin = max_sequence_length - i
        offset = i + 1  # largest possible offset
        while n_sequences_to_bin > 0:
            if (length_to_bin + offset) in tmp_strategies_per_length:
                # Extract shortest pack that will get modified.
                n_sequences_to_pack, pack = tmp_strategies_per_length[length_to_bin + offset].pop()
                new_pack = pack + [length_to_bin]
                count = min(n_sequences_to_pack, n_sequences_to_bin)

                if n_sequences_to_pack > n_sequences_to_bin:
                    # Old pack gets reduced.
                    n_sequences_to_pack -= n_sequences_to_bin
                    tmp_strategies_per_length[length_to_bin + offset].append(
                        (n_sequences_to_pack, pack)
                    )
                    n_sequences_to_bin = 0
                else:
                    n_sequences_to_bin -= n_sequences_to_pack

                spfhp_add_pack(
                    new_pack,
                    count,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_sequences_per_pack,
                    offset,
                )

                # Clean up to speed up main key search.
                if not tmp_strategies_per_length[length_to_bin + offset]:
                    tmp_strategies_per_length.pop(length_to_bin + offset)

            else:
                offset -= 1

            # Does not fit anywhere. Create new pack.
            if offset < 0:
                spfhp_add_pack(
                    [length_to_bin],
                    n_sequences_to_bin,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_sequences_per_pack,
                    i,
                )
                n_sequences_to_bin = 0

    # Merge all strategies.
    for key in tmp_strategies_per_length:
        strategies_per_length[key].extend(tmp_strategies_per_length[key])

    # Flatten strategies dictionary.
    strategy_set = []
    strategy_repeat_count = []
    for key in strategies_per_length:
        for count, pack in strategies_per_length[key]:
            pack.reverse()
            strategy_set.append(pack)
            strategy_repeat_count.append(count)

    # Summarize efficiency of solution.
    duration = time.time() - start
    sequence_lengths = np.arange(1, max_sequence_length + 1)
    strategy_repeat_count = np.array(strategy_repeat_count)
    old_number_of_samples = histogram.sum()
    new_number_of_samples = strategy_repeat_count.sum()
    total_tokens = max_sequence_length * new_number_of_samples
    empty_tokens = sum(
        [
            count * (max_sequence_length - sum(pack))
            for count, pack in zip(strategy_repeat_count, strategy_set)
        ]
    )
    efficiency = 100 - empty_tokens / total_tokens * 100
    speedup_upper_bound = 1.0 / (
        1
        - (histogram * (1 - sequence_lengths / max_sequence_length)).sum() / old_number_of_samples
    )

    if verbose:
        print(
            f"SPFHP packing efficiency (fraction of real tokens): {efficiency:3.4f}\n"
            f"Speed-up theoretical limit: {speedup_upper_bound:3.4f}\n"
            f"Achieved speed-up over un-packed dataset: {old_number_of_samples/new_number_of_samples:3.5f}\n"
            f"Runtime: Packed {old_number_of_samples} sequences in {duration:3.3f} seconds."
        )

    return strategy_set, np.array(strategy_repeat_count)


def pack_using_lpfhp(
    histogram: np.ndarray,
    max_sequence_length: int,
    max_sequences_per_pack: int | str = "max",
    distribute: bool = True,
    verbose: bool = True,
) -> Tuple[List[List[int]], np.ndarray]:
    """Execute longest-pack-first histogram-packing (LPFHP).

    Adapted from
    https://github.com/graphcore/examples/blob/v3.2.0/tutorials/blogs_code/packedBERT/lpfhp.py.

    :param histogram: Histogram (NumPy array) of sequence lengths.
    :param max_sequence_length: Maximum sequence length.
    :param max_sequences_per_pack: Maximum number of sequences per pack,
    or "max" for no limit.
    :param distribute: Whether to distribute sequences evenly across
    packs.
    :param verbose: Whether to print detailed information about the
    packing process.
    :return: A tuple containing a list of packed sequences and a NumPy
    array of their counts.
    """
    start = time.time()
    reversed_histogram = np.flip(histogram)

    # Initialize main strategy data dictionary.
    # The key indicates how many tokens are left for full length.
    # The value is a list of tuples, consisting of counts and respective packs.
    # A pack is a (sorted) list of sequence length values that get concatenated.
    tmp_strategies_per_length = defaultdict(list)
    strategies_per_length = defaultdict(list)
    if max_sequences_per_pack == "max":
        max_sequences_per_pack = max_sequence_length

    # Index i indicates here, how much space is left, due to reversed histogram.
    for i in range(max_sequence_length):
        n_sequences_to_bin = reversed_histogram[i]
        length_to_bin = max_sequence_length - i
        offset = 0  # Smallest possible offset for perfect fit.
        while n_sequences_to_bin > 0:
            if (length_to_bin + offset) in tmp_strategies_per_length:
                # Extract worst pack that will get modified.
                n_sequences_to_pack, pack = tmp_strategies_per_length[length_to_bin + offset].pop()

                # Calculate how often the current sequence maximally fits in.
                repeat = min(1 + offset // length_to_bin, max_sequences_per_pack - len(pack))

                # Correct dependent on count.
                while n_sequences_to_bin // repeat == 0:
                    repeat -= 1
                if not distribute:
                    repeat = 1
                new_pack = pack + [length_to_bin] * repeat
                count = min(n_sequences_to_pack, n_sequences_to_bin // repeat)
                if n_sequences_to_pack > count:
                    # Old pack gets reduced.
                    n_sequences_to_pack -= count
                    tmp_strategies_per_length[length_to_bin + offset].append(
                        (n_sequences_to_pack, pack)
                    )
                    n_sequences_to_bin -= count * repeat
                else:
                    n_sequences_to_bin -= n_sequences_to_pack * repeat

                lpfhp_add_pack(
                    new_pack,
                    count,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_sequences_per_pack,
                    offset - (repeat - 1) * length_to_bin,
                    max_sequence_length,
                )

                # Clean up to speed up main key search.
                if not tmp_strategies_per_length[length_to_bin + offset]:
                    tmp_strategies_per_length.pop(length_to_bin + offset)

                # Reset offset in case best fit changed.
                offset = 0
            else:
                offset += 1

            # Does not fit anywhere. Create new pack.
            if offset >= max_sequence_length - length_to_bin + 1:
                # Similar repetition but no dependence on pack.
                repeat = min(max_sequence_length // length_to_bin, max_sequences_per_pack)
                while n_sequences_to_bin // repeat == 0:
                    repeat -= 1
                if not distribute:
                    repeat = 1

                lpfhp_add_pack(
                    [length_to_bin] * repeat,
                    n_sequences_to_bin // repeat,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_sequences_per_pack,
                    max_sequence_length - length_to_bin * repeat,
                    max_sequence_length,
                )
                n_sequences_to_bin -= n_sequences_to_bin // repeat * repeat

    # Merge all strategies.
    for key in tmp_strategies_per_length:
        strategies_per_length[key].extend(tmp_strategies_per_length[key])

    # Flatten strategies dictionary.
    strategy_set = []
    strategy_repeat_count = []
    for key in strategies_per_length:
        for count, pack in strategies_per_length[key]:
            pack.reverse()
            strategy_set.append(pack)
            strategy_repeat_count.append(count)

    # Summarize efficiency of solution.
    duration = time.time() - start
    sequence_lengths = np.arange(1, max_sequence_length + 1)
    strategy_repeat_count = np.array(strategy_repeat_count)
    old_number_of_samples = histogram.sum()
    new_number_of_samples = strategy_repeat_count.sum()
    total_tokens = max_sequence_length * new_number_of_samples
    empty_tokens = sum(
        [
            count * (max_sequence_length - sum(pack))
            for count, pack in zip(strategy_repeat_count, strategy_set)
        ]
    )
    efficiency = 100 - empty_tokens / total_tokens * 100
    speedup_upper_bound = 1.0 / (
        1
        - (histogram * (1 - sequence_lengths / max_sequence_length)).sum() / old_number_of_samples
    )

    if verbose:
        print(
            f"LPFHP packing efficiency (fraction of real tokens): {efficiency:3.4f}\n"
            f"Speed-up theoretical limit: {speedup_upper_bound:3.4f}\n"
            f"Achieved speed-up over un-packed dataset: {old_number_of_samples/new_number_of_samples:3.5f}\n"
            f"Runtime: Packed {old_number_of_samples} sequences in {duration:3.3f} seconds."
        )

    return strategy_set, strategy_repeat_count


# classes


PACKING_FUNCTION_MAP = {
    "spfhp": pack_using_spfhp,
    "lpfhp": pack_using_lpfhp,
}


class PackedDataset(Dataset):
    """Dataset that packs examples from a tokenized dataset into packs of sequences, by default
    using a pack-first histogram-packing (PFHP) algorithm.

    :param tokenized_dataset: The tokenized dataset to pack.
    :param tokenized_data_len_key: Key in the dataset that contains the
        sequence lengths. Defaults to "seq" which will be prefixed
        by `data_features_prefix`.
    :param data_features_prefix: Prefix for the keys in the
        dataset that contain the features to pack. Defaults to an empty
        string.
    :param data_features: List of features to pack from the dataset.
        Defaults to ["seq"]. Position IDs (pos_ids) and sequence IDs
        (seq_ids) are automatically added to the packed features.
    :param packing_fn: Function to use for packing sequences, defaults
        to "lpfhp" (longest-pack-first histogram-packing). Can also be
        "spfhp" (shortest-pack-first histogram-packing).
    :param max_seq_len: Maximum length of sequences in the
        dataset.
    :param max_seq_per_pack: Maximum number of sequences in a pack,
        or "max" for no limit.
    :param distribute: Whether to distribute sequences evenly across
        packs.
    :param plot_distribution: Whether to plot the distribution of
        sequence lengths in the packs.
    :param overfitting: If True, only uses the first two examples for
        overfitting purposes.
    :param verbose: Whether to print detailed information about the
        packing process.
    """

    def __init__(
        self,
        tokenized_dataset: Dataset,
        data_len_key: str = "seq",
        data_features_prefix: str = "",
        data_features: List[str] = ["seq"],
        packing_fn: PACKING_FUNCTION = "lpfhp",
        max_seq_len: int = 128,
        max_seq_per_pack: int | str = 6,
        distribute: bool = True,
        plot_distribution: bool = False,
        overfitting: bool = False,
        verbose: bool = True,
    ):
        self.tokenized_dataset = tokenized_dataset
        self.data_len_key = data_len_key
        self.data_features_prefix = data_features_prefix
        self.data_features = data_features
        self.max_seq_len = max_seq_len
        self.max_seq_per_pack = max_seq_per_pack
        self.overfitting = overfitting

        # 1) Collect lengths of each example.
        dataset_seq_lens = np.array(
            [
                len(self.tokenized_dataset[i][f"{data_features_prefix}{data_len_key}"])
                for i in tqdm(
                    range(len(self.tokenized_dataset)),
                    desc="Collecting sequence lengths for packing" if verbose else "",
                )
            ]
        )

        # 2) Build histogram for lengths `1..max_seq_len`.
        histogram = np.zeros(max_seq_len, dtype=np.int64)
        seq_lens, strategy_repeat_count = np.unique(dataset_seq_lens, return_counts=True)
        histogram[seq_lens - 1] = strategy_repeat_count
        # histogram[i-1] = # of examples of length i

        # 3) Call the provided packing function.
        packing_fn = PACKING_FUNCTION_MAP.get(packing_fn, None)
        assert packing_fn is not None, f"Unknown packing function: {packing_fn}"

        strategy_set, strategy_repeat_count = packing_fn(
            histogram=histogram,
            max_sequence_length=max_seq_len,
            max_sequences_per_pack=max_seq_per_pack,
            distribute=distribute,
            verbose=verbose,
        )
        # strategy_set: List[List[int]] sequence‐length lists
        # strategy_repeat_count: np.ndarray giving how many packs of each pattern

        if plot_distribution:
            import matplotlib.pyplot as plt

            # Formatting.
            plt.style.use(plt.style.available[-2])
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rcParams["figure.figsize"] = (10, 4)
            plt.rcParams["figure.dpi"] = 100

            # Plotting.
            plt.subplot(1, 2, 1)
            plt.hist(histogram, bins=[k for k in range(0, max_seq_len, 10)])
            plt.title("Sequence length histogram")
            plt.xlabel("Sequence lengths")
            plt.ylabel("Frequency")

            plt.subplot(1, 2, 2)
            plt.plot(np.arange(max_seq_len) + 1, histogram)
            plt.title("Sequence length distribution")
            plt.xlabel("Sequence lengths")
            plt.ylabel("Num. samples")

            plt.show()

        # 4) Bucket example indices by length.
        buckets: dict[int, List[int]] = defaultdict(list)
        for idx, L in enumerate(dataset_seq_lens):
            buckets[L].append(idx)

        # 5) Instantiate packs of indices.
        self.packs: List[List[int]] = []
        for _, (strategy, count) in enumerate(zip(strategy_set, strategy_repeat_count)):
            for __ in range(int(count)):
                pack_idxs: List[int] = []
                for L in strategy:
                    try:
                        pack_idxs.append(buckets[L].pop())
                    except IndexError:
                        raise RuntimeError(f"Not enough examples of length {L} to fill packs")
                self.packs.append(pack_idxs)

    def __len__(self) -> int:
        """Return the number of packs in the dataset."""
        return len(self.packs)

    def __getitem__(self, pack_idx: int) -> Dict[str, torch.Tensor]:
        """Return a pack of sequences by its index.

        :param pack_idx: Index of the pack to retrieve. :return: A dictionary containing the packed
        features for the     specified pack index.
        """
        prefix = self.data_features_prefix
        ids = [0, 1] if self.overfitting else self.packs[pack_idx]
        packed_features = defaultdict(list)

        # Load all features from tokenized dataset.
        for i in ids:
            tokenized_example = self.tokenized_dataset.__getitem__(i)
            for feature in self.data_features:
                key = f"{prefix}{feature}"
                packed_features[key].append(tokenized_example[key])

            # Add position and sequence IDs.
            length = tokenized_example[f"{prefix}{self.data_len_key}"].shape[0]
            packed_features[f"{prefix}pos_ids"].append(torch.arange(1, length + 1))
            packed_features[f"{prefix}seq_ids"].append(
                torch.full((length,), i + 1, dtype=torch.long)
            )

        # Concatenate all entries.
        for key in packed_features:
            packed_features[key] = torch.cat(packed_features[key], dim=0)

        return dict(packed_features)
