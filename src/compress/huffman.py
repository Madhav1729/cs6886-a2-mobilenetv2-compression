"""Canonical Huffman coding over quantized weight codes, written from scratch
(plain heap, no zlib/gzip/library). Quantized codes cluster near zero, so
actual entropy sits well below the nominal bit-width -- Huffman recovers that
gap losslessly, for free."""

import heapq
from collections import Counter
from typing import Dict, Iterable, Tuple

import torch


def code_lengths(symbol_counts: Dict[int, int]) -> Dict[int, int]:
    """Optimal prefix-code length per symbol, via Huffman tree construction.

    Returns {symbol: bit_length}. A single distinct symbol still costs 1 bit.
    """
    if not symbol_counts:
        return {}
    if len(symbol_counts) == 1:
        return {next(iter(symbol_counts)): 1}

    # heap entries: (weight, tie_breaker, {symbol: depth_so_far})
    heap = [(count, i, {sym: 0}) for i, (sym, count) in enumerate(symbol_counts.items())]
    heapq.heapify(heap)
    tie = len(heap)

    while len(heap) > 1:
        w1, _, d1 = heapq.heappop(heap)
        w2, _, d2 = heapq.heappop(heap)
        merged = {s: depth + 1 for s, depth in d1.items()}
        merged.update({s: depth + 1 for s, depth in d2.items()})
        heapq.heappush(heap, (w1 + w2, tie, merged))
        tie += 1

    return heap[0][2]


def encoded_bits(codes: torch.Tensor) -> Tuple[int, int]:
    """Total bits to Huffman-encode `codes`, plus the code-table cost.

    Table cost assumes canonical Huffman: one byte of code length per symbol,
    which is all a decoder needs to rebuild the table.
    """
    counts = Counter(codes.flatten().tolist())
    lengths = code_lengths(counts)
    payload = sum(counts[sym] * lengths[sym] for sym in counts)
    table = len(counts) * 8
    return payload, table


def entropy_bits_per_symbol(codes: torch.Tensor) -> float:
    """Shannon entropy of the code distribution -- the Huffman lower bound."""
    _, counts = torch.unique(codes.flatten(), return_counts=True)
    p = counts.float() / counts.sum()
    return float(-(p * p.log2()).sum())


def total_encoded_bits(code_tensors: Iterable[torch.Tensor]) -> Tuple[int, int]:
    """Encode several tensors with one shared code table (cheaper than per-tensor)."""
    counts = Counter()
    for t in code_tensors:
        counts.update(t.flatten().tolist())
    lengths = code_lengths(counts)
    payload = sum(counts[sym] * lengths[sym] for sym in counts)
    return payload, len(counts) * 8
