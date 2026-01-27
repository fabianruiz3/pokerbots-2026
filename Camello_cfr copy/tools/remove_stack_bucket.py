#!/usr/bin/env python3
"""Collapse the stack_bucket dimension in a tossem CFR binary table.

What it does
------------
- Reads a tossem CFR table (.bin) produced by tossem_cfr_cpp (magic 'TOSS').
- Merges entries that are identical in every key field *except* stack_bucket.
- Writes a new .bin with the same record layout, but with stack_bucket=0 for all
  keys and merged regrets/strategy-sums.

Why keep the same record layout?
-------------------------------
So you don't have to change the C++/Python loader format. Your runtime bot can
simply set stack_bucket=0 (or keep your backoff that drops stack_bucket) and use
this smaller table.

Binary format (little-endian)
-----------------------------
Header:
  u32 magic        0x544F5353 ('TOSS')
  u32 version
  i64 iterations
  u64 num_nodes

Per-node key:
  u8  player
  u8  street
  u16 hole_bucket
  u16 board_bucket
  u8  pot_bucket
  u8  stack_bucket
  u8  hist_bucket
  u8  bb_discarded
  u8  sb_discarded
  u8  legal_mask

Then:
  double regrets[num_actions]
  double strat_sum[num_actions]

Defaults:
  num_actions=4 (matches the provided tossem_cfr_cpp build)

Usage
-----
python tools/remove_stack_bucket.py --in cfr_strategy.bin --out cfr_strategy_nostack.bin

Optional:
  --num_actions 4
  --merge sum|avg   (default: sum)
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

MAGIC_TOSS = 0x544F5353  # 'TOSS' little-endian when read as u32


def read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"Unexpected EOF: wanted {n} bytes, got {len(b)}")
    return b


@dataclass(frozen=True)
class Key:
    player: int
    street: int
    hole_bucket: int
    board_bucket: int
    pot_bucket: int
    stack_bucket: int
    hist_bucket: int
    bb_discarded: int
    sb_discarded: int
    legal_mask: int

    def without_stack(self) -> Tuple[int, ...]:
        return (
            self.player,
            self.street,
            self.hole_bucket,
            self.board_bucket,
            self.pot_bucket,
            # stack omitted
            self.hist_bucket,
            self.bb_discarded,
            self.sb_discarded,
            self.legal_mask,
        )


def read_header(f) -> Tuple[int, int, int, int]:
    magic = struct.unpack("<I", read_exact(f, 4))[0]
    if magic != MAGIC_TOSS:
        raise ValueError(
            f"Bad magic: expected 0x{MAGIC_TOSS:08X} ('TOSS'), got 0x{magic:08X}"
        )
    version = struct.unpack("<I", read_exact(f, 4))[0]
    iterations = struct.unpack("<q", read_exact(f, 8))[0]
    num_nodes = struct.unpack("<Q", read_exact(f, 8))[0]
    return magic, version, iterations, num_nodes


def write_header(f, version: int, iterations: int, num_nodes: int) -> None:
    f.write(struct.pack("<I", MAGIC_TOSS))
    f.write(struct.pack("<I", int(version)))
    f.write(struct.pack("<q", int(iterations)))
    f.write(struct.pack("<Q", int(num_nodes)))


def read_key(f) -> Key:
    # Matches tossem_cfr_cpp write order
    player = struct.unpack("<B", read_exact(f, 1))[0]
    street = struct.unpack("<B", read_exact(f, 1))[0]
    hole_bucket = struct.unpack("<H", read_exact(f, 2))[0]
    board_bucket = struct.unpack("<H", read_exact(f, 2))[0]
    pot_bucket = struct.unpack("<B", read_exact(f, 1))[0]
    stack_bucket = struct.unpack("<B", read_exact(f, 1))[0]
    hist_bucket = struct.unpack("<B", read_exact(f, 1))[0]
    bb_discarded = struct.unpack("<B", read_exact(f, 1))[0]
    sb_discarded = struct.unpack("<B", read_exact(f, 1))[0]
    legal_mask = struct.unpack("<B", read_exact(f, 1))[0]
    return Key(
        player=player,
        street=street,
        hole_bucket=hole_bucket,
        board_bucket=board_bucket,
        pot_bucket=pot_bucket,
        stack_bucket=stack_bucket,
        hist_bucket=hist_bucket,
        bb_discarded=bb_discarded,
        sb_discarded=sb_discarded,
        legal_mask=legal_mask,
    )


def write_key(f, k: Key) -> None:
    f.write(struct.pack("<B", k.player))
    f.write(struct.pack("<B", k.street))
    f.write(struct.pack("<H", k.hole_bucket))
    f.write(struct.pack("<H", k.board_bucket))
    f.write(struct.pack("<B", k.pot_bucket))
    f.write(struct.pack("<B", k.stack_bucket))
    f.write(struct.pack("<B", k.hist_bucket))
    f.write(struct.pack("<B", k.bb_discarded))
    f.write(struct.pack("<B", k.sb_discarded))
    f.write(struct.pack("<B", k.legal_mask))


def read_vec_doubles(f, n: int) -> List[float]:
    raw = read_exact(f, 8 * n)
    return list(struct.unpack("<" + "d" * n, raw))


def write_vec_doubles(f, xs: List[float]) -> None:
    f.write(struct.pack("<" + "d" * len(xs), *xs))


def sanitize(xs: List[float]) -> Tuple[List[float], int]:
    bad = 0
    out = []
    for x in xs:
        if x is None or not math.isfinite(x):
            bad += 1
            out.append(0.0)
        else:
            out.append(float(x))
    return out, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input .bin path")
    ap.add_argument("--out", dest="out", required=True, help="Output .bin path")
    ap.add_argument(
        "--num_actions",
        type=int,
        default=4,
        help="Number of actions stored per node (default 4)",
    )
    ap.add_argument(
        "--merge",
        choices=["sum", "avg"],
        default="sum",
        help="How to merge regrets/strategy-sums across stack buckets (default sum)",
    )
    args = ap.parse_args()

    num_actions = int(args.num_actions)

    merged: Dict[Tuple[int, ...], Tuple[Key, List[float], List[float], int]] = {}
    bad_vals = 0

    with open(args.inp, "rb") as f:
        magic, version, iterations, num_nodes = read_header(f)

        for i in range(num_nodes):
            k = read_key(f)
            regrets = read_vec_doubles(f, num_actions)
            strat = read_vec_doubles(f, num_actions)

            regrets, b1 = sanitize(regrets)
            strat, b2 = sanitize(strat)
            bad_vals += b1 + b2

            k2_tuple = k.without_stack()
            if k2_tuple not in merged:
                k_out = Key(
                    player=k.player,
                    street=k.street,
                    hole_bucket=k.hole_bucket,
                    board_bucket=k.board_bucket,
                    pot_bucket=k.pot_bucket,
                    stack_bucket=0,
                    hist_bucket=k.hist_bucket,
                    bb_discarded=k.bb_discarded,
                    sb_discarded=k.sb_discarded,
                    legal_mask=k.legal_mask,
                )
                merged[k2_tuple] = (k_out, regrets, strat, 1)
            else:
                k_out, r_acc, s_acc, cnt = merged[k2_tuple]
                for j in range(num_actions):
                    r_acc[j] += regrets[j]
                    s_acc[j] += strat[j]
                merged[k2_tuple] = (k_out, r_acc, s_acc, cnt + 1)

    # Optionally average
    if args.merge == "avg":
        for kk, (k_out, r_acc, s_acc, cnt) in list(merged.items()):
            if cnt > 1:
                for j in range(num_actions):
                    r_acc[j] /= cnt
                    s_acc[j] /= cnt
                merged[kk] = (k_out, r_acc, s_acc, cnt)

    # Deterministic order
    items = sorted(merged.values(), key=lambda t: (
        t[0].player,
        t[0].street,
        t[0].hole_bucket,
        t[0].board_bucket,
        t[0].pot_bucket,
        t[0].hist_bucket,
        t[0].bb_discarded,
        t[0].sb_discarded,
        t[0].legal_mask,
    ))

    with open(args.out, "wb") as f:
        write_header(f, version=version, iterations=iterations, num_nodes=len(items))
        for (k_out, r_acc, s_acc, _cnt) in items:
            write_key(f, k_out)
            write_vec_doubles(f, r_acc)
            write_vec_doubles(f, s_acc)

    print(
        f"Read {num_nodes} nodes -> wrote {len(items)} nodes (stack collapsed). "
        f"Bad numeric values replaced: {bad_vals}."
    )


if __name__ == "__main__":
    main()
