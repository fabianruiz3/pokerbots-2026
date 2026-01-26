"""
Robust Python loader for C++ CFR strategy binary.

Supports:
- Standard C++ format written as:
  uint32 magic = 0x544F5353 ('TOSS' as u32; bytes appear as b'SSOT' in file)
  uint32 version
  int64  iterations
  uint64 num_nodes
  then per-node records.

Auto-detects per-node record layout by using:
- file size
- header fields
- num_nodes

Also supports "nostack" variants where stack_bucket is omitted from the key.

Adds:
- safe get_action_probs signature that accepts legal_actions kwarg
- miss instrumentation + unique miss tracking
- nostack aggregated fallback (drop stack bucket at lookup time)
"""

import os
import struct
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from abstraction import (
    NUM_ACTIONS,
    get_hole_bucket_3card, get_hole_bucket_2card, get_board_bucket,
    get_pot_bucket, get_stack_bucket, get_history_bucket,
    compute_legal_mask,
)

MAGIC_U32_TOSS = 0x544F5353  # 'TOSS' as a uint32 (little-endian bytes in file look like b'SSOT')


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError("Unexpected EOF")
    return b


@dataclass(frozen=True)
class InfoKey:
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


class CppCFR:
    """
    Loads CFR strategy binary and provides action probability lookups.

    Key features:
    - Robust binary parsing (auto-detect record layout)
    - Exact-then-backoff lookup that can ignore stack bucket via aggregated index
    - Miss instrumentation (grouped by dimension) + unique miss keys
    """

    def __init__(self, bin_path: str, drop_stack_fallback: bool = True, verbose: bool = True):
        self.bin_path = bin_path
        self.verbose = verbose
        self.drop_stack_fallback = drop_stack_fallback

        self.strategy: Dict[InfoKey, List[float]] = {}
        # Aggregated index that ignores stack_bucket: (key with stack_bucket=ANY) -> avg strat
        self.strategy_nostack: Dict[Tuple[int, int, int, int, int, int, int, int, int], List[float]] = {}

        self.num_nodes: int = 0
        self.iterations: int = 0

        # Detected format
        self.num_actions: int = NUM_ACTIONS
        self.value_dtype: str = "d"  # 'd' (double) or 'f' (float)
        self.has_stack_bucket: bool = True
        self.arrays_per_node: int = 2  # 2 => regret+strat, 1 => strat only

        # Miss instrumentation
        self.miss_total = 0
        self.miss_unique = set()

        self.miss_by_street: Dict[int, int] = {}
        self.miss_by_hist: Dict[int, int] = {}
        self.miss_by_pot: Dict[int, int] = {}
        self.miss_by_board: Dict[int, int] = {}
        self.miss_by_hole: Dict[int, int] = {}

        if not os.path.exists(bin_path):
            if self.verbose:
                print(f"[CppCFR] WARNING: Strategy file not found: {bin_path}")
                print("[CppCFR] Will use uniform random strategy as fallback")
            return

        t0 = time.time()
        self._load_binary(bin_path)
        t1 = time.time()
        if self.verbose:
            print(
                f"[CppCFR] Loaded {self.num_nodes} nodes from {bin_path} "
                f"({self.iterations} iterations) in {t1 - t0:.2f}s"
            )
            if self.drop_stack_fallback:
                print(f"[CppCFR] Built nostack index with {len(self.strategy_nostack)} keys")

    def _detect_format(self, file_size: int, header_size: int, num_nodes: int) -> Tuple[int, str, bool, int, int]:
        """
        Infer (node_size, dtype_char, has_stack_bucket, arrays_per_node, num_actions)
        by matching remaining bytes to plausible per-node layouts.
        """
        remaining = file_size - header_size
        if num_nodes <= 0:
            return 0, "d", True, 2, NUM_ACTIONS
        if remaining % num_nodes != 0:
            raise ValueError(
                f"[CppCFR] File size mismatch: remaining={remaining} not divisible by num_nodes={num_nodes} "
                f"(header_size={header_size}, file_size={file_size})"
            )

        node_size = remaining // num_nodes

        # Key layouts we might see:
        # With stack:  1+1+2+2+1+1+1+1+1+1 = 12 bytes
        # No stack:    1+1+2+2+1+  1+1+1+1+1 = 11 bytes (stack_bucket omitted)
        # (player, street, hole_u16, board_u16, pot_u8, [stack_u8], hist_u8, bb_u8, sb_u8, legal_u8)
        key_sizes = [(12, True), (11, False)]

        # Dtypes to try
        dtype_candidates = [("d", 8), ("f", 4)]  # double, float

        # Prefer your project defaults first
        preferred_actions = [NUM_ACTIONS, 4, 5, 6, 3, 2]

        # Try to match layouts:
        # data_bytes = node_size - key_size
        # arrays_per_node in {1,2}
        # data_bytes == arrays_per_node * num_actions * bytes_per_value
        best = None
        candidates = []

        for key_size, has_stack in key_sizes:
            data_bytes = node_size - key_size
            if data_bytes <= 0:
                continue
            for dtype_char, bpv in dtype_candidates:
                for arrays_per_node in (2, 1):
                    # num_actions must be integer
                    denom = arrays_per_node * bpv
                    if data_bytes % denom != 0:
                        continue
                    na = data_bytes // denom
                    if na < 2 or na > 16:
                        continue
                    candidates.append((dtype_char, has_stack, arrays_per_node, na))

        if not candidates:
            raise ValueError(
                f"[CppCFR] Could not detect node layout: node_size={node_size} bytes. "
                f"Try checking your writer format / remove script."
            )

        # Rank candidates: prefer (double, has_stack=True, arrays_per_node=2, na close to NUM_ACTIONS)
        def score(c):
            dtype_char, has_stack, arrays_per_node, na = c
            s = 0
            s += 100 if dtype_char == "d" else 0
            s += 30 if has_stack else 0
            s += 20 if arrays_per_node == 2 else 0
            # closeness / preference of num_actions
            if na in preferred_actions:
                s += 10 * (len(preferred_actions) - preferred_actions.index(na))
            s -= abs(na - NUM_ACTIONS)
            return s

        best = max(candidates, key=score)
        dtype_char, has_stack_bucket, arrays_per_node, num_actions = best
        return node_size, dtype_char, has_stack_bucket, arrays_per_node, num_actions

    def _load_binary(self, bin_path: str) -> None:
        file_size = os.path.getsize(bin_path)
        with open(bin_path, "rb") as f:
            # Header
            magic_u32 = struct.unpack("<I", _read_exact(f, 4))[0]
            if magic_u32 != MAGIC_U32_TOSS:
                # Helpful debug
                f.seek(0)
                first8 = f.read(8)
                raise ValueError(
                    f"Bad magic header: {first8!r} (u32={hex(magic_u32)} expected u32={hex(MAGIC_U32_TOSS)})"
                )

            version = struct.unpack("<I", _read_exact(f, 4))[0]
            if version != 1 and self.verbose:
                print(f"[CppCFR] WARNING: Unknown version {version}, attempting to load anyway")

            self.iterations = struct.unpack("<q", _read_exact(f, 8))[0]  # int64
            self.num_nodes = struct.unpack("<Q", _read_exact(f, 8))[0]   # uint64

            header_size = f.tell()

            # Auto-detect record layout
            node_size, dtype_char, has_stack, arrays_per_node, num_actions = self._detect_format(
                file_size=file_size, header_size=header_size, num_nodes=self.num_nodes
            )
            self.value_dtype = dtype_char
            self.has_stack_bucket = has_stack
            self.arrays_per_node = arrays_per_node
            self.num_actions = num_actions

            if self.verbose:
                print(
                    f"[CppCFR] Detected layout: node_size={node_size}, dtype={dtype_char}, "
                    f"has_stack={has_stack}, arrays_per_node={arrays_per_node}, num_actions={num_actions}"
                )

            # Build nostack aggregator as sums then normalize
            agg_sum: Dict[Tuple[int, int, int, int, int, int, int, int, int], List[float]] = {}
            agg_cnt: Dict[Tuple[int, int, int, int, int, int, int, int, int], int] = {}

            # Node parsing helpers
            value_fmt = "<" + (dtype_char * self.num_actions)
            value_bytes = (8 if dtype_char == "d" else 4) * self.num_actions

            for _ in range(self.num_nodes):
                player = struct.unpack("<B", _read_exact(f, 1))[0]
                street = struct.unpack("<B", _read_exact(f, 1))[0]
                hole_bucket = struct.unpack("<H", _read_exact(f, 2))[0]
                board_bucket = struct.unpack("<H", _read_exact(f, 2))[0]
                pot_bucket = struct.unpack("<B", _read_exact(f, 1))[0]

                if self.has_stack_bucket:
                    stack_bucket = struct.unpack("<B", _read_exact(f, 1))[0]
                else:
                    stack_bucket = 0  # omitted in file; treat as 0

                hist_bucket = struct.unpack("<B", _read_exact(f, 1))[0]
                bb_discarded = struct.unpack("<B", _read_exact(f, 1))[0]
                sb_discarded = struct.unpack("<B", _read_exact(f, 1))[0]
                legal_mask = struct.unpack("<B", _read_exact(f, 1))[0]

                # arrays_per_node: 2 => regret then strat_sum, 1 => strat_sum only
                if self.arrays_per_node == 2:
                    _ = struct.unpack(value_fmt, _read_exact(f, value_bytes))  # regret (unused)
                    strat_vals = list(struct.unpack(value_fmt, _read_exact(f, value_bytes)))
                else:
                    strat_vals = list(struct.unpack(value_fmt, _read_exact(f, value_bytes)))

                # Convert strategy sums to average strategy
                total = 0.0
                for s in strat_vals:
                    if s > 0:
                        total += s
                if total > 0:
                    avg = [(s if s > 0 else 0.0) / total for s in strat_vals]
                else:
                    avg = [1.0 / self.num_actions] * self.num_actions

                key = InfoKey(
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
                self.strategy[key] = avg

                nostack_key = (
                    player, street, hole_bucket, board_bucket, pot_bucket,
                    hist_bucket, bb_discarded, sb_discarded, legal_mask
                )
                if nostack_key not in agg_sum:
                    agg_sum[nostack_key] = [0.0] * self.num_actions
                    agg_cnt[nostack_key] = 0
                for i in range(self.num_actions):
                    agg_sum[nostack_key][i] += avg[i]
                agg_cnt[nostack_key] += 1

            # finalize nostack index
            for k, s in agg_sum.items():
                c = agg_cnt[k]
                if c > 0:
                    self.strategy_nostack[k] = [x / c for x in s]

    def _build_info_key(
        self,
        player: int,
        street: int,
        hole_cards: List[str],
        board_cards: List[str],
        pot: int,
        effective_stack: int,
        betting_history: List[Tuple[int, int]],
        bb_discarded: bool,
        sb_discarded: bool,
        legal_actions: Optional[List[int]] = None,
    ) -> InfoKey:
        # Hole bucket
        if len(hole_cards) >= 3:
            hole_bucket = get_hole_bucket_3card(hole_cards)
        else:
            hole_bucket = get_hole_bucket_2card(hole_cards)

        board_bucket = get_board_bucket(board_cards)
        pot_bucket = get_pot_bucket(pot)
        stack_bucket = get_stack_bucket(effective_stack)
        hist_bucket = get_history_bucket(betting_history)

        if legal_actions is None:
            legal_mask = 0
        else:
            legal_mask = compute_legal_mask(legal_actions)

        return InfoKey(
            player=player,
            street=street,
            hole_bucket=hole_bucket,
            board_bucket=board_bucket,
            pot_bucket=pot_bucket,
            stack_bucket=stack_bucket,
            hist_bucket=hist_bucket,
            bb_discarded=1 if bb_discarded else 0,
            sb_discarded=1 if sb_discarded else 0,
            legal_mask=legal_mask,
        )

    def _record_miss(self, key: InfoKey) -> None:
        self.miss_total += 1

        miss_sig = (
            key.player, key.street, key.hole_bucket, key.board_bucket,
            key.pot_bucket, key.stack_bucket, key.hist_bucket,
            key.bb_discarded, key.sb_discarded, key.legal_mask
        )
        self.miss_unique.add(miss_sig)

        self.miss_by_street[key.street] = self.miss_by_street.get(key.street, 0) + 1
        self.miss_by_hist[key.hist_bucket] = self.miss_by_hist.get(key.hist_bucket, 0) + 1
        self.miss_by_pot[key.pot_bucket] = self.miss_by_pot.get(key.pot_bucket, 0) + 1
        self.miss_by_board[key.board_bucket] = self.miss_by_board.get(key.board_bucket, 0) + 1
        self.miss_by_hole[key.hole_bucket] = self.miss_by_hole.get(key.hole_bucket, 0) + 1

    def get_action_probs(
        self,
        player: int,
        street: int,
        hole_cards: List[str],
        board_cards: List[str],
        pot: int,
        effective_stack: int,
        betting_history: List[Tuple[int, int]],
        bb_discarded: bool,
        sb_discarded: bool,
        legal_actions: Optional[List[int]] = None,
        **kwargs: Any,  # ignore unexpected kwargs safely
    ) -> Dict[int, float]:
        """
        Returns dict[action_id] = probability, normalized over the provided legal_actions if given.

        - First tries exact key (includes stack_bucket)
        - If miss and drop_stack_fallback enabled, tries nostack aggregated key
        - If still miss, uses uniform over legal actions
        """
        key = self._build_info_key(
            player=player,
            street=street,
            hole_cards=hole_cards,
            board_cards=board_cards,
            pot=pot,
            effective_stack=effective_stack,
            betting_history=betting_history,
            bb_discarded=bb_discarded,
            sb_discarded=sb_discarded,
            legal_actions=legal_actions,
        )

        strat = self.strategy.get(key)
        if strat is None and self.drop_stack_fallback:
            nostack_key = (
                key.player, key.street, key.hole_bucket, key.board_bucket, key.pot_bucket,
                key.hist_bucket, key.bb_discarded, key.sb_discarded, key.legal_mask
            )
            strat = self.strategy_nostack.get(nostack_key)

        if strat is None:
            self._record_miss(key)

            # Uniform fallback over legal actions
            if legal_actions:
                p = 1.0 / max(1, len(legal_actions))
                return {a: p for a in legal_actions}
            return {a: 1.0 / self.num_actions for a in range(self.num_actions)}

        # Filter to legal actions (if provided)
        if legal_actions:
            total = 0.0
            out = {}
            for a in legal_actions:
                if 0 <= a < len(strat):
                    out[a] = max(0.0, float(strat[a]))
                    total += out[a]
            if total > 0:
                for a in out:
                    out[a] /= total
                return out
            # fallback uniform if all zero
            p = 1.0 / max(1, len(legal_actions))
            return {a: p for a in legal_actions}

        # If no legal_actions supplied, return full distribution
        return {a: float(strat[a]) for a in range(min(self.num_actions, len(strat)))}
    
    # inside class CppCFR:
    def debug_miss_summary(self, topk: int = 5) -> str:
        """
        Backwards-compatible alias used by player.py.
        """
        # If you already have miss_summary(), just delegate.
        if hasattr(self, "miss_summary"):
            return self.miss_summary(topk=topk)

        # Fallback if your file uses a different internal stats object name
        stats = getattr(self, "stats", None)
        if stats is not None and hasattr(stats, "summary"):
            return stats.summary(topk=topk)

        return "[CppCFR] No miss-summary stats available."


    def miss_summary(self, topk: int = 8) -> str:
        """
        Small human-readable snapshot of miss stats.
        """
        uniq = len(self.miss_unique)
        tot = self.miss_total
        if tot == 0:
            return "[CppCFR] MISS summary: no misses recorded."

        def top_items(d):
            return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:topk]

        parts = [
            f"[CppCFR] MISS summary: total={tot}, unique={uniq}",
            f"  by_street: {top_items(self.miss_by_street)}",
            f"  by_hist:   {top_items(self.miss_by_hist)}",
            f"  by_pot:    {top_items(self.miss_by_pot)}",
            f"  by_board:  {top_items(self.miss_by_board)}",
            f"  by_hole:   {top_items(self.miss_by_hole)}",
        ]
        return "\n".join(parts)

