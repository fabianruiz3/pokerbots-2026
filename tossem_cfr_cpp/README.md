# Toss'em Hold'em MCCFR (C++)

This is a **standalone, fast C++ external-sampling MCCFR trainer** for the Pokerbots 2026 Toss'em Hold'em abstraction.

Why this exists:
- OpenSpiel MCCFR is fast **only when the game is in C++**. When the game dynamics are implemented in Python, MCCFR calls back into Python for every `state.child()` and becomes **extremely slow** (what you saw as ~2 iter/s).
- This trainer keeps the hot loop entirely in C++ (including state transitions + bucketing), but matches your Python abstraction shape:
  - same streets / discards
  - same 4 betting actions
  - info-state = tuple of small ints + optional legal-action mask
  - discards are **not learned** here (uniform / averaged), leaving discard learning to a separate DiscardNet if you want.

## Build

```bash
cd tossem_cfr_cpp
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

## Run

```bash
./build/train_mccfr --iters 2000000 --threads 8 --batch 200000 --out cfr_strategy.bin
```

It prints per-batch throughput and number of discovered info-states.

## Output format

Binary file `cfr_strategy.bin`:
- magic = `TCFR1` (5 bytes)
- `int64 iterations`
- `uint64 num_nodes`
- then repeated per-node:
  - `InfoKey` fields (player, street, hole_bucket, board_bucket, pot_bucket, stack_bucket, hist_bucket, bb_discarded, sb_discarded, legal_mask)
  - `regret[4]` doubles
  - `strategy_sum[4]` doubles

You can load this into Python or your C++ runtime bot.

## Notes
- This is intended as the **first big step** toward “all-C++” training without needing to compile a custom OpenSpiel game.
- If you later want OpenSpiel’s exploitability tools, you can still implement the game in C++ and plug into OpenSpiel, but that is a larger build + integration effort.
