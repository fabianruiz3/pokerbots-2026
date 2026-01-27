# Toss'em Hold'em CFR Trainer (Fixed Version)

## Street System (7 streets, 0-6)

The game engine uses this street numbering:

| Street | Description | Actions |
|--------|-------------|---------|
| 0 | Preflop betting | Fold/Call/Check/Raise |
| 1 | Flop dealt | (skipped - no player actions) |
| 2 | BB Discard | Discard if BB, else CheckAction |
| 3 | SB Discard | Discard if SB, else CheckAction |
| 4 | Flop betting | Fold/Call/Check/Raise |
| 5 | Turn betting | Fold/Call/Check/Raise |
| 6 | River betting | Fold/Call/Check/Raise |

## Changes from Previous Version

1. **Street numbering fixed**: Now matches game engine (0, 2, 3, 4, 5, 6)
2. **Removed stack_bucket**: Simplified abstraction
3. **V2 binary format**: 75 bytes per node (vs 76+ in V1)
4. **Checkpointing**: Saves progress every 500k iterations

## Binary Format V2

Header (24 bytes):
- magic: 4 bytes (0x544F5353 = 'TOSS')
- version: 4 bytes (2)
- iterations: 8 bytes
- num_nodes: 8 bytes

Per node (75 bytes):
- player: 1 byte
- street: 1 byte (0, 2, 3, 4, 5, 6)
- hole_bucket: 2 bytes (0-39 for 3-card, 0-168 for 2-card)
- board_bucket: 2 bytes (0-24)
- pot_bucket: 1 byte (0-5)
- hist_bucket: 1 byte (0-5)
- flags: 1 byte (bb_discarded:1, sb_discarded:1, legal_mask:6)
- regret: 32 bytes (4 doubles)
- strat_sum: 32 bytes (4 doubles)
- reserved: 2 bytes

## Building

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Running

```bash
# Default: 1M iterations
./train_mccfr

# Custom settings
./train_mccfr -i 5000000 -t 32 -o cfr_strategy.bin

# Options:
#   -i, --iters       Total iterations (default: 1000000)
#   -t, --threads     Number of threads (default: auto)
#   -b, --batch       Batch size per thread (default: 20000)
#   -c, --checkpoint  Checkpoint interval (default: 500000)
#   -o, --out         Output file (default: cfr_strategy.bin)
```

## Expected Output

With sufficient iterations (5M+), expect:
- ~500k-2M unique states
- Good preflop coverage (street 0: 2-5% of nodes)
- Balanced distribution across streets 4, 5, 6

## Bucket Sizes

- Hole buckets (3-card): 40
- Hole buckets (2-card): 169
- Board buckets: 25
- Pot buckets: 6
- History buckets: 6
