#!/usr/bin/env python3
"""
Diagnostic tool to analyze CFR strategy binary contents.
Run this locally where you have the cfr_strategy.bin file.

Usage: python3 diagnose_cfr.py /path/to/cfr_strategy.bin
"""

import struct
import sys
from collections import Counter

def analyze_binary(bin_path):
    with open(bin_path, 'rb') as f:
        # Read header
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        iterations = struct.unpack('<q', f.read(8))[0]
        num_nodes = struct.unpack('<Q', f.read(8))[0]
        
        print(f"=== CFR Binary Analysis ===")
        print(f"Magic: {hex(magic)} ({'TOSS' if magic == 0x544F5353 else 'UNKNOWN'})")
        print(f"Version: {version}")
        print(f"Iterations: {iterations:,}")
        print(f"Nodes: {num_nodes:,}")
        
        header_size = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(header_size)
        
        remaining = file_size - header_size
        node_size = remaining // num_nodes
        print(f"File size: {file_size:,} bytes")
        print(f"Record size: {node_size} bytes per node")
        
        # Detect if stack bucket is present
        # With stack: key = 12 bytes (1+1+2+2+1+1+1+1+1+1)
        # No stack:   key = 11 bytes (1+1+2+2+1+1+1+1+1)
        # Data for 4 actions, 2 arrays (regret + strat), double = 4*2*8 = 64 bytes
        # Data for 4 actions, 2 arrays, float = 4*2*4 = 32 bytes
        
        if node_size == 12 + 64:
            print("Layout: WITH stack bucket, double precision")
            has_stack = True
        elif node_size == 11 + 64:
            print("Layout: NO stack bucket, double precision")
            has_stack = False
        elif node_size == 12 + 32:
            print("Layout: WITH stack bucket, float precision")
            has_stack = True
        elif node_size == 11 + 32:
            print("Layout: NO stack bucket, float precision")
            has_stack = False
        else:
            print(f"Layout: UNKNOWN (node_size={node_size})")
            has_stack = None
        
        # Sample nodes
        print(f"\n=== Sampling {min(5000, num_nodes)} nodes ===")
        
        street_counts = Counter()
        pot_counts = Counter()
        hist_counts = Counter()
        bb_disc_counts = Counter()
        sb_disc_counts = Counter()
        player_counts = Counter()
        
        sample_size = min(5000, num_nodes)
        for i in range(sample_size):
            rec = f.read(node_size)
            if len(rec) < 11:
                break
            
            player = rec[0]
            street = rec[1]
            hole_bucket = struct.unpack('<H', rec[2:4])[0]
            board_bucket = struct.unpack('<H', rec[4:6])[0]
            pot_bucket = rec[6]
            
            if has_stack:
                stack_bucket = rec[7]
                hist_bucket = rec[8]
                bb_discarded = rec[9]
                sb_discarded = rec[10]
            else:
                hist_bucket = rec[7]
                bb_discarded = rec[8]
                sb_discarded = rec[9]
            
            player_counts[player] += 1
            street_counts[street] += 1
            pot_counts[pot_bucket] += 1
            hist_counts[hist_bucket] += 1
            bb_disc_counts[bb_discarded] += 1
            sb_disc_counts[sb_discarded] += 1
        
        print(f"\nPlayer distribution:")
        for p, c in sorted(player_counts.items()):
            print(f"  Player {p}: {c} ({100*c/sample_size:.1f}%)")
        
        print(f"\nStreet distribution:")
        street_names = {0: 'PREFLOP', 1: 'FLOP', 2: 'BB_DISCARD', 3: 'SB_DISCARD', 4: 'TURN', 5: 'RIVER'}
        for s, c in sorted(street_counts.items()):
            name = street_names.get(s, f'UNKNOWN_{s}')
            print(f"  Street {s} ({name}): {c} ({100*c/sample_size:.1f}%)")
        
        print(f"\nPot bucket distribution:")
        for p, c in sorted(pot_counts.items()):
            print(f"  Pot {p}: {c} ({100*c/sample_size:.1f}%)")
        
        print(f"\nHistory bucket distribution:")
        for h, c in sorted(hist_counts.items()):
            print(f"  Hist {h}: {c} ({100*c/sample_size:.1f}%)")
        
        print(f"\nBB discarded distribution:")
        for d, c in sorted(bb_disc_counts.items()):
            print(f"  BB_disc={d}: {c} ({100*c/sample_size:.1f}%)")
        
        print(f"\nSB discarded distribution:")
        for d, c in sorted(sb_disc_counts.items()):
            print(f"  SB_disc={d}: {c} ({100*c/sample_size:.1f}%)")
        
        print("\n=== What Python expects ===")
        print("Street 0 (PREFLOP): board_len=0")
        print("Street 1 (FLOP): board_len=2, no discards yet")
        print("Street 4 (TURN): board_len=5, both discarded")
        print("Street 5 (RIVER): board_len=6, both discarded")
        print("\nIf the binary has NO street 4/5 nodes, that's the mismatch!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} ./cfr_strategy.bin")
        sys.exit(1)
    analyze_binary(sys.argv[1])
