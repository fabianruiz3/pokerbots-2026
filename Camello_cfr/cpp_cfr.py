"""
C++ CFR Strategy Loader - V2 Format (75 bytes per node, no stack_bucket)

Binary format V2:
  Header (24 bytes):
    - magic: 4 bytes ('TOSS')
    - version: 4 bytes (should be 2)
    - iterations: 8 bytes
    - num_nodes: 8 bytes
    
  Per node (75 bytes):
    - player: 1 byte
    - street: 1 byte
    - hole_bucket: 2 bytes (uint16)
    - board_bucket: 2 bytes (uint16)
    - pot_bucket: 1 byte
    - hist_bucket: 1 byte
    - flags: 1 byte (bb_discarded:1, sb_discarded:1, legal_mask:6)
    - regret: 32 bytes (4 doubles)
    - strat_sum: 32 bytes (4 doubles)
    - reserved: 2 bytes
"""

import struct
import os
from collections import defaultdict

from abstraction import (
    FOLD, CHECK_CALL, RAISE_SMALL, RAISE_LARGE, NUM_ACTIONS,
    get_hole_bucket, get_board_bucket, get_pot_bucket, get_history_bucket,
    card_str_to_int, compute_legal_mask
)


class CppCFR:
    """Loader and lookup for C++ CFR strategy binary (V2 format)."""
    
    def __init__(self, bin_path='cfr_strategy.bin'):
        self.nodes = {}
        self.iterations = 0
        self.num_nodes = 0
        self._last_lookup_hit = False
        
        # Debug tracking
        self._miss_counts = defaultdict(int)
        self._total_lookups = 0
        self._hits = 0
        
        if os.path.exists(bin_path):
            self._load_binary(bin_path)
        else:
            print(f"[CppCFR] WARNING: Strategy file not found: {bin_path}")
    
    def _load_binary(self, path):
        """Load V2 binary format (75 bytes per node)."""
        with open(path, 'rb') as f:
            # Header
            magic, version, iterations, num_nodes = struct.unpack('<IIQQ', f.read(24))
            
            if magic != 0x544F5353:  # 'TOSS'
                raise ValueError(f"Invalid magic: {hex(magic)}")
            
            if version == 1:
                print(f"[CppCFR] WARNING: V1 format detected, attempting V1 load...")
                self._load_binary_v1(path)
                return
            elif version != 2:
                raise ValueError(f"Unsupported version: {version}")
            
            self.iterations = iterations
            self.num_nodes = num_nodes
            
            print(f"[CppCFR] Loading V2: {num_nodes} nodes, {iterations} iterations")
            
            # Read nodes (75 bytes each)
            for _ in range(num_nodes):
                data = f.read(75)
                if len(data) < 75:
                    break
                
                # Unpack key (9 bytes)
                player = data[0]
                street = data[1]
                hole_bucket = struct.unpack('<H', data[2:4])[0]
                board_bucket = struct.unpack('<H', data[4:6])[0]
                pot_bucket = data[6]
                hist_bucket = data[7]
                flags = data[8]
                
                bb_discarded = (flags & 0x80) != 0
                sb_discarded = (flags & 0x40) != 0
                legal_mask = flags & 0x3F
                
                # Unpack data (64 bytes)
                regret = struct.unpack('<4d', data[9:41])
                strat_sum = struct.unpack('<4d', data[41:73])
                # reserved = struct.unpack('<H', data[73:75])  # Ignored
                
                # Create key tuple
                key = (player, street, hole_bucket, board_bucket, pot_bucket,
                       hist_bucket, int(bb_discarded), int(sb_discarded), legal_mask)
                
                self.nodes[key] = {
                    'regret': list(regret),
                    'strat_sum': list(strat_sum)
                }
            
            print(f"[CppCFR] Loaded {len(self.nodes)} nodes")
    
    def _load_binary_v1(self, path):
        """Fallback loader for V1 format (with stack_bucket)."""
        with open(path, 'rb') as f:
            # Header
            magic, version, iterations, num_nodes = struct.unpack('<IIQQ', f.read(24))
            
            self.iterations = iterations
            self.num_nodes = num_nodes
            
            print(f"[CppCFR] Loading V1: {num_nodes} nodes, {iterations} iterations")
            
            # V1 format has 10-byte key + 64-byte data = 74+ bytes per node
            # Try to detect node size
            remaining = os.path.getsize(path) - 24
            bytes_per_node = remaining // num_nodes if num_nodes > 0 else 74
            
            for _ in range(num_nodes):
                # V1 key (10 bytes with stack_bucket)
                player = struct.unpack('B', f.read(1))[0]
                street = struct.unpack('B', f.read(1))[0]
                hole_bucket = struct.unpack('<H', f.read(2))[0]
                board_bucket = struct.unpack('<H', f.read(2))[0]
                pot_bucket = struct.unpack('B', f.read(1))[0]
                stack_bucket = struct.unpack('B', f.read(1))[0]  # V1 has this
                hist_bucket = struct.unpack('B', f.read(1))[0]
                bb_discarded = struct.unpack('B', f.read(1))[0]
                sb_discarded = struct.unpack('B', f.read(1))[0]
                legal_mask = struct.unpack('B', f.read(1))[0]
                
                regret = struct.unpack('<4d', f.read(32))
                strat_sum = struct.unpack('<4d', f.read(32))
                
                # Convert to V2 key format (ignore stack_bucket)
                key = (player, street, hole_bucket, board_bucket, pot_bucket,
                       hist_bucket, bb_discarded, sb_discarded, legal_mask)
                
                self.nodes[key] = {
                    'regret': list(regret),
                    'strat_sum': list(strat_sum)
                }
            
            print(f"[CppCFR] Loaded {len(self.nodes)} nodes (V1 format)")
    
    def _make_key(self, player, street, hole_bucket, board_bucket, pot_bucket,
                  hist_bucket, bb_discarded, sb_discarded, legal_mask):
        """Create lookup key tuple."""
        return (player, street, hole_bucket, board_bucket, pot_bucket,
                hist_bucket, int(bb_discarded), int(sb_discarded), legal_mask)
    
    def get_action_probs(self, player, street, hole_cards, board_cards, pot,
                         effective_stack, betting_history, bb_discarded, sb_discarded,
                         legal_actions):
        """
        Get action probabilities from CFR strategy.
        
        Args:
            player: 0 (SB) or 1 (BB)
            street: Game engine street number (0-6)
            hole_cards: List of card strings (e.g., ['Ah', 'Kd', '2c'])
            board_cards: List of card strings
            pot: Current pot size
            effective_stack: Min of both stacks
            betting_history: List of (player, action_id) tuples
            bb_discarded: Whether BB has discarded
            sb_discarded: Whether SB has discarded
            legal_actions: List of legal action IDs
        
        Returns:
            Dict mapping action_id -> probability
        """
        self._total_lookups += 1
        
        # Compute buckets
        hole_bucket = get_hole_bucket(hole_cards)
        board_bucket = get_board_bucket(board_cards)
        pot_bucket = get_pot_bucket(pot)
        hist_bucket = get_history_bucket(betting_history)
        legal_mask = compute_legal_mask(legal_actions)
        
        # Create key
        key = self._make_key(player, street, hole_bucket, board_bucket, pot_bucket,
                            hist_bucket, bb_discarded, sb_discarded, legal_mask)
        
        # Debug output
        # print(f"[DEBUG] Lookup: street={street}, hole={hole_bucket}, board={board_bucket}, pot={pot_bucket}, hist={hist_bucket}, bb={bb_discarded}, sb={sb_discarded}")
        
        # Lookup
        node = self.nodes.get(key)
        
        if node is None:
            self._last_lookup_hit = False
            self._miss_counts[(street, hole_bucket, board_bucket, pot_bucket, hist_bucket)] += 1
            # Return uniform over legal actions
            probs = {}
            for a in legal_actions:
                if 0 <= a < NUM_ACTIONS:
                    probs[a] = 1.0 / len(legal_actions)
            return probs
        
        self._last_lookup_hit = True
        self._hits += 1
        
        # Regret matching
        strat_sum = node['strat_sum']
        total = sum(max(0, strat_sum[a]) for a in legal_actions if 0 <= a < NUM_ACTIONS)
        
        probs = {}
        if total > 0:
            for a in legal_actions:
                if 0 <= a < NUM_ACTIONS:
                    probs[a] = max(0, strat_sum[a]) / total
        else:
            # Uniform if no strategy accumulated
            for a in legal_actions:
                if 0 <= a < NUM_ACTIONS:
                    probs[a] = 1.0 / len(legal_actions)
        
        return probs
    
    def debug_miss_summary(self, topk=5):
        """Get summary of most common misses."""
        sorted_misses = sorted(self._miss_counts.items(), key=lambda x: -x[1])[:topk]
        lines = ["[CppCFR] Top misses:"]
        for (street, hole, board, pot, hist), count in sorted_misses:
            lines.append(f"  street={street} hole={hole} board={board} pot={pot} hist={hist}: {count}")
        return "\n".join(lines)
    
    def debug_street_distribution(self):
        """Get distribution of nodes by street."""
        street_counts = defaultdict(int)
        for key in self.nodes:
            street_counts[key[1]] += 1
        
        lines = ["[CppCFR] Nodes by street:"]
        for street in sorted(street_counts.keys()):
            pct = 100 * street_counts[street] / len(self.nodes) if self.nodes else 0
            lines.append(f"  Street {street}: {street_counts[street]} ({pct:.1f}%)")
        return "\n".join(lines)
    
    def debug_hist_distribution(self):
        """Get distribution of nodes by history bucket."""
        hist_counts = defaultdict(int)
        for key in self.nodes:
            hist_counts[key[5]] += 1  # hist_bucket is at index 5
        
        lines = ["[CppCFR] Nodes by history bucket:"]
        for hist in sorted(hist_counts.keys()):
            pct = 100 * hist_counts[hist] / len(self.nodes) if self.nodes else 0
            lines.append(f"  Hist {hist}: {hist_counts[hist]} ({pct:.1f}%)")
        return "\n".join(lines)
