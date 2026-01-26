"""
Print all preflop equities and analyze the normalization.
"""

import pickle
import sys

def print_equity_table(table_file='preflop_equity_table.pkl', output_file='preflop_equities.txt'):
    """Print all equities sorted by strength."""
    
    try:
        with open(table_file, 'rb') as f:
            table_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {table_file} not found!")
        print("Please provide the path to your preflop_equity_table.pkl file")
        return
    
    equity_table = table_data['equity_table']
    
    # Sort by equity descending
    sorted_hands = sorted(equity_table.items(), key=lambda x: x[1], reverse=True)
    
    rank_names = {14:'A', 13:'K', 12:'Q', 11:'J', 10:'T', 9:'9', 8:'8', 7:'7', 6:'6', 5:'5', 4:'4', 3:'3', 2:'2'}
    suit_names = {0: 'rainbow', 1: 'two-suited', 2: 'three-suited'}
    
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PREFLOP 3-CARD HAND EQUITIES (vs random hand)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total unique hand classes: {len(equity_table)}\n")
        f.write(f"Simulations per hand: {table_data.get('sims_per_hand', 'unknown')}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rank':<6} {'Hand':<20} {'Suit Pattern':<15} {'Equity':<10}\n")
        f.write("-" * 70 + "\n")
        
        for i, (hand_class, equity) in enumerate(sorted_hands, 1):
            r1, r2, r3, suit_pattern = hand_class
            hand_str = f"{rank_names[r1]}{rank_names[r2]}{rank_names[r3]}"
            suit_str = suit_names[suit_pattern]
            f.write(f"{i:<6} {hand_str:<20} {suit_str:<15} {equity:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        # Top 20
        f.write("TOP 20 HANDS:\n")
        for i, (hand_class, equity) in enumerate(sorted_hands[:20], 1):
            r1, r2, r3, suit_pattern = hand_class
            hand_str = f"{rank_names[r1]}{rank_names[r2]}{rank_names[r3]}"
            suit_str = suit_names[suit_pattern]
            f.write(f"  {i:2}. {hand_str} ({suit_str}): {equity:.4f}\n")
        
        f.write("\n")
        
        # Bottom 20
        f.write("BOTTOM 20 HANDS:\n")
        for i, (hand_class, equity) in enumerate(sorted_hands[-20:], len(sorted_hands)-19):
            r1, r2, r3, suit_pattern = hand_class
            hand_str = f"{rank_names[r1]}{rank_names[r2]}{rank_names[r3]}"
            suit_str = suit_names[suit_pattern]
            f.write(f"  {i:2}. {hand_str} ({suit_str}): {equity:.4f}\n")
        
        f.write("\n")
        
        # Stats
        equities = list(equity_table.values())
        f.write(f"Equity Statistics:\n")
        f.write(f"  Min: {min(equities):.4f}\n")
        f.write(f"  Max: {max(equities):.4f}\n")
        f.write(f"  Avg: {sum(equities)/len(equities):.4f}\n")
        
        # Count by suit pattern
        f.write(f"\nHands by suit pattern:\n")
        for sp in [0, 1, 2]:
            count = sum(1 for h in equity_table if h[3] == sp)
            avg_eq = sum(equity_table[h] for h in equity_table if h[3] == sp) / max(1, count)
            f.write(f"  {suit_names[sp]}: {count} hands, avg equity {avg_eq:.4f}\n")
    
    print(f"Wrote {len(equity_table)} hand equities to {output_file}")
    return output_file


def analyze_normalization_problem():
    """
    Show why the current normalization loses information.
    """
    print("\n" + "=" * 70)
    print("NORMALIZATION PROBLEM ANALYSIS")
    print("=" * 70)
    
    print("""
The current normalize_hand() function has a FLAW:

It treats these hands as IDENTICAL:
  [As, Ks, 9h] → (14, 13, 9, 1)  "two-suited"
  [As, Kh, 9s] → (14, 13, 9, 1)  "two-suited"
  [Ah, Ks, 9s] → (14, 13, 9, 1)  "two-suited"

But they have DIFFERENT equities because:
  - [As, Ks, 9h] = AK suited (spades) + 9 kicker → STRONG flush draw
  - [As, Kh, 9s] = A9 suited (spades) + K kicker → WEAKER flush draw
  - [Ah, Ks, 9s] = K9 suited (spades) + A kicker → MEDIUM flush draw

The suited cards determine flush potential, not just "some two cards match."
""")
    
    print("BETTER NORMALIZATION:")
    print("""
Track WHICH cards are suited:
  - (14, 13, 9, 'high')  → highest two cards suited (AK suited)
  - (14, 13, 9, 'low')   → lowest two cards suited (K9 suited)  
  - (14, 13, 9, 'ends')  → high and low suited (A9 suited)
  - (14, 13, 9, 'all')   → all three suited
  - (14, 13, 9, 'none')  → rainbow

This creates ~5x more hand classes but much more accurate equities.
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        table_file = sys.argv[1]
    else:
        table_file = 'preflop_equity_table_v2.pkl'
    
    output_file = print_equity_table(table_file, 'preflop_equities.txt')
    analyze_normalization_problem()