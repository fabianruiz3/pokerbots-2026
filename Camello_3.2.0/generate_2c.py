"""
STEP 1: Generate 2-Card Equity Table

This computes REAL equity for all 169 unique 2-card hands:
- 13 pairs (AA, KK, QQ, ... 22)
- 78 suited combos (AKs, AQs, ... 32s)
- 78 offsuit combos (AKo, AQo, ... 32o)

Uses Monte Carlo simulation: your 2 cards vs opponent's random 2 cards,
with 6 board cards dealt (matching Toss'Em structure).

Output: two_card_equity.pkl
"""

import pkrbot
import pickle
import time

RANK_ORDER = '23456789TJQKA'
RANKS = list(RANK_ORDER)
SUITS = ['s', 'h', 'd', 'c']


def get_2card_key(card1, card2):
    """
    Convert two cards to canonical key: 'AA', 'AKs', 'AKo', etc.
    """
    r1 = str(card1)[0]
    r2 = str(card2)[0]
    s1 = str(card1)[1]
    s2 = str(card2)[1]
    
    v1 = RANK_ORDER.index(r1)
    v2 = RANK_ORDER.index(r2)
    
    # Order by rank (higher first)
    if v1 < v2:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    
    # Pairs don't have suited/offsuit designation
    if r1 == r2:
        return f"{r1}{r2}"
    
    suited = 's' if s1 == s2 else 'o'
    return f"{r1}{r2}{suited}"


def compute_2card_equity(card1, card2, sims=20000):
    """
    Compute equity for 2-card hand vs random 2-card hand.
    
    Simulates Toss'Em structure:
    - You have 2 cards
    - Opponent has 2 cards (from remaining deck)
    - 6 board cards dealt
    - Best 5-card hand wins
    """
    hole = [card1, card2]
    
    deck = pkrbot.Deck()
    for c in hole:
        if c in deck.cards:
            deck.cards.remove(c)
    
    wins = 0
    ties = 0
    
    for _ in range(sims):
        deck.shuffle()
        
        # Opponent gets 2 cards, board gets 6 cards
        draw = deck.peek(8)
        opp = draw[:2]
        board = draw[2:8]
        
        my_val = pkrbot.evaluate(hole + board)
        opp_val = pkrbot.evaluate(opp + board)
        
        if my_val > opp_val:
            wins += 1
        elif my_val == opp_val:
            ties += 1
    
    return (wins + 0.5 * ties) / sims


def generate_2card_equity_table(output_file='two_card_equity.pkl', 
                                 txt_file='two_card_equity.txt',
                                 sims_per_hand=20000):
    """
    Generate equity table for all 169 unique 2-card hand classes.
    """
    print("="*60)
    print("GENERATING 2-CARD EQUITY TABLE")
    print("="*60)
    print(f"\nSimulations per hand: {sims_per_hand}")
    print("Total unique hands: 169 (13 pairs + 78 suited + 78 offsuit)\n")
    
    equity_table = {}
    
    start_time = time.time()
    count = 0
    
    # Generate one example of each hand class
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i > j:  # Only upper triangle (r1 >= r2 by rank)
                continue
            
            # Determine hand type
            if r1 == r2:
                # Pair - just need one example
                card1 = pkrbot.Card(f"{r1}s")
                card2 = pkrbot.Card(f"{r2}h")
                key = f"{r1}{r2}"
                
                eq = compute_2card_equity(card1, card2, sims=sims_per_hand)
                equity_table[key] = eq
                count += 1
                
                if count % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Computed {count}/169 hands... ({elapsed:.1f}s)")
            else:
                # Non-pair - need suited and offsuit
                # Suited
                card1 = pkrbot.Card(f"{r2}s")  # r2 is higher (we iterated i <= j)
                card2 = pkrbot.Card(f"{r1}s")
                key_s = get_2card_key(card1, card2)
                
                eq_s = compute_2card_equity(card1, card2, sims=sims_per_hand)
                equity_table[key_s] = eq_s
                count += 1
                
                # Offsuit
                card1 = pkrbot.Card(f"{r2}s")
                card2 = pkrbot.Card(f"{r1}h")
                key_o = get_2card_key(card1, card2)
                
                eq_o = compute_2card_equity(card1, card2, sims=sims_per_hand)
                equity_table[key_o] = eq_o
                count += 1
                
                if count % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Computed {count}/169 hands... ({elapsed:.1f}s)")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Total hands: {len(equity_table)}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Save pickle
    with open(output_file, 'wb') as f:
        pickle.dump({
            'equity_table': equity_table,
            'sims_per_hand': sims_per_hand,
            'generation_time': elapsed,
        }, f)
    print(f"\nSaved to: {output_file}")
    
    # Write text file
    write_2card_txt(equity_table, txt_file, sims_per_hand)
    print(f"Saved to: {txt_file}")
    
    return equity_table


def write_2card_txt(equity_table, txt_file, sims_per_hand):
    """Write equity table to human-readable text file."""
    
    sorted_hands = sorted(equity_table.items(), key=lambda x: x[1], reverse=True)
    
    with open(txt_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("2-CARD EQUITY TABLE FOR TOSS'EM HOLD'EM\n")
        f.write("="*60 + "\n\n")
        f.write(f"Simulations per hand: {sims_per_hand}\n")
        f.write(f"Total unique hands: {len(equity_table)}\n\n")
        f.write("Equity = Win% + (Tie%/2) vs random 2-card hand\n\n")
        
        # Full ranking
        f.write("-"*40 + "\n")
        f.write(f"{'Rank':<6} {'Hand':<8} {'Equity':<10}\n")
        f.write("-"*40 + "\n")
        
        for i, (hand, equity) in enumerate(sorted_hands, 1):
            f.write(f"{i:<6} {hand:<8} {equity:.4f}\n")
        
        # By category
        f.write("\n" + "="*60 + "\n")
        f.write("BY CATEGORY\n")
        f.write("="*60 + "\n")
        
        # Pairs
        f.write("\nPAIRS:\n")
        pairs = [(h, e) for h, e in sorted_hands if len(h) == 2]
        for hand, equity in pairs:
            f.write(f"  {hand}: {equity:.4f}\n")
        
        # Suited
        f.write("\nSUITED (top 20):\n")
        suited = [(h, e) for h, e in sorted_hands if h.endswith('s')]
        for hand, equity in suited[:20]:
            f.write(f"  {hand}: {equity:.4f}\n")
        
        # Offsuit
        f.write("\nOFFSUIT (top 20):\n")
        offsuit = [(h, e) for h, e in sorted_hands if h.endswith('o')]
        for hand, equity in offsuit[:20]:
            f.write(f"  {hand}: {equity:.4f}\n")
        
        # Stats
        f.write("\n" + "="*60 + "\n")
        f.write("STATISTICS\n")
        f.write("="*60 + "\n\n")
        
        equities = list(equity_table.values())
        f.write(f"Min equity: {min(equities):.4f}\n")
        f.write(f"Max equity: {max(equities):.4f}\n")
        f.write(f"Avg equity: {sum(equities)/len(equities):.4f}\n")
        
        # Category averages
        pairs_eq = [e for h, e in equity_table.items() if len(h) == 2]
        suited_eq = [e for h, e in equity_table.items() if h.endswith('s')]
        offsuit_eq = [e for h, e in equity_table.items() if h.endswith('o')]
        
        f.write(f"\nAverage by category:\n")
        f.write(f"  Pairs:   {sum(pairs_eq)/len(pairs_eq):.4f}\n")
        f.write(f"  Suited:  {sum(suited_eq)/len(suited_eq):.4f}\n")
        f.write(f"  Offsuit: {sum(offsuit_eq)/len(offsuit_eq):.4f}\n")


if __name__ == "__main__":
    print("Starting 2-card equity generation...")
    print("This will take ~5-10 minutes with 20k sims per hand\n")
    
    generate_2card_equity_table(sims_per_hand=50000)
    
    print("\n" + "="*60)
    print("DONE! Now run step2_three_card_preflop.py")
    print("="*60)