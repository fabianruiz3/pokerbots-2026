"""
Pre-compute preflop 3-card equity table for instant lookups.
This eliminates all preflop MC simulations, saving massive clock time.
"""

import pkrbot
import pickle
from collections import defaultdict
import time

def normalize_hand(cards):
    """
    Normalize a 3-card hand to its canonical form.
    
    Returns a tuple representing the hand class:
    - Ranks in descending order
    - Suit pattern (0=offsuit, 1=two suited, 2=three suited)
    
    Examples:
    - Ah Kh Qh → (14,13,12,2) [three suited]
    - Ah Ks Qh → (14,13,12,0) [offsuit]
    - Ah Kh Qd → (14,13,12,1) [two suited, high cards suited]
    """
    rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
                'T':10,'J':11,'Q':12,'K':13,'A':14}
    
    # Extract ranks and suits
    ranks = []
    suits = []
    for card in cards:
        card_str = str(card)
        ranks.append(rank_map[card_str[0]])
        suits.append(card_str[1])
    
    # Sort ranks descending
    ranks.sort(reverse=True)
    
    # Determine suit pattern
    if suits[0] == suits[1] == suits[2]:
        suit_pattern = 2  # Three suited (best flush potential)
    elif suits[0] == suits[1] or suits[1] == suits[2] or suits[0] == suits[2]:
        # Two suited - normalize so highest two cards are suited
        if suits[0] == suits[1]:
            suit_pattern = 1  # Highest two suited
        elif suits[0] == suits[2]:
            suit_pattern = 1  # First and third
        else:
            suit_pattern = 1  # Second and third
    else:
        suit_pattern = 0  # Rainbow (no flush potential)
    
    return (ranks[0], ranks[1], ranks[2], suit_pattern)

def compute_equity(cards, sims=5000):
    """
    Compute equity for a 3-card hand vs random 3-card hand.
    Uses high sim count for accuracy since this is pre-computed.
    """
    hole = [pkrbot.Card(str(c)) for c in cards]
    
    deck = pkrbot.Deck()
    for c in hole:
        if c in deck.cards:
            deck.cards.remove(c)
    
    wins = 0
    ties = 0
    
    for _ in range(sims):
        deck.shuffle()
        draw = deck.peek(9)  # 3 opp cards + 6 board cards
        opp = draw[:3]
        board = draw[3:]
        
        my_val = pkrbot.evaluate(hole + board)
        opp_val = pkrbot.evaluate(opp + board)
        
        if my_val > opp_val:
            wins += 1
        elif my_val == opp_val:
            ties += 1
    
    return (wins + 0.5 * ties) / sims

def generate_preflop_table(output_file='preflop_equity_table.pkl', sims_per_hand=5000):
    """
    Generate complete preflop equity table.
    
    This takes ~30-60 minutes to run but only needs to be done ONCE.
    The resulting table can be loaded instantly during games.
    """
    print("Generating preflop equity table...")
    print(f"Using {sims_per_hand} simulations per unique hand class")
    print("This will take 30-60 minutes but only needs to be done once!\n")
    
    equity_table = {}
    hand_to_class = {}  # Maps actual hands to their normalized class
    
    # Generate all possible 3-card hands
    deck = pkrbot.Deck()
    all_cards = list(deck.cards)
    
    total_hands = 0
    unique_classes = set()
    
    start_time = time.time()
    
    # Iterate through all C(52,3) = 22,100 combinations
    for i in range(len(all_cards)):
        for j in range(i+1, len(all_cards)):
            for k in range(j+1, len(all_cards)):
                hand = [all_cards[i], all_cards[j], all_cards[k]]
                hand_class = normalize_hand(hand)
                
                # Store mapping from actual hand to class
                hand_key = tuple(sorted([str(c) for c in hand]))
                hand_to_class[hand_key] = hand_class
                
                # Compute equity for new hand classes only
                if hand_class not in equity_table:
                    eq = compute_equity(hand, sims=sims_per_hand)
                    equity_table[hand_class] = eq
                    unique_classes.add(hand_class)
                    
                    # Progress update every 50 new classes
                    if len(unique_classes) % 50 == 0:
                        elapsed = time.time() - start_time
                        print(f"Computed {len(unique_classes)} unique classes... "
                              f"({elapsed:.1f}s elapsed)")
                
                total_hands += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total 3-card hands: {total_hands:,}")
    print(f"Unique hand classes: {len(unique_classes):,}")
    print(f"Reduction ratio: {total_hands / len(unique_classes):.1f}x")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Table size: ~{len(equity_table) * 32 / 1024:.1f} KB")
    
    # Save both tables
    table_data = {
        'equity_table': equity_table,
        'hand_to_class': hand_to_class,
        'sims_per_hand': sims_per_hand,
        'generation_time': elapsed
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(table_data, f)
    
    print(f"\nSaved to: {output_file}")
    print(f"\nUsage in your bot:")
    print(f"  1. Load table in __init__: self.preflop_table = pickle.load(...)")
    print(f"  2. Lookup equity: eq = self.preflop_table[normalize_hand(hole)]")
    print(f"  3. INSTANT results - no MC needed!")
    
    return table_data

def test_lookup_speed(table_file='preflop_equity_table.pkl', num_lookups=10000):
    """
    Test how fast lookups are compared to MC simulation.
    """
    print("Testing lookup speed...")
    
    with open(table_file, 'rb') as f:
        table_data = pickle.load(f)
    
    equity_table = table_data['equity_table']
    
    # Generate random hands for testing
    import random
    deck = pkrbot.Deck()
    test_hands = []
    for _ in range(num_lookups):
        deck.shuffle()
        cards = deck.peek(3)
        test_hands.append(cards)
    
    # Time table lookups
    start = time.time()
    for hand in test_hands:
        hand_class = normalize_hand(hand)
        eq = equity_table[hand_class]
    lookup_time = time.time() - start
    
    # Time MC simulations (100 sims - minimal)
    start = time.time()
    for hand in test_hands[:100]:  # Only 100 to avoid waiting forever
        eq = compute_equity(hand, sims=100)
    mc_time = (time.time() - start) * 100  # Scale to 10k
    
    print(f"\nResults for {num_lookups:,} lookups:")
    print(f"Table lookup:  {lookup_time*1000:.1f}ms ({lookup_time/num_lookups*1e6:.2f}µs per lookup)")
    print(f"MC simulation: {mc_time*1000:.1f}ms ({mc_time/num_lookups*1e6:.2f}µs per lookup)")
    print(f"Speedup: {mc_time/lookup_time:.0f}x faster!")
    print(f"\nTable lookups are essentially FREE compared to MC!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test existing table
        test_lookup_speed()
    else:
        # Generate new table
        print("Starting preflop table generation...")
        print("WARNING: This will take 30-60 minutes!")
        print("Press Ctrl+C within 5 seconds to cancel...\n")
        
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nCancelled!")
            sys.exit(0)
        
        generate_preflop_table()
        
        print("\n" + "="*60)
        print("Now testing lookup speed...")
        print("="*60 + "\n")
        test_lookup_speed()