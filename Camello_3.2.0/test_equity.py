"""
Test POST-DISCARD equity - what actually matters in Toss'Em

The key insight: You ALWAYS discard one card.
So AA9 and AAK both become just "AA" after discarding.

But wait - the discarded card goes to the BOARD!
So the question is:
  - Is it better to put K on the board or 9 on the board?
"""

import pkrbot

def compute_post_discard_equity(keep_cards, discard_card, sims=50000):
    """
    Compute equity when:
    - You keep 2 cards
    - Your discarded card goes to the board
    - Opponent has random 3 cards, discards one randomly
    - 6 total board cards (your discard + their discard + 4 dealt)
    """
    keep = [pkrbot.Card(c) for c in keep_cards]
    discard = pkrbot.Card(discard_card)
    
    deck = pkrbot.Deck()
    for c in keep + [discard]:
        if c in deck.cards:
            deck.cards.remove(c)
    
    wins = 0
    ties = 0
    
    for _ in range(sims):
        deck.shuffle()
        
        # Opponent gets 3 cards
        opp_3 = deck.peek(3)
        
        # Opponent discards one (let's say randomly for now - index 0)
        # In reality opponent makes strategic choice, but random is fair baseline
        import random
        opp_discard_idx = random.randint(0, 2)
        opp_keep = [opp_3[i] for i in range(3) if i != opp_discard_idx]
        opp_discard = opp_3[opp_discard_idx]
        
        # Board = your discard + opponent discard + 4 more cards
        remaining = deck.peek(7)[3:]  # skip opp's 3, take next 4
        board = [discard, opp_discard] + remaining
        
        # Evaluate best 5-card hand from 2 hole + 6 board
        my_val = pkrbot.evaluate(keep + board)
        opp_val = pkrbot.evaluate(opp_keep + board)
        
        if my_val > opp_val:
            wins += 1
        elif my_val == opp_val:
            ties += 1
    
    return (wins + 0.5 * ties) / sims


def compute_preflop_3card_equity(cards, sims=50000):
    """Original preflop equity - 3 cards vs 3 cards, full board runout."""
    hole = [pkrbot.Card(c) for c in cards]
    
    deck = pkrbot.Deck()
    for c in hole:
        if c in deck.cards:
            deck.cards.remove(c)
    
    wins = 0
    ties = 0
    
    for _ in range(sims):
        deck.shuffle()
        draw = deck.peek(9)
        opp = draw[:3]
        board = draw[3:]
        
        my_val = pkrbot.evaluate(hole + board)
        opp_val = pkrbot.evaluate(opp + board)
        
        if my_val > opp_val:
            wins += 1
        elif my_val == opp_val:
            ties += 1
    
    return (wins + 0.5 * ties) / sims


if __name__ == "__main__":
    print("="*70)
    print("COMPARING PREFLOP vs POST-DISCARD EQUITY")
    print("="*70)
    
    print("\n** PREFLOP 3-CARD EQUITY (before discard) **\n")
    
    hands_3card = [
        (['As', 'Ah', 'Ks'], "AAK (AK suited)"),
        (['As', 'Ah', 'Kc'], "AAK (rainbow)"),
        (['As', 'Ah', '9s'], "AA9 (A9 suited)"),
        (['As', 'Ah', '9c'], "AA9 (rainbow)"),
    ]
    
    for cards, desc in hands_3card:
        eq = compute_preflop_3card_equity(cards, sims=30000)
        print(f"  {desc}: {eq:.4f}")
    
    print("\n" + "="*70)
    print("** POST-DISCARD EQUITY (what actually matters!) **")
    print("This simulates: you keep AA, discard the third card to board")
    print("="*70 + "\n")
    
    # The key comparison: discarding K vs discarding 9
    discard_scenarios = [
        (['As', 'Ah'], 'Ks', "Keep AA, discard Ks (K goes to board)"),
        (['As', 'Ah'], 'Kc', "Keep AA, discard Kc (K goes to board)"),
        (['As', 'Ah'], '9s', "Keep AA, discard 9s (9 goes to board)"),
        (['As', 'Ah'], '9c', "Keep AA, discard 9c (9 goes to board)"),
    ]
    
    print("Scenario: You have AA + one card, opponent has random 3 cards\n")
    
    results = []
    for keep, discard, desc in discard_scenarios:
        eq = compute_post_discard_equity(keep, discard, sims=30000)
        results.append((eq, desc))
        print(f"  {desc}: {eq:.4f}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The POST-DISCARD equity is what matters for actual play!

Key insight: When you discard a card, it goes to the board and
BOTH players can use it. So:

- Discarding K: Puts a high card on board that helps opponent too
- Discarding 9: Puts a low card on board that's less useful

But YOU still have AA regardless. The question is whether having
K on the board hurts you (opponent can use it) or helps you 
(broadway straight potential with your Ace).
""")
    
    print("\nSorted by post-discard equity:")
    results.sort(reverse=True)
    for eq, desc in results:
        print(f"  {eq:.4f}  {desc}")