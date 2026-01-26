"""
STEP 2: Generate 3-Card Preflop Score Table (PURE 3-CARD MC)

SIMPLIFIED APPROACH:
- Score = raw 3-card Monte Carlo equity
- That's it. No 2-card lookups added to the score.

The MC simulation already has both players discarding optimally,
so it IS the true preflop equity. Adding 2-card lookups double-counts.

We still store 2-card info for reference (what to discard, flexibility analysis)
but it doesn't affect the score.

Output: preflop_scores.pkl, preflop_scores.txt
"""

import pkrbot
import pickle
import time
import random

RANK_ORDER = '23456789TJQKA'


def load_2card_equity(filepath='two_card_equity.pkl'):
    """Load the 2-card equity table from Step 1."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['equity_table']


def get_2card_key(card1, card2):
    """Convert two cards to canonical key: 'AA', 'AKs', 'AKo', etc."""
    r1 = str(card1)[0]
    r2 = str(card2)[0]
    s1 = str(card1)[1]
    s2 = str(card2)[1]
    
    v1 = RANK_ORDER.index(r1)
    v2 = RANK_ORDER.index(r2)
    
    if v1 < v2:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    
    if r1 == r2:
        return f"{r1}{r2}"
    
    suited = 's' if s1 == s2 else 'o'
    return f"{r1}{r2}{suited}"


def get_best_2card_hand(cards, equity_2card):
    """
    Given 3 cards, find the best 2-card hand to keep.
    Returns (best_keep, best_discard, best_key, best_equity)
    """
    c1, c2, c3 = cards
    
    options = [
        ((c1, c2), c3),
        ((c1, c3), c2),
        ((c2, c3), c1),
    ]
    
    best_eq = -1
    best_keep = None
    best_discard = None
    best_key = None
    
    for (keep, discard) in options:
        key = get_2card_key(keep[0], keep[1])
        eq = equity_2card.get(key, 0.45)
        if eq > best_eq:
            best_eq = eq
            best_keep = keep
            best_discard = discard
            best_key = key
    
    return best_keep, best_discard, best_key, best_eq


def compute_3card_equity(cards, equity_2card, sims=10000):
    """
    Monte Carlo equity for 3-card hand vs random 3-card hand.
    
    Both players discard optimally (based on 2-card equity lookup).
    Discards go to the board along with 4 more random cards.
    Best 5-card hand from 2 hole + 6 board wins.
    
    THIS IS THE SCORE - no adjustments needed.
    """
    hole = list(cards)
    
    deck = pkrbot.Deck()
    for c in hole:
        if c in deck.cards:
            deck.cards.remove(c)
    
    wins = 0
    ties = 0
    
    for _ in range(sims):
        deck.shuffle()
        
        remaining = list(deck.cards)
        opp_cards = remaining[:3]
        rest_of_deck = remaining[3:]
        
        # Both players discard optimally
        my_keep, my_discard, _, _ = get_best_2card_hand(hole, equity_2card)
        opp_keep, opp_discard, _, _ = get_best_2card_hand(opp_cards, equity_2card)
        
        # Board = both discards + 4 more cards
        board = [my_discard, opp_discard] + rest_of_deck[:4]
        
        my_val = pkrbot.evaluate(list(my_keep) + board)
        opp_val = pkrbot.evaluate(list(opp_keep) + board)
        
        if my_val > opp_val:
            wins += 1
        elif my_val == opp_val:
            ties += 1
    
    return (wins + 0.5 * ties) / sims


def normalize_3card_hand(cards):
    """Normalize 3-card hand for table lookup."""
    rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
                'T':10,'J':11,'Q':12,'K':13,'A':14}
    
    cards_info = []
    for card in cards:
        card_str = str(card)
        cards_info.append((rank_map[card_str[0]], card_str[1]))
    
    cards_info.sort(key=lambda x: x[0], reverse=True)
    
    ranks = [c[0] for c in cards_info]
    suits = [c[1] for c in cards_info]
    
    if suits[0] == suits[1] == suits[2]:
        suit_pattern = 'AAA'
    elif suits[0] == suits[1]:
        suit_pattern = 'AA_'
    elif suits[0] == suits[2]:
        suit_pattern = 'A_A'
    elif suits[1] == suits[2]:
        suit_pattern = '_AA'
    else:
        suit_pattern = '___'
    
    return (ranks[0], ranks[1], ranks[2], suit_pattern)


def evaluate_3card_hand(cards, equity_2card, sims=10000):
    """
    Evaluate a 3-card hand.
    
    SCORE = raw 3-card MC equity (PURE, no adjustments)
    
    2-card info stored for reference only.
    """
    c1, c2, c3 = cards
    
    # === THE SCORE: Pure 3-card Monte Carlo ===
    raw_3card_equity = compute_3card_equity(cards, equity_2card, sims=sims)
    
    # === 2-card info (for reference/debugging only) ===
    options = [
        ((c1, c2), c3),
        ((c1, c3), c2),
        ((c2, c3), c1),
    ]
    
    two_card_data = []
    for (keep, discard) in options:
        key = get_2card_key(keep[0], keep[1])
        eq = equity_2card.get(key, 0.45)
        two_card_data.append({
            'keep': keep,
            'discard': discard,
            'key': key,
            'equity': eq
        })
    
    two_card_data.sort(key=lambda x: x['equity'], reverse=True)
    
    best_2card = two_card_data[0]
    second_2card = two_card_data[1]
    third_2card = two_card_data[2]
    
    # Flexibility: how close are the discard options?
    # High flexibility = multiple similar options (like AKQ)
    # Low flexibility = one clear best (like AA2)
    flexibility = 1.0 - (best_2card['equity'] - second_2card['equity'])
    flexibility = max(0.0, min(1.0, flexibility))
    
    return {
        'cards': [str(c) for c in cards],
        # THE SCORE - pure 3-card equity
        'preflop_score': raw_3card_equity,
        'raw_3card_equity': raw_3card_equity,
        # 2-card reference info
        'best_keep': [str(c) for c in best_2card['keep']],
        'best_discard': str(best_2card['discard']),
        'best_2card_key': best_2card['key'],
        'best_2card_equity': best_2card['equity'],
        'second_2card_key': second_2card['key'],
        'second_2card_equity': second_2card['equity'],
        'third_2card_key': third_2card['key'],
        'third_2card_equity': third_2card['equity'],
        'flexibility': flexibility,
    }


def generate_3card_preflop_table(equity_2card_file='two_card_equity.pkl',
                                  output_file='preflop_scores.pkl',
                                  txt_file='preflop_scores.txt',
                                  sims_per_hand=10000):
    """Generate preflop score table for all 3-card hand classes."""
    print("="*70)
    print("GENERATING 3-CARD PREFLOP TABLE (PURE MC)")
    print("="*70)
    print("\nScore = raw 3-card Monte Carlo equity")
    print("(Both players discard optimally in simulation)\n")
    
    print(f"Loading 2-card equity from {equity_2card_file}...")
    equity_2card = load_2card_equity(equity_2card_file)
    print(f"Loaded {len(equity_2card)} 2-card hand equities")
    
    print(f"\nMonte Carlo simulations per hand: {sims_per_hand}")
    print("Estimated time: 60-120 minutes for ~1900 hand classes\n")
    
    score_table = {}
    
    deck = pkrbot.Deck()
    all_cards = list(deck.cards)
    
    start_time = time.time()
    count = 0
    
    for i in range(len(all_cards)):
        for j in range(i+1, len(all_cards)):
            for k in range(j+1, len(all_cards)):
                hand = [all_cards[i], all_cards[j], all_cards[k]]
                hand_class = normalize_3card_hand(hand)
                
                if hand_class not in score_table:
                    result = evaluate_3card_hand(hand, equity_2card, sims=sims_per_hand)
                    score_table[hand_class] = result
                    count += 1
                    
                    if count % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = count / elapsed if elapsed > 0 else 0
                        remaining = (1911 - count) / rate if rate > 0 else 0
                        print(f"  Computed {count}/~1911 hands... "
                              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"COMPLETE!")
    print(f"{'='*70}")
    print(f"Total 3-card hand classes: {len(score_table)}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    with open(output_file, 'wb') as f:
        pickle.dump({
            'score_table': score_table,
            'sims_per_hand': sims_per_hand,
            'generation_time': elapsed,
            'scoring_method': 'pure_3card_mc',
        }, f)
    print(f"\nSaved to: {output_file}")
    
    write_3card_txt(score_table, txt_file, sims_per_hand)
    print(f"Saved to: {txt_file}")
    
    return score_table


def write_3card_txt(score_table, txt_file, sims_per_hand):
    """Write scores to human-readable text file."""
    
    rank_names = {14:'A', 13:'K', 12:'Q', 11:'J', 10:'T',
                  9:'9', 8:'8', 7:'7', 6:'6', 5:'5', 4:'4', 3:'3', 2:'2'}
    
    sorted_hands = sorted(score_table.items(),
                          key=lambda x: x[1]['preflop_score'],
                          reverse=True)
    
    with open(txt_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("3-CARD PREFLOP SCORES - TOSS'EM HOLD'EM\n")
        f.write("="*100 + "\n\n")
        f.write(f"Simulations per hand: {sims_per_hand}\n\n")
        f.write("Scoring: preflop_score = pure 3-card MC equity\n")
        f.write("         (both players discard optimally in simulation)\n\n")
        f.write(f"Total hand classes: {len(score_table)}\n\n")
        
        # Column headers
        f.write("-"*90 + "\n")
        f.write(f"{'Rank':<6} {'Hand':<6} {'Score':<8} {'Pattern':<8} {'Keep':<6} "
                f"{'Best2c':<8} {'2nd2c':<8} {'Flex':<6}\n")
        f.write("-"*90 + "\n")
        
        for i, (hand_class, data) in enumerate(sorted_hands, 1):
            r1, r2, r3, suit_pattern = hand_class
            hand_str = f"{rank_names[r1]}{rank_names[r2]}{rank_names[r3]}"
            
            f.write(f"{i:<6} {hand_str:<6} {data['preflop_score']:<8.4f} "
                    f"{suit_pattern:<8} {data['best_2card_key']:<6} "
                    f"{data['best_2card_equity']:<8.4f} {data['second_2card_equity']:<8.4f} "
                    f"{data['flexibility']:<6.3f}\n")
        
        # === TOP 50 ===
        f.write("\n" + "="*100 + "\n")
        f.write("TOP 50 HANDS\n")
        f.write("="*100 + "\n\n")
        
        for i, (hand_class, data) in enumerate(sorted_hands[:50], 1):
            r1, r2, r3, suit_pattern = hand_class
            hand_str = f"{rank_names[r1]}{rank_names[r2]}{rank_names[r3]}"
            
            f.write(f"{i:>3}. {hand_str} ({suit_pattern}): {data['preflop_score']:.4f} "
                    f"- keep {data['best_2card_key']}\n")
        
        # === BOTTOM 50 ===
        f.write("\n" + "="*100 + "\n")
        f.write("BOTTOM 50 HANDS\n")
        f.write("="*100 + "\n\n")
        
        for i, (hand_class, data) in enumerate(sorted_hands[-50:], len(sorted_hands)-49):
            r1, r2, r3, suit_pattern = hand_class
            hand_str = f"{rank_names[r1]}{rank_names[r2]}{rank_names[r3]}"
            f.write(f"{i:>4}. {hand_str} ({suit_pattern}): {data['preflop_score']:.4f}\n")
        
        # === TRIPS ===
        f.write("\n" + "="*100 + "\n")
        f.write("TRIPS\n")
        f.write("="*100 + "\n\n")
        
        trips = [(h, d) for h, d in score_table.items() if h[0] == h[1] == h[2]]
        trips.sort(key=lambda x: x[1]['preflop_score'], reverse=True)
        for h, d in trips:
            hand_str = f"{rank_names[h[0]]}{rank_names[h[1]]}{rank_names[h[2]]}"
            f.write(f"  {hand_str}: {d['preflop_score']:.4f}\n")
        
        # === PAIRS ===
        f.write("\n" + "="*100 + "\n")
        f.write("PAIRS (Top 30)\n")
        f.write("="*100 + "\n\n")
        
        pairs = [(h, d) for h, d in score_table.items()
                 if (h[0] == h[1] or h[1] == h[2]) and not (h[0] == h[1] == h[2])]
        pairs.sort(key=lambda x: x[1]['preflop_score'], reverse=True)
        for h, d in pairs[:30]:
            hand_str = f"{rank_names[h[0]]}{rank_names[h[1]]}{rank_names[h[2]]}"
            f.write(f"  {hand_str} ({h[3]}): {d['preflop_score']:.4f} → keep {d['best_2card_key']}\n")
        
        # === THREE SUITED ===
        f.write("\n" + "="*100 + "\n")
        f.write("THREE SUITED (Top 30)\n")
        f.write("="*100 + "\n\n")
        
        three_suited = [(h, d) for h, d in score_table.items() if h[3] == 'AAA']
        three_suited.sort(key=lambda x: x[1]['preflop_score'], reverse=True)
        for h, d in three_suited[:30]:
            hand_str = f"{rank_names[h[0]]}{rank_names[h[1]]}{rank_names[h[2]]}"
            f.write(f"  {hand_str}: {d['preflop_score']:.4f}\n")
        
        # === STATISTICS ===
        f.write("\n" + "="*100 + "\n")
        f.write("STATISTICS\n")
        f.write("="*100 + "\n\n")
        
        scores = [d['preflop_score'] for d in score_table.values()]
        
        f.write(f"Score (3-card MC equity):\n")
        f.write(f"  Min:  {min(scores):.4f}\n")
        f.write(f"  Max:  {max(scores):.4f}\n")
        f.write(f"  Mean: {sum(scores)/len(scores):.4f}\n")
        
        # Percentiles
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        f.write(f"\nPercentiles:\n")
        for p in [10, 25, 50, 75, 90]:
            idx = int(n * p / 100)
            f.write(f"  {p}th: {sorted_scores[idx]:.4f}\n")
        
        # By suit pattern
        f.write("\nAverage by suit pattern:\n")
        for sp in ['AAA', 'AA_', 'A_A', '_AA', '___']:
            sp_scores = [d['preflop_score'] for h, d in score_table.items() if h[3] == sp]
            if sp_scores:
                f.write(f"  {sp}: {sum(sp_scores)/len(sp_scores):.4f} ({len(sp_scores)} hands)\n")


def test_specific_hands(equity_2card_file='two_card_equity.pkl', sims=5000):
    """Test specific hands."""
    
    print("="*70)
    print("TESTING HANDS (Pure 3-Card MC)")
    print("="*70)
    print(f"Sims per hand: {sims}\n")
    
    equity_2card = load_2card_equity(equity_2card_file)
    
    test_hands = [
        ['As', 'Ah', 'Ac'],  # AAA trips
        ['As', 'Ah', 'Ks'],  # AAK suited
        ['As', 'Ah', 'Kc'],  # AAK rainbow
        ['Ks', 'Kh', 'Kc'],  # KKK trips
        ['As', 'Ks', 'Qs'],  # AKQ all suited
        ['As', 'Kh', 'Qc'],  # AKQ rainbow
        ['As', 'Ah', '2c'],  # AA2 (clear discard)
        ['Js', 'Ts', '9s'],  # JT9 suited connected
        ['Jd', 'Tc', '9h'],  # JT9 rainbow
        ['7s', '5h', '2c'],  # Trash
    ]
    
    results = []
    for cards_str in test_hands:
        cards = [pkrbot.Card(c) for c in cards_str]
        result = evaluate_3card_hand(cards, equity_2card, sims=sims)
        results.append((result['preflop_score'], cards_str, result))
        
        print(f"{cards_str}")
        print(f"  SCORE: {result['preflop_score']:.4f}")
        print(f"  Keep:  {result['best_2card_key']} ({result['best_2card_equity']:.3f})")
        print(f"  Flex:  {result['flexibility']:.3f}")
        print()
    
    print("="*70)
    print("RANKED:")
    print("="*70)
    results.sort(reverse=True)
    for i, (score, cards, data) in enumerate(results, 1):
        print(f"{i:>2}. {score:.4f}  {cards}  (keep {data['best_2card_key']})")


if __name__ == "__main__":
    import sys
    import os
    
    if not os.path.exists('two_card_equity.pkl'):
        print("ERROR: two_card_equity.pkl not found!")
        print("Run step1_two_card_equity.py first.")
        sys.exit(1)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_specific_hands(sims=5000)
    else:
        test_specific_hands(sims=5000)
        print("\n")
        generate_3card_preflop_table(sims_per_hand=50000)
        print("\n" + "="*70)
        print("DONE!")
        print("="*70)