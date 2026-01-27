"""
Action abstraction constants for CFR strategy - matches C++ tossem_abs namespace.
VERSION 3: 7-street system matching game engine
"""

# Betting action IDs (0..3) - matches tossem_abs
FOLD = 0
CHECK_CALL = 1
RAISE_SMALL = 2   # ~0.55x pot
RAISE_LARGE = 3   # ~1x pot
NUM_ACTIONS = 4

# Discard action IDs (4..6)
DISCARD0 = 4
DISCARD1 = 5
DISCARD2 = 6
NUM_DISCARD_ACTIONS = 3
NUM_DISTINCT_ACTIONS = NUM_ACTIONS + NUM_DISCARD_ACTIONS  # 7

# Street constants - 7-street system matching game engine
# Street 0: Preflop betting
# Street 1: Flop dealt (no actions - skip in CFR)
# Street 2: BB discard (or forced CheckAction if you're SB)
# Street 3: SB discard (or forced CheckAction if you're BB)
# Street 4: Flop betting (post-discards)
# Street 5: Turn betting
# Street 6: River betting
STREET_PREFLOP = 0
STREET_FLOP_DEAL = 1     # Flop dealt, no player actions
STREET_BB_DISCARD = 2
STREET_SB_DISCARD = 3
STREET_FLOP_BET = 4      # Flop betting (after discards)
STREET_TURN = 5
STREET_RIVER = 6
NUM_STREETS = 7

# Stack/pot constants
STARTING_STACK = 400
SMALL_BLIND = 1
BIG_BLIND = 2


def engine_street_to_cfr_street(engine_street, is_bb, bb_discarded, sb_discarded):
    """
    Convert game engine street number to CFR street.
    
    The game engine uses streets 0-6 directly:
    - 0: Preflop betting
    - 1: (Flop dealt - no actions)
    - 2: First discard phase (BB discards if you're BB, else CheckAction)
    - 3: Second discard phase (SB discards if you're SB, else CheckAction)
    - 4: Flop betting (post-discards)
    - 5: Turn betting
    - 6: River betting
    
    For CFR lookup, we use the same numbering.
    """
    return engine_street


def get_hole_bucket_3card(hole_cards):
    """
    Compute hole bucket for 3-card hand - matches C++ get_hole_bucket.
    Returns bucket 0-39 (40 buckets total).
    """
    cards = []
    for c in hole_cards:
        if isinstance(c, int):
            cards.append(c)
        else:
            cards.append(card_str_to_int(str(c)))
    
    ranks = sorted([c // 4 for c in cards], reverse=True)
    suits = [c % 4 for c in cards]
    
    a, b, c = ranks[0], ranks[1], ranks[2]
    
    # Trips/pair detection
    trips = (a == b == c)
    pair = (a == b) or (b == c) or (a == c)
    
    # Flush count
    suit_cnt = [0, 0, 0, 0]
    for s in suits:
        suit_cnt[s] += 1
    flush_count = max(suit_cnt)
    
    # Straight potential
    uniq = sorted(set(ranks), reverse=True)
    straight_potential = 0
    if len(uniq) >= 2:
        for i in range(len(uniq) - 1):
            if uniq[i] - uniq[i+1] <= 2:
                straight_potential += 1
    
    # Strength calculation (matches C++)
    strength = a * 2 + b + c
    if trips:
        strength += 30
    elif pair:
        strength += 15
    strength += (flush_count - 1) * 8
    strength += straight_potential * 5
    
    # Bucket into 40 bins
    bucket = strength // 6
    return max(0, min(39, bucket))


def get_hole_bucket_2card(hole_cards):
    """
    Compute hole bucket for 2-card hand - matches C++ get_hole_bucket_2card.
    """
    cards = []
    for c in hole_cards:
        if isinstance(c, int):
            cards.append(c)
        else:
            cards.append(card_str_to_int(str(c)))
    
    r0 = cards[0] // 4
    r1 = cards[1] // 4
    hi = max(r0, r1)
    lo = min(r0, r1)
    suited = (cards[0] % 4) == (cards[1] % 4)
    
    # Pairs: buckets 0-12
    if hi == lo:
        return hi
    
    # Non-pairs
    base = 13 + (hi * (hi - 1)) // 2 + lo
    if suited:
        base += 78
    return base


def get_hole_bucket(hole_cards):
    """Get hole bucket for 2 or 3 card hand."""
    if len(hole_cards) == 2:
        return get_hole_bucket_2card(hole_cards)
    return get_hole_bucket_3card(hole_cards)


def get_board_bucket(board_cards):
    """
    Compute board bucket - matches C++ get_board_bucket.
    Returns bucket 0-24 (25 buckets total).
    """
    if not board_cards:
        return 0
    
    cards = []
    for c in board_cards:
        if isinstance(c, int):
            cards.append(c)
        else:
            cards.append(card_str_to_int(str(c)))
    
    ranks = [c // 4 for c in cards]
    suits = [c % 4 for c in cards]
    
    # Rank counts
    rc = [0] * 13
    for r in ranks:
        rc[r] += 1
    max_rank_count = max(rc)
    
    # Suit counts
    sc = [0] * 4
    for s in suits:
        sc[s] += 1
    max_suit_count = max(sc)
    
    # Straight potential
    uniq = sorted(set(ranks))
    straight_potential = 0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if uniq[j] - uniq[i] <= 4:
                straight_potential = max(straight_potential, j - i + 1)
    
    high_card = max(ranks)
    
    # Simplified features
    paired = 1 if max_rank_count >= 2 else 0
    flush_draw = min(2, max_suit_count - 1)
    straight_draw = min(2, max(0, straight_potential - 2))
    high = 1 if high_card >= 10 else 0  # T=8, J=9, Q=10, K=11, A=12

    # Combine into ~25 buckets
    bucket = paired * 12 + flush_draw * 4 + straight_draw * 2 + high
    return min(24, bucket)


def get_pot_bucket(pot):
    """Matches C++ get_pot_bucket. Returns 0-5."""
    if pot <= 4:
        return 0
    if pot <= 10:
        return 1
    if pot <= 25:
        return 2
    if pot <= 60:
        return 3
    if pot <= 140:
        return 4
    return 5


def get_history_bucket(betting_history):
    """
    Matches C++ get_history_bucket. Returns 0-5.
    """
    if not betting_history:
        return 0
    
    raises = 0
    large_raises = 0
    for _, a in betting_history:
        if a == RAISE_SMALL:
            raises += 1
        elif a == RAISE_LARGE:
            raises += 1
            large_raises += 1
    
    if raises == 0:
        return 1  # passive
    if raises == 1 and large_raises == 0:
        return 2  # one small raise
    if raises == 1 and large_raises == 1:
        return 3  # one large raise
    if raises == 2:
        return 4  # two raises
    return 5  # very aggressive


def card_str_to_int(card_str):
    """
    Convert card string like 'Ah' to int format (rank*4 + suit).
    
    Rank: 2=0, 3=1, ..., T=8, J=9, Q=10, K=11, A=12
    Suit: c=0, d=1, h=2, s=3
    """
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    
    r = rank_map.get(card_str[0].upper(), 0)
    s = suit_map.get(card_str[1].lower(), 0)
    return r * 4 + s


def compute_legal_mask(legal_actions):
    """Compute bitmask of legal actions."""
    mask = 0
    for a in legal_actions:
        if 0 <= a < NUM_DISTINCT_ACTIONS:
            mask |= (1 << a)
    return mask
