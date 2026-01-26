"""
Action abstraction constants for CFR strategy - matches C++ tossem_abs namespace.
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

# Street constants - matches C++ tossem_abs
STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_BB_DISCARD = 2
STREET_SB_DISCARD = 3
STREET_TURN = 4
STREET_RIVER = 5

# Stack/pot constants
STARTING_STACK = 400
SMALL_BLIND = 1
BIG_BLIND = 2


def board_len_to_street(board_len, bb_discarded, sb_discarded):
    """
    Convert board length and discard state to street number.
    
    Board progression in Toss'em:
    - Preflop: board_len = 0
    - Flop: board_len = 2 (after 2-card flop dealt)
    - BB Discard: board_len = 2, waiting for BB to discard
    - SB Discard: board_len = 3 (BB's card added to board)
    - Turn: board_len = 5 (SB's card + turn card added)
    - River: board_len = 6
    """
    if board_len == 0:
        return STREET_PREFLOP
    elif board_len == 2:
        # Flop is dealt, but discards haven't happened yet OR we're in flop betting
        if not bb_discarded:
            return STREET_FLOP  # Still on flop betting or BB about to discard
        else:
            return STREET_FLOP  # Flop betting after would be weird, but default to flop
    elif board_len == 3:
        # BB has discarded their card to the board
        return STREET_SB_DISCARD if bb_discarded and not sb_discarded else STREET_FLOP
    elif board_len == 4:
        # This shouldn't happen in normal Toss'em flow
        return STREET_TURN
    elif board_len == 5:
        # SB discarded + turn card dealt
        return STREET_TURN
    elif board_len >= 6:
        return STREET_RIVER
    else:
        return STREET_FLOP


def get_hole_bucket_3card(hole_cards):
    """
    Compute hole bucket for 3-card hand - matches C++ hole_bucket_3card.
    
    Args:
        hole_cards: list of card ints (rank*4 + suit format) or strings
    
    Returns:
        bucket index 0-59
    """
    # Convert to rank*4+suit format if needed
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
    
    # Bucket into 60 bins
    bucket = strength // 4
    return max(0, min(59, bucket))


def get_hole_bucket_2card(hole_cards):
    """
    Compute hole bucket for 2-card hand - matches C++ hole_bucket_2card.
    
    Args:
        hole_cards: list of 2 card ints or strings
    
    Returns:
        bucket index
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


def get_board_bucket(board_cards):
    """
    Compute board bucket - matches C++ board_bucket.
    
    Args:
        board_cards: list of card ints or strings
    
    Returns:
        bucket index 0-79
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
    
    # Broadway count (rank >= 8, which is T in 0-12 scale)
    broadway = sum(1 for r in ranks if r >= 8)
    
    # Feature vector: [max_rank_count, max_suit_count, straight_potential, high_card, broadway]
    bucket = 0
    bucket += (max_rank_count - 1) * 20
    bucket += (max_suit_count - 1) * 8
    bucket += max(0, straight_potential - 2) * 4
    bucket += broadway * 2
    bucket += high_card // 2
    
    return max(0, min(79, bucket))


def get_pot_bucket(pot):
    """Matches C++ pot_bucket."""
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


def get_stack_bucket(eff_stack):
    """Matches C++ stack_bucket."""
    if eff_stack <= 50:
        return 0
    if eff_stack <= 120:
        return 1
    if eff_stack <= 220:
        return 2
    if eff_stack <= 320:
        return 3
    if eff_stack <= 400:
        return 4
    return 5


def get_history_bucket(betting_history):
    """
    Matches C++ history_bucket.
    
    Args:
        betting_history: list of (player, action_id) tuples
    """
    L = len(betting_history)
    if L == 0:
        return 0
    if L <= 2:
        a = betting_history[-1][1]
        return min(3, a) + 1
    
    raises = sum(1 for _, a in betting_history if a >= RAISE_SMALL)
    
    if L <= 4:
        return 4 + min(3, raises)
    
    if raises == 0:
        return 8
    if raises == 1:
        return 9
    if raises == 2:
        return 10
    return 11


def card_str_to_int(card_str):
    """
    Convert card string like 'Ah' to int format (rank*4 + suit).
    
    Rank: 2=0, 3=1, ..., T=8, J=9, Q=10, K=11, A=12
    Suit: c=0, d=1, h=2, s=3 (or any consistent mapping)
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
