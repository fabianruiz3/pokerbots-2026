"""
MC-driven bot for Pokerbots 2026 (3 hole cards, discard twice, 6-card public board).

VERSION 4 FIXES:
- RESPECT OVERBETS: Massive bets = fold unless we have the nuts
- DON'T OVERVALUE WEAK HANDS: Two pair/trips are NOT strong enough to stack off
- CORRECT CRUISE THRESHOLD: 1.5 * remaining rounds (lose 1.5 chips/round avg if always fold)
- SHOVE DETECTION: When opponent shoves, only call with very strong hands
"""

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import NUM_ROUNDS, STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import pkrbot

FINAL_BOARD_CARDS = 6


class Player(Bot):
    def __init__(self):
        # Monte Carlo base simulation counts
        self.base_sims_post = 400
        self.base_sims_discard = 400
        self.base_sims_pre = 500

        self.cruise_mode = False
        
        # Opponent tracking
        self.opponent_fold_count = 0
        self.opponent_overbet_count = 0  # Track how often they overbet
        self.opponent_overbet_showdown_wins = 0  # How often their overbets are value
        self.total_hands = 0
        
        # Load preflop equity table
        import pickle
        import os
        try:
            table_path = os.path.join(os.path.dirname(__file__), 'preflop_scores.pkl')
            with open(table_path, 'rb') as f:
                table_data = pickle.load(f)
            self.preflop_table = table_data.get('score_table', table_data.get('equity_table', {}))
            print(f"[Player] Loaded preflop table: {len(self.preflop_table)} hand classes")
        except Exception as e:
            print(f"[Player] WARNING: Could not load preflop table: {e}")
            self.preflop_table = None

    # ---------- Utility helpers ----------

    def _normalize_hand(self, cards):
        """Normalize a 3-card hand for table lookup."""
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

    def _clock_mult(self, game_clock):
        """Clock multiplier for simulation count."""
        if game_clock < 7.0:
            return 0.10
        elif game_clock < 12.0:
            return 0.30
        elif game_clock < 20.0:
            return 0.50
        elif game_clock < 30.0:
            return 0.70
        elif game_clock < 45.0:
            return 0.90
        else:
            return 1.0

    def _get_board_cards(self, round_state):
        """Return the current public board as a flat list."""
        return list(round_state.board)

    def _to_card_list(self, cards):
        """Safely convert cards to pkrbot.Card objects."""
        out = []
        for c in cards:
            if isinstance(c, pkrbot.Card):
                out.append(c)
            else:
                out.append(pkrbot.Card(str(c)))
        return out

    def _get_street_multiplier(self, board_len):
        """Later streets = more meaningful bets."""
        if board_len == 0:
            return 0.6
        elif board_len <= 2:
            return 1.0
        elif board_len <= 4:
            return 1.3
        else:
            return 1.6

    # ---------- Cruise Control ----------

    def _should_cruise(self, game_state):
        """
        Check if we should enter cruise mode.
        
        If we fold every hand, we lose ~1.5 chips per hand on average
        (posting blinds). So we can cruise if:
        bankroll >= 1.5 * remaining_rounds
        """
        bankroll = game_state.bankroll
        remaining = max(1, NUM_ROUNDS - game_state.round_num)
        # 1.5 chips per round average blind loss
        safety_margin = 1.5 * remaining + 2
        return bankroll >= safety_margin

    def _our_cruise_proximity(self, game_state):
        """How close are WE to cruising?"""
        my_bankroll = game_state.bankroll
        remaining = max(1, NUM_ROUNDS - game_state.round_num)
        cruise_threshold = 1.5 * remaining
        
        if my_bankroll <= 0:
            return {'status': 'BEHIND', 'tightness': 1.0, 'fold_more': False, 'avoid_big_pots': False}
        
        proximity = my_bankroll / max(1, cruise_threshold)
        
        if proximity >= 0.95:
            return {
                'status': 'ALMOST_THERE',
                'tightness': 2.0,
                'fold_more': True,
                'avoid_big_pots': True,
            }
        elif proximity >= 0.80:
            return {
                'status': 'CLOSE',
                'tightness': 1.5,
                'fold_more': True,
                'avoid_big_pots': True,
            }
        elif proximity >= 0.60:
            return {
                'status': 'AHEAD',
                'tightness': 1.2,
                'fold_more': False,
                'avoid_big_pots': False,
            }
        else:
            return {
                'status': 'NORMAL',
                'tightness': 1.0,
                'fold_more': False,
                'avoid_big_pots': False,
            }

    def _opponent_cruise_proximity(self, game_state):
        """How close is OPPONENT to cruising?"""
        my_bankroll = game_state.bankroll
        opp_bankroll = -my_bankroll
        remaining = max(1, NUM_ROUNDS - game_state.round_num)
        cruise_threshold = 1.5 * remaining
        
        if opp_bankroll <= 0:
            return {'status': 'BEHIND', 'urgency': 'NONE', 'aggression': 1.0}
        
        opp_proximity = opp_bankroll / max(1, cruise_threshold)
        
        if opp_proximity >= 0.95:
            return {
                'status': 'CRITICAL',
                'urgency': 'DESPERATE',
                'aggression': 2.0,
                'raise_more': True,
                'shove_threshold': 0.58,
            }
        elif opp_proximity >= 0.80:
            return {
                'status': 'DANGEROUS',
                'urgency': 'HIGH',
                'aggression': 1.6,
                'raise_more': True,
                'shove_threshold': 0.65,
            }
        elif opp_proximity >= 0.60:
            return {
                'status': 'AHEAD',
                'urgency': 'MEDIUM',
                'aggression': 1.3,
                'raise_more': True,
                'shove_threshold': 0.72,
            }
        else:
            return {
                'status': 'NORMAL',
                'urgency': 'NONE',
                'aggression': 1.0,
                'raise_more': False,
                'shove_threshold': 0.80,
            }

    # ---------- BET ANALYSIS ----------

    def _analyze_bet(self, continue_cost, pot, my_stack, opp_stack):
        """
        Analyze the opponent's bet.
        
        Returns classification and recommended response threshold.
        """
        if continue_cost <= 0:
            return {'type': 'NO_BET', 'overbet': False, 'shove': False}
        
        pot_before_bet = pot - continue_cost
        if pot_before_bet <= 0:
            pot_before_bet = 1
        
        bet_to_pot = continue_cost / pot_before_bet
        
        # Is this a shove (or near-shove)?
        is_shove = continue_cost >= my_stack * 0.9 or continue_cost >= opp_stack * 0.9
        
        # Is this an all-in that would put us all-in?
        commits_us = continue_cost >= my_stack * 0.5
        
        if is_shove:
            return {
                'type': 'SHOVE',
                'overbet': True,
                'shove': True,
                'bet_to_pot': bet_to_pot,
                'commits_us': commits_us,
                'min_nuttedness_to_call': 7,  # Need full house+ to call a shove
            }
        elif bet_to_pot > 1.5:
            return {
                'type': 'MASSIVE_OVERBET',
                'overbet': True,
                'shove': False,
                'bet_to_pot': bet_to_pot,
                'commits_us': commits_us,
                'min_nuttedness_to_call': 6,  # Need flush+ to call
            }
        elif bet_to_pot > 1.0:
            return {
                'type': 'OVERBET',
                'overbet': True,
                'shove': False,
                'bet_to_pot': bet_to_pot,
                'commits_us': commits_us,
                'min_nuttedness_to_call': 5,  # Need straight+ to call
            }
        elif bet_to_pot > 0.66:
            return {
                'type': 'LARGE',
                'overbet': False,
                'shove': False,
                'bet_to_pot': bet_to_pot,
                'commits_us': commits_us,
                'min_nuttedness_to_call': 3,  # Need trips+ comfortably
            }
        elif bet_to_pot > 0.33:
            return {
                'type': 'STANDARD',
                'overbet': False,
                'shove': False,
                'bet_to_pot': bet_to_pot,
                'commits_us': commits_us,
                'min_nuttedness_to_call': 0,
            }
        else:
            return {
                'type': 'SMALL',
                'overbet': False,
                'shove': False,
                'bet_to_pot': bet_to_pot,
                'commits_us': commits_us,
                'min_nuttedness_to_call': 0,
            }

    # ---------- Board & Hand Analysis ----------

    def _compute_board_nuttedness(self, board):
        """How many nutted hands are possible on this board."""
        if len(board) < 2:
            return 0.0
        
        board_cards = self._to_card_list(board)
        
        ranks = []
        suits = []
        rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
                    'T':10,'J':11,'Q':12,'K':13,'A':14}
        
        for c in board_cards:
            cs = str(c)
            ranks.append(rank_map[cs[0]])
            suits.append(cs[1])
        
        board_nut_score = 0.0
        
        # Flush possibility
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_suited = max(suit_counts.values()) if suit_counts else 0
        
        if max_suited >= 5:
            board_nut_score += 8.0
        elif max_suited >= 4:
            board_nut_score += 5.0
        elif max_suited >= 3:
            board_nut_score += 2.0
        
        # Straight possibility
        sorted_ranks = sorted(set(ranks))
        max_connected = 1
        current_run = 1
        for i in range(1, len(sorted_ranks)):
            gap = sorted_ranks[i] - sorted_ranks[i-1]
            if gap <= 2:
                current_run += 1
                max_connected = max(max_connected, current_run)
            else:
                current_run = 1
        
        has_wheel_cards = 14 in ranks and any(r <= 5 for r in ranks)
        
        if max_connected >= 5 or (max_connected >= 4 and has_wheel_cards):
            board_nut_score += 6.0
        elif max_connected >= 4:
            board_nut_score += 4.0
        elif max_connected >= 3:
            board_nut_score += 2.0
        
        # Paired board
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        max_of_kind = max(rank_counts.values()) if rank_counts else 0
        num_pairs = sum(1 for c in rank_counts.values() if c >= 2)
        
        if max_of_kind >= 3:
            board_nut_score += 5.0
        elif num_pairs >= 2:
            board_nut_score += 3.0
        elif num_pairs >= 1:
            board_nut_score += 1.0
        
        return board_nut_score

    def _compute_our_nuttedness(self, hole, board):
        """
        How nutted is our hand?
        
        CRITICAL: Two pair and below are WEAK. Don't stack off with them.
        """
        if len(board) < 2 or len(hole) < 2:
            return 0.0
        
        hole_cards = self._to_card_list(hole)
        board_cards = self._to_card_list(board)
        
        all_cards = hole_cards + board_cards
        our_val = pkrbot.evaluate(all_cards)
        our_type = pkrbot.handtype(our_val)
        
        # Nuttedness scoring - BE CONSERVATIVE
        nuttedness_map = {
            'Straight Flush': 12,
            'Quads': 11,
            'Full House': 8,
            'Flush': 6,
            'Straight': 5,
            'Trips': 3,      # MEDIUM - don't stack off
            'Two Pair': 1,   # WEAK - definitely don't stack off
            'Pair': 0,
            'High Card': 0,
        }
        our_nuttedness = nuttedness_map.get(our_type, 0)
        
        # Bonuses for nut versions
        if our_type == 'Flush':
            suit_counts = {}
            for c in board_cards:
                s = str(c)[1]
                suit_counts[s] = suit_counts.get(s, 0) + 1
            
            if suit_counts:
                flush_suit = max(suit_counts.keys(), key=lambda s: suit_counts[s])
                hole_has_ace = any(
                    str(c)[0] == 'A' and str(c)[1] == flush_suit 
                    for c in hole_cards
                )
                if hole_has_ace:
                    our_nuttedness += 3  # Nut flush
        
        elif our_type == 'Full House':
            rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
                        'T':10,'J':11,'Q':12,'K':13,'A':14}
            hole_ranks = [rank_map[str(c)[0]] for c in hole_cards]
            if max(hole_ranks) >= 12:
                our_nuttedness += 2  # High full house
        
        return our_nuttedness

    def _compute_opponent_aggression(self, round_state, active_player):
        """Compute opponent aggression from current state."""
        board = self._get_board_cards(round_state)
        street_mult = self._get_street_multiplier(len(board))
        
        my_stack = round_state.stacks[active_player]
        opp_stack = round_state.stacks[1 - active_player]
        
        my_contrib = STARTING_STACK - my_stack
        opp_contrib = STARTING_STACK - opp_stack
        
        pot = my_contrib + opp_contrib
        
        if pot <= 3:
            return 0.0
        
        opp_share = opp_contrib / max(1, pot)
        aggression_score = 0.0
        
        if opp_share > 0.5:
            aggression_score += (opp_share - 0.5) * 10
        
        my_pip = round_state.pips[active_player]
        opp_pip = round_state.pips[1 - active_player]
        continue_cost = opp_pip - my_pip
        
        if continue_cost > 0:
            pot_before_bet = pot - continue_cost
            bet_fraction = continue_cost / max(1, pot_before_bet)
            
            if bet_fraction > 1.5:
                aggression_score += 6.0
            elif bet_fraction > 1.0:
                aggression_score += 4.0
            elif bet_fraction > 0.66:
                aggression_score += 2.5
            elif bet_fraction > 0.33:
                aggression_score += 1.5
            else:
                aggression_score += 0.5
        
        aggression_score *= street_mult
        
        return aggression_score

    def _compute_total_danger(self, hole, board, round_state, active_player):
        """Total danger score."""
        board_nut = self._compute_board_nuttedness(board)
        our_nut = self._compute_our_nuttedness(hole, board)
        opp_agg = self._compute_opponent_aggression(round_state, active_player)
        
        total_danger = board_nut - our_nut + opp_agg
        
        return {
            'total_danger': total_danger,
            'board_nuttedness': board_nut,
            'our_nuttedness': our_nut,
            'opponent_aggression': opp_agg,
        }

    # ---------- Core Equity Engine ----------

    def _opp_bias_from_action(self, continue_cost, pot, street_n):
        if continue_cost <= 0:
            return 0.0
        frac = continue_cost / max(1.0, pot)
        street_boost = 1.0 + 0.08 * max(0, street_n - 3)
        x = frac * street_boost
        return max(0.0, min(1.0, 1.4 * x))

    def mc_equity(self, round_state, my_hole_cards, sims, opp_bias=0.0):
        raw_board = list(round_state.board)
        raw_hole = list(my_hole_cards)

        board = self._to_card_list(raw_board)
        hole = self._to_card_list(raw_hole)

        opp_hole_n = 3 if (len(hole) == 3 and len(board) < 2) else 2
        remaining_board = max(0, FINAL_BOARD_CARDS - len(board))

        deck = pkrbot.Deck()
        used = hole + board
        for c in used:
            if c in deck.cards:
                deck.cards.remove(c)

        tier = {
            "High Card": 0, "Pair": 1, "Two Pair": 2, "Trips": 3,
            "Straight": 4, "Flush": 5, "Full House": 6, "Quads": 7, "Straight Flush": 8,
        }

        wins = ties = iters = 0

        while iters < sims:
            deck.shuffle()
            draw = deck.peek(opp_hole_n + remaining_board)
            opp = draw[:opp_hole_n]
            runout = draw[opp_hole_n:]

            my_val = pkrbot.evaluate(hole + board + runout)
            opp_val = pkrbot.evaluate(opp + board + runout)

            if opp_bias > 0.0:
                opp_class = pkrbot.handtype(opp_val)
                t = tier.get(opp_class, 0)
                accept_p = min(1.0, max(0.18,
                    1.0 - 0.60 * opp_bias + 0.10 * t + 0.06 * opp_bias * t))
                if random.random() >= accept_p:
                    continue

            if my_val > opp_val:
                wins += 1
            elif my_val == opp_val:
                ties += 1
            iters += 1

        return (wins + 0.5 * ties) / max(1, sims)

    def mc_equity_with_board(self, my_hole_cards, board, sims, opp_bias=0.0):
        board = self._to_card_list(board)
        hole = self._to_card_list(my_hole_cards)

        remaining_board = max(0, FINAL_BOARD_CARDS - len(board))

        deck = pkrbot.Deck()
        for c in hole + board:
            if c in deck.cards:
                deck.cards.remove(c)

        tier = {
            "High Card": 0, "Pair": 1, "Two Pair": 2, "Trips": 3,
            "Straight": 4, "Flush": 5, "Full House": 6, "Quads": 7, "Straight Flush": 8,
        }

        wins = ties = iters = 0

        while iters < sims:
            deck.shuffle()
            draw = deck.peek(2 + remaining_board)
            opp = draw[:2]
            runout = draw[2:]

            my_val = pkrbot.evaluate(hole + board + runout)
            opp_val = pkrbot.evaluate(opp + board + runout)

            if opp_bias > 0.0:
                opp_class = pkrbot.handtype(opp_val)
                t = tier.get(opp_class, 0)
                accept_p = min(1.0, max(0.18,
                    1.0 - 0.60 * opp_bias + 0.10 * t + 0.06 * opp_bias * t))
                if random.random() >= accept_p:
                    continue

            if my_val > opp_val:
                wins += 1
            elif my_val == opp_val:
                ties += 1
            iters += 1

        return (wins + 0.5 * ties) / max(1, sims)

    # ---------- Discard Logic ----------

    def choose_discard_mc(self, game_state, round_state, active_player):
        hole = list(round_state.hands[active_player])
        board = self._get_board_cards(round_state)
        sims = int(self.base_sims_discard * self._clock_mult(game_state.game_clock))

        best_i = 0
        best_ev = -1.0
        
        for i in range(3):
            kept = [hole[j] for j in range(3) if j != i]
            discarded = hole[i]
            temp_board = board + [discarded]
            ev = self.mc_equity_with_board(kept, temp_board, sims=sims, opp_bias=0.0)
            
            if ev > best_ev:
                best_ev = ev
                best_i = i
        
        return best_i

    # ---------- Preflop ----------

    def preflop_action(self, game_state, round_state, active_player):
        legal = round_state.legal_actions()

        my_pip = round_state.pips[active_player]
        opp_pip = round_state.pips[1 - active_player]
        continue_cost = opp_pip - my_pip

        my_stack = round_state.stacks[active_player]
        opp_stack = round_state.stacks[1 - active_player]
        pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)

        hole = list(round_state.hands[active_player])

        our_cruise = self._our_cruise_proximity(game_state)
        opp_cruise = self._opponent_cruise_proximity(game_state)

        # Get equity
        if self.preflop_table:
            hand_class = self._normalize_hand(hole)
            if hand_class in self.preflop_table:
                eq = self.preflop_table[hand_class]['preflop_score']
            else:
                sims = int(self.base_sims_pre * self._clock_mult(game_state.game_clock))
                eq = self.mc_equity(round_state, hole, sims=sims)
        else:
            sims = int(self.base_sims_pre * self._clock_mult(game_state.game_clock))
            eq = self.mc_equity(round_state, hole, sims=sims)

        tightness = our_cruise['tightness']
        aggression = opp_cruise['aggression']

        # Facing a bet
        if continue_cost > 0:
            bet_analysis = self._analyze_bet(continue_cost, pot, my_stack, opp_stack)
            pot_odds = continue_cost / (pot + continue_cost)
            
            # SHOVE/MASSIVE OVERBET preflop - need very strong hand
            if bet_analysis['shove'] or bet_analysis['type'] == 'MASSIVE_OVERBET':
                # Only call with top ~15% of hands
                if eq < 0.58:
                    return FoldAction() if FoldAction in legal else CheckAction()
                print("[Preflop] Calling shove with strong hand", hole, eq)
            
            fold_margin = 0.04 * tightness
            if our_cruise.get('fold_more', False):
                fold_margin += 0.03
            
            if eq < pot_odds + fold_margin:
                return FoldAction() if FoldAction in legal else CheckAction()
            
            # Raise with strong hands
            raise_threshold = 0.72 / aggression
            if RaiseAction in legal and eq >= raise_threshold:
                mn, mx = round_state.raise_bounds()
                mult = 2.5 * aggression
                target = int(max(mn, min(mx, pot * mult)))
                return RaiseAction(target)
            
            return CallAction() if CallAction in legal else CheckAction()

        # No bet facing us
        raise_threshold_high = 0.70 / aggression * tightness
        raise_threshold_med = 0.55 / aggression * tightness

        if RaiseAction in legal and eq >= raise_threshold_high:
            mn, mx = round_state.raise_bounds()
            mult = 3.0 * aggression
            target = int(max(mn, min(mx, pot * mult)))
            return RaiseAction(target)
        
        elif RaiseAction in legal and eq >= raise_threshold_med:
            mn, mx = round_state.raise_bounds()
            mult = 2.2 * aggression
            target = int(max(mn, min(mx, pot * mult)))
            return RaiseAction(target)
        
        elif CheckAction in legal:
            return CheckAction()
        elif CallAction in legal:
            return CallAction()
        
        return FoldAction()

    # ---------- Postflop ----------

    def postflop_action(self, game_state, round_state, active_player):
        legal = round_state.legal_actions()
        board = self._get_board_cards(round_state)
        street_n = len(board)

        my_pip = round_state.pips[active_player]
        opp_pip = round_state.pips[1 - active_player]
        continue_cost = opp_pip - my_pip

        my_stack = round_state.stacks[active_player]
        opp_stack = round_state.stacks[1 - active_player]
        pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)

        hole = list(round_state.hands[active_player])

        our_cruise = self._our_cruise_proximity(game_state)
        opp_cruise = self._opponent_cruise_proximity(game_state)
        
        danger = self._compute_total_danger(hole, board, round_state, active_player)
        our_nuttedness = danger['our_nuttedness']

        sims = int(self.base_sims_post * self._clock_mult(game_state.game_clock))
        opp_bias = self._opp_bias_from_action(continue_cost, pot, street_n)
        equity = self.mc_equity(round_state, hole, sims=sims, opp_bias=opp_bias)

        tightness = our_cruise['tightness']
        aggression = opp_cruise['aggression']

        # =====================
        # FACING A BET
        # =====================
        if continue_cost > 0:
            bet_analysis = self._analyze_bet(continue_cost, pot, my_stack, opp_stack)
            pot_odds = continue_cost / (pot + continue_cost)
            
            # ====== CRITICAL: RESPECT BIG BETS ======
            
            # SHOVE - only call with very strong hands
            if bet_analysis['shove']:
                min_nut = bet_analysis.get('min_nuttedness_to_call', 7)
                if our_nuttedness < min_nut:
                    # Don't call shoves without the goods
                    return FoldAction() if FoldAction in legal else CheckAction()
                # We have a strong hand - call
                return CallAction() if CallAction in legal else CheckAction()
            
            # MASSIVE OVERBET (>150% pot)
            if bet_analysis['type'] == 'MASSIVE_OVERBET':
                min_nut = bet_analysis.get('min_nuttedness_to_call', 6)
                if our_nuttedness < min_nut:
                    return FoldAction() if FoldAction in legal else CheckAction()
            
            # OVERBET (>100% pot)
            if bet_analysis['type'] == 'OVERBET':
                min_nut = bet_analysis.get('min_nuttedness_to_call', 5)
                if our_nuttedness < min_nut:
                    # Need at least a straight to call an overbet
                    return FoldAction() if FoldAction in legal else CheckAction()
            
            # LARGE BET (66-100% pot)
            if bet_analysis['type'] == 'LARGE':
                if our_nuttedness < 3:
                    # Two pair or less - be very careful
                    danger_margin = 0.08
                    if equity < pot_odds + danger_margin:
                        return FoldAction() if FoldAction in legal else CheckAction()
            
            # ====== STANDARD POT ODDS DECISION ======
            
            danger_score = danger['total_danger']
            danger_adjustment = max(0, (danger_score - 3) * 0.02)
            margin = 0.03 * tightness + danger_adjustment
            
            if our_cruise.get('avoid_big_pots', False):
                margin += 0.05
            
            if equity < pot_odds + margin:
                return FoldAction() if FoldAction in legal else CheckAction()
            
            # ====== RAISING ======
            # Only raise for value with strong hands
            
            if our_nuttedness >= 7:  # Full house+
                raise_threshold = 0.50
            elif our_nuttedness >= 5:  # Straight/flush
                raise_threshold = 0.60
            else:
                raise_threshold = 0.75 / aggression
            
            if RaiseAction in legal and equity >= raise_threshold and our_nuttedness >= 5:
                mn, mx = round_state.raise_bounds()
                
                if our_nuttedness >= 8:
                    mult = 3.0 * aggression
                else:
                    mult = 2.5 * aggression
                
                target = int(max(mn, min(mx, pot + mult * continue_cost)))
                return RaiseAction(target)
            
            return CallAction() if CallAction in legal else CheckAction()

        # =====================
        # NO BET FACING US
        # =====================
        if RaiseAction not in legal:
            return CheckAction()

        # Bet threshold
        base_threshold = 0.50 * tightness
        
        board_nut = danger['board_nuttedness']
        
        if board_nut >= 8 and our_nuttedness < 5:
            # Very scary board, we don't have it
            base_threshold += 0.15
        elif board_nut >= 5 and our_nuttedness < 3:
            base_threshold += 0.08
        elif board_nut < 3 and our_nuttedness >= 5:
            base_threshold -= 0.10
        
        if our_cruise.get('avoid_big_pots', False):
            base_threshold += 0.08

        if equity < base_threshold:
            return CheckAction()

        # Bet sizing
        mn, mx = round_state.raise_bounds()
        
        if our_nuttedness >= 8:
            frac = 0.90 * aggression
        elif our_nuttedness >= 5:
            frac = 0.70 * aggression
        elif our_nuttedness >= 3:
            frac = 0.55 * aggression
        else:
            frac = 0.40
        
        amt = int(max(mn, min(mx, frac * pot)))
        return RaiseAction(amt)

    # ---------- Framework Hooks ----------

    def handle_new_round(self, game_state, round_state, active_player):
        self.total_hands += 1

    def handle_round_over(self, game_state, terminal_state, active_player):
        self.cruise_mode = self._should_cruise(game_state)
        
        my_delta = terminal_state.deltas[active_player]
        if my_delta > 0 and my_delta <= 2:
            self.opponent_fold_count += 1

    def get_action(self, game_state, round_state, active_player):
        legal = round_state.legal_actions()

        # Cruise control
        if self.cruise_mode:
            if FoldAction in legal:
                return FoldAction()
            if CheckAction in legal:
                return CheckAction()
            return CallAction()

        # Discard phase
        if DiscardAction in legal:
            idx = self.choose_discard_mc(game_state, round_state, active_player)
            return DiscardAction(idx)

        street_n = len(self._get_board_cards(round_state))

        if street_n == 0:
            return self.preflop_action(game_state, round_state, active_player)

        return self.postflop_action(game_state, round_state, active_player)


if __name__ == "__main__":
    run_bot(Player(), parse_args())