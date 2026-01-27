"""
CFR-driven bot for Pokerbots 2026 (3 hole cards, discard twice, 6-card public board).

Integrates C++ trained CFR strategy with FULL MC bot fallback:
- Uses CFR when lookup succeeds
- Falls back to complete MC bot logic (preflop table, danger analysis, etc.) when CFR misses
- Cruise control for bankroll management
- Safety overrides for extreme opponent bets
"""

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import NUM_ROUNDS, STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import os
import pickle
import pkrbot

from cpp_cfr import CppCFR
from abstraction import (
    FOLD, CHECK_CALL, RAISE_SMALL, RAISE_LARGE, NUM_ACTIONS,
    STREET_PREFLOP, STREET_BB_DISCARD, STREET_SB_DISCARD, STREET_FLOP_BET, STREET_TURN, STREET_RIVER,
    BIG_BLIND
)

FINAL_BOARD_CARDS = 6


class Player(Bot):
    def __init__(self):
        # ==================
        # CFR Strategy Setup
        # ==================
        bin_path = os.path.join(os.path.dirname(__file__), 'cfr_strategy.bin')
        self.cfr = CppCFR(bin_path=bin_path)
        print(f"[Player] CFR nodes loaded: {self.cfr.num_nodes}")
        
        # Betting history for current hand: list of (player, action_id)
        self.betting_history = []
        
        # Track discards
        self.bb_discarded = False
        self.sb_discarded = False
        
        # Debug counters
        self.cfr_hits = 0
        self.cfr_misses = 0
        
        # ==================
        # Monte Carlo Config (fallback)
        # ==================
        self.base_sims_post = 400
        self.base_sims_discard = 400
        self.base_sims_pre = 500

        # ==================
        # Cruise Control
        # ==================
        self.cruise_mode = False
        
        # ==================
        # Opponent Tracking
        # ==================
        self.opponent_fold_count = 0
        self.opponent_overbet_count = 0
        self.opponent_overbet_showdown_wins = 0
        self.total_hands = 0
        
        # ==================
        # Preflop Equity Table
        # ==================
        try:
            table_path = os.path.join(os.path.dirname(__file__), 'preflop_scores.pkl')
            with open(table_path, 'rb') as f:
                table_data = pickle.load(f)
            self.preflop_table = table_data.get('score_table', table_data.get('equity_table', {}))
            print(f"[Player] Loaded preflop table: {len(self.preflop_table)} hand classes")
        except Exception as e:
            print(f"[Player] WARNING: Could not load preflop table: {e}")
            self.preflop_table = None

    # =====================
    # Utility Helpers
    # =====================

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

    def _to_card_list(self, cards):
        """Safely convert cards to pkrbot.Card objects."""
        out = []
        for c in cards:
            if isinstance(c, pkrbot.Card):
                out.append(c)
            else:
                out.append(pkrbot.Card(str(c)))
        return out

    def _to_card_strings(self, cards):
        """Convert cards to string representations."""
        return [str(c) for c in cards]

    def _get_board_cards(self, round_state):
        """Return the current public board as a flat list."""
        return list(round_state.board)

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

    # =====================
    # Cruise Control
    # =====================

    def should_drain_to_one(self, game_state, hero_idx):
        lead = game_state.bankroll
        R = NUM_ROUNDS - game_state.round_num + 1  # remaining hands incl this one

        hero_is_sb = bool(hero_idx)  # usually true in Pokerbots engines

        if hero_is_sb:
            sb_cnt = (R + 1) // 2
            bb_cnt = R // 2
        else:
            bb_cnt = (R + 1) // 2
            sb_cnt = R // 2

        drain_cost = sb_cnt * 1 + bb_cnt * 2
        return lead >= (drain_cost + 1)


    def _should_cruise(self, game_state):
        """Check if we should enter cruise mode."""
        bankroll = game_state.bankroll
        remaining = max(1, NUM_ROUNDS - game_state.round_num)
        safety_margin = 1.5 * remaining + 5
        return bankroll > safety_margin

    def _our_cruise_proximity(self, game_state):
        """How close are WE to cruising?"""
        my_bankroll = game_state.bankroll
        remaining = max(1, NUM_ROUNDS - game_state.round_num)
        cruise_threshold = 1.5 * remaining
        
        if my_bankroll <= 0:
            return {'status': 'BEHIND', 'tightness': 1.0, 'fold_more': False, 'avoid_big_pots': False}
        
        proximity = my_bankroll / max(1, cruise_threshold)
        
        if proximity >= 0.95:
            return {'status': 'ALMOST_THERE', 'tightness': 2.0, 'fold_more': True, 'avoid_big_pots': True}
        elif proximity >= 0.80:
            return {'status': 'CLOSE', 'tightness': 1.5, 'fold_more': True, 'avoid_big_pots': True}
        elif proximity >= 0.60:
            return {'status': 'AHEAD', 'tightness': 1.2, 'fold_more': False, 'avoid_big_pots': False}
        else:
            return {'status': 'NORMAL', 'tightness': 1.0, 'fold_more': False, 'avoid_big_pots': False}

    def _opponent_cruise_proximity(self, game_state):
        """How close is OPPONENT to cruising?"""
        my_bankroll = game_state.bankroll
        opp_bankroll = -my_bankroll
        remaining = max(1, NUM_ROUNDS - game_state.round_num)
        cruise_threshold = 1.5 * remaining
        
        if opp_bankroll <= 0:
            return {'status': 'BEHIND', 'urgency': 'NONE', 'aggression': 1.0, 'raise_more': False, 'shove_threshold': 0.80}
        
        opp_proximity = opp_bankroll / max(1, cruise_threshold)
        
        if opp_proximity >= 0.95:
            return {'status': 'CRITICAL', 'urgency': 'DESPERATE', 'aggression': 2.0, 'raise_more': True, 'shove_threshold': 0.58}
        elif opp_proximity >= 0.80:
            return {'status': 'DANGEROUS', 'urgency': 'HIGH', 'aggression': 1.6, 'raise_more': True, 'shove_threshold': 0.65}
        elif opp_proximity >= 0.60:
            return {'status': 'AHEAD', 'urgency': 'MEDIUM', 'aggression': 1.3, 'raise_more': True, 'shove_threshold': 0.72}
        else:
            return {'status': 'NORMAL', 'urgency': 'NONE', 'aggression': 1.0, 'raise_more': False, 'shove_threshold': 0.80}

    # =====================
    # Bet Analysis
    # =====================

    def _analyze_bet(self, continue_cost, pot, my_stack, opp_stack):
        """Analyze the opponent's bet."""
        if continue_cost <= 0:
            return {'type': 'NO_BET', 'overbet': False, 'shove': False}
        
        pot_before_bet = pot - continue_cost
        if pot_before_bet <= 0:
            pot_before_bet = 1
        
        bet_to_pot = continue_cost / pot_before_bet
        is_shove = continue_cost >= my_stack * 0.9 or continue_cost >= opp_stack * 0.9
        commits_us = continue_cost >= my_stack * 0.5
        
        if is_shove:
            return {'type': 'SHOVE', 'overbet': True, 'shove': True, 'bet_to_pot': bet_to_pot, 'commits_us': commits_us, 'min_nuttedness_to_call': 7}
        elif bet_to_pot > 1.5:
            return {'type': 'MASSIVE_OVERBET', 'overbet': True, 'shove': False, 'bet_to_pot': bet_to_pot, 'commits_us': commits_us, 'min_nuttedness_to_call': 6}
        elif bet_to_pot > 1.0:
            return {'type': 'OVERBET', 'overbet': True, 'shove': False, 'bet_to_pot': bet_to_pot, 'commits_us': commits_us, 'min_nuttedness_to_call': 5}
        elif bet_to_pot > 0.66:
            return {'type': 'LARGE', 'overbet': False, 'shove': False, 'bet_to_pot': bet_to_pot, 'commits_us': commits_us, 'min_nuttedness_to_call': 3}
        elif bet_to_pot > 0.33:
            return {'type': 'STANDARD', 'overbet': False, 'shove': False, 'bet_to_pot': bet_to_pot, 'commits_us': commits_us, 'min_nuttedness_to_call': 0}
        else:
            return {'type': 'SMALL', 'overbet': False, 'shove': False, 'bet_to_pot': bet_to_pot, 'commits_us': commits_us, 'min_nuttedness_to_call': 0}

    # =====================
    # Board & Hand Analysis
    # =====================

    def _compute_board_nuttedness(self, board):
        """How many nutted hands are possible on this board."""
        if len(board) < 2:
            return 0.0
        
        board_cards = self._to_card_list(board)
        ranks = []
        suits = []
        rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
        
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
        """How nutted is our hand?"""
        if len(board) < 2 or len(hole) < 2:
            return 0.0
        
        hole_cards = self._to_card_list(hole)
        board_cards = self._to_card_list(board)
        
        all_cards = hole_cards + board_cards
        our_val = pkrbot.evaluate(all_cards)
        our_type = pkrbot.handtype(our_val)
        
        nuttedness_map = {
            'Straight Flush': 12, 'Quads': 11, 'Full House': 8, 'Flush': 6,
            'Straight': 5, 'Trips': 3, 'Two Pair': 1, 'Pair': 0, 'High Card': 0,
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
                hole_has_ace = any(str(c)[0] == 'A' and str(c)[1] == flush_suit for c in hole_cards)
                if hole_has_ace:
                    our_nuttedness += 3
        elif our_type == 'Full House':
            rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
            hole_ranks = [rank_map[str(c)[0]] for c in hole_cards]
            if max(hole_ranks) >= 12:
                our_nuttedness += 2
        
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
        
        return aggression_score * street_mult

    def _compute_total_danger(self, hole, board, round_state, active_player):
        """Total danger score."""
        board_nut = self._compute_board_nuttedness(board)
        our_nut = self._compute_our_nuttedness(hole, board)
        opp_agg = self._compute_opponent_aggression(round_state, active_player)
        return {
            'total_danger': board_nut - our_nut + opp_agg,
            'board_nuttedness': board_nut,
            'our_nuttedness': our_nut,
            'opponent_aggression': opp_agg,
        }

    # =====================
    # Monte Carlo Equity
    # =====================

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
        for c in hole + board:
            if c in deck.cards:
                deck.cards.remove(c)

        tier = {"High Card": 0, "Pair": 1, "Two Pair": 2, "Trips": 3, "Straight": 4, "Flush": 5, "Full House": 6, "Quads": 7, "Straight Flush": 8}
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
                accept_p = min(1.0, max(0.18, 1.0 - 0.60 * opp_bias + 0.10 * t + 0.06 * opp_bias * t))
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

        tier = {"High Card": 0, "Pair": 1, "Two Pair": 2, "Trips": 3, "Straight": 4, "Flush": 5, "Full House": 6, "Quads": 7, "Straight Flush": 8}
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
                accept_p = min(1.0, max(0.18, 1.0 - 0.60 * opp_bias + 0.10 * t + 0.06 * opp_bias * t))
                if random.random() >= accept_p:
                    continue

            if my_val > opp_val:
                wins += 1
            elif my_val == opp_val:
                ties += 1
            iters += 1

        return (wins + 0.5 * ties) / max(1, sims)

    # =====================
    # Discard Logic
    # =====================

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

    # =====================
    # MC Fallback: Preflop
    # =====================

    def mc_preflop_action(self, game_state, round_state, active_player):
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

        # Get equity - use preflop table if available
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

        if continue_cost > 0:
            bet_analysis = self._analyze_bet(continue_cost, pot, my_stack, opp_stack)
            pot_odds = continue_cost / (pot + continue_cost)
            
            if bet_analysis['shove'] or bet_analysis['type'] == 'MASSIVE_OVERBET':
                if eq < 0.58:
                    return FoldAction() if FoldAction in legal else CheckAction()
            
            fold_margin = 0.04 * tightness
            if our_cruise.get('fold_more', False):
                fold_margin += 0.03
            
            if eq < pot_odds + fold_margin:
                return FoldAction() if FoldAction in legal else CheckAction()
            
            raise_threshold = 0.72 / aggression
            if RaiseAction in legal and eq >= raise_threshold:
                mn, mx = round_state.raise_bounds()
                target = int(max(mn, min(mx, pot * 2.5 * aggression)))
                return RaiseAction(target)
            
            return CallAction() if CallAction in legal else CheckAction()

        raise_threshold_high = 0.70 / aggression * tightness
        raise_threshold_med = 0.55 / aggression * tightness

        if RaiseAction in legal and eq >= raise_threshold_high:
            mn, mx = round_state.raise_bounds()
            target = int(max(mn, min(mx, pot * 3.0 * aggression)))
            return RaiseAction(target)
        elif RaiseAction in legal and eq >= raise_threshold_med:
            mn, mx = round_state.raise_bounds()
            target = int(max(mn, min(mx, pot * 2.2 * aggression)))
            return RaiseAction(target)
        elif CheckAction in legal:
            return CheckAction()
        elif CallAction in legal:
            return CallAction()
        return FoldAction()

    # =====================
    # MC Fallback: Postflop
    # =====================

    def mc_postflop_action(self, game_state, round_state, active_player):
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

        if continue_cost > 0:
            bet_analysis = self._analyze_bet(continue_cost, pot, my_stack, opp_stack)
            pot_odds = continue_cost / (pot + continue_cost)
            
            if bet_analysis['shove']:
                min_nut = bet_analysis.get('min_nuttedness_to_call', 7)
                if our_nuttedness < min_nut:
                    return FoldAction() if FoldAction in legal else CheckAction()
                return CallAction() if CallAction in legal else CheckAction()
            
            if bet_analysis['type'] == 'MASSIVE_OVERBET':
                if our_nuttedness < bet_analysis.get('min_nuttedness_to_call', 6):
                    return FoldAction() if FoldAction in legal else CheckAction()
            
            if bet_analysis['type'] == 'OVERBET':
                if our_nuttedness < bet_analysis.get('min_nuttedness_to_call', 5):
                    return FoldAction() if FoldAction in legal else CheckAction()
            
            if bet_analysis['type'] == 'LARGE' and our_nuttedness < 3:
                if equity < pot_odds + 0.08:
                    return FoldAction() if FoldAction in legal else CheckAction()
            
            danger_score = danger['total_danger']
            margin = 0.03 * tightness + max(0, (danger_score - 3) * 0.02)
            if our_cruise.get('avoid_big_pots', False):
                margin += 0.05
            
            if equity < pot_odds + margin:
                return FoldAction() if FoldAction in legal else CheckAction()
            
            if our_nuttedness >= 7:
                raise_threshold = 0.50
            elif our_nuttedness >= 5:
                raise_threshold = 0.60
            else:
                raise_threshold = 0.75 / aggression
            
            if RaiseAction in legal and equity >= raise_threshold and our_nuttedness >= 5:
                mn, mx = round_state.raise_bounds()
                mult = 3.0 if our_nuttedness >= 8 else 2.5
                target = int(max(mn, min(mx, pot + mult * aggression * continue_cost)))
                return RaiseAction(target)
            
            return CallAction() if CallAction in legal else CheckAction()

        if RaiseAction not in legal:
            return CheckAction()

        base_threshold = 0.50 * tightness
        board_nut = danger['board_nuttedness']
        
        if board_nut >= 8 and our_nuttedness < 5:
            base_threshold += 0.15
        elif board_nut >= 5 and our_nuttedness < 3:
            base_threshold += 0.08
        elif board_nut < 3 and our_nuttedness >= 5:
            base_threshold -= 0.10
        
        if our_cruise.get('avoid_big_pots', False):
            base_threshold += 0.08

        if equity < base_threshold:
            return CheckAction()

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

    # =====================
    # Street Detection
    # =====================
    
    def _get_street(self, round_state):
        """
        Get the current street from round_state.
        
        Game engine street values:
        - 0: Preflop betting
        - 2: BB discard (or CheckAction if you're SB)
        - 3: SB discard (or CheckAction if you're BB)
        - 4: Flop betting (post-discards)
        - 5: Turn betting
        - 6: River betting
        
        Note: Street 1 is skipped (flop dealt, no actions).
        """
        return round_state.street

    # =====================
    # CFR Action Selection
    # =====================

    def _get_legal_cfr_actions(self, round_state, active_player):
        legal = round_state.legal_actions()
        cfr_legal = []
        if FoldAction in legal:
            cfr_legal.append(FOLD)
        if CheckAction in legal or CallAction in legal:
            cfr_legal.append(CHECK_CALL)
        if RaiseAction in legal:
            cfr_legal.append(RAISE_SMALL)
            cfr_legal.append(RAISE_LARGE)
        return cfr_legal

    def _cfr_action_to_skeleton(self, cfr_action, round_state, active_player, aggression_mult=1.0):
        legal = round_state.legal_actions()
        my_pip = round_state.pips[active_player]
        opp_pip = round_state.pips[1 - active_player]
        my_stack = round_state.stacks[active_player]
        opp_stack = round_state.stacks[1 - active_player]
        pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)
        continue_cost = opp_pip - my_pip
        
        if cfr_action == FOLD:
            return FoldAction() if FoldAction in legal else (CheckAction() if CheckAction in legal else CallAction())
        elif cfr_action == CHECK_CALL:
            if continue_cost > 0:
                return CallAction() if CallAction in legal else CheckAction()
            return CheckAction() if CheckAction in legal else CallAction()
        elif cfr_action == RAISE_SMALL:
            if RaiseAction in legal:
                mn, mx = round_state.raise_bounds()
                return RaiseAction(max(mn, min(mx, int(pot * 0.55 * aggression_mult))))
            return CallAction() if CallAction in legal else CheckAction()
        elif cfr_action == RAISE_LARGE:
            if RaiseAction in legal:
                mn, mx = round_state.raise_bounds()
                return RaiseAction(max(mn, min(mx, int(pot * 1.0 * aggression_mult))))
            return CallAction() if CallAction in legal else CheckAction()
        return CheckAction() if CheckAction in legal else CallAction()

    def pick_cfr_action(self, game_state, round_state, active_player):
        board = self._get_board_cards(round_state)
        hole = list(round_state.hands[active_player])
        my_stack = round_state.stacks[active_player]
        opp_stack = round_state.stacks[1 - active_player]
        pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)
        effective_stack = min(my_stack, opp_stack)
        street = self._get_street(round_state)
        hole_strs = self._to_card_strings(hole)
        board_strs = self._to_card_strings(board)
        cfr_legal = self._get_legal_cfr_actions(round_state, active_player)
        
        probs = self.cfr.get_action_probs(
            player=active_player, street=street, hole_cards=hole_strs, board_cards=board_strs,
            pot=pot, effective_stack=effective_stack, betting_history=self.betting_history,
            bb_discarded=self.bb_discarded, sb_discarded=self.sb_discarded, legal_actions=cfr_legal,
        )
        
        cfr_hit = getattr(self.cfr, '_last_lookup_hit', False)
        if cfr_hit:
            self.cfr_hits += 1
        else:
            self.cfr_misses += 1
        
        actions = list(probs.keys())
        weights = [probs[a] for a in actions]
        cfr_action = random.choices(actions, weights=weights, k=1)[0] if sum(weights) > 0 else CHECK_CALL
        
        opp_cruise = self._opponent_cruise_proximity(game_state)
        skeleton_action = self._cfr_action_to_skeleton(cfr_action, round_state, active_player, opp_cruise['aggression'])
        
        return cfr_action, skeleton_action, cfr_hit

    # =====================
    # Main Action Logic
    # =====================

    def get_betting_action(self, game_state, round_state, active_player):
        legal = round_state.legal_actions()
        board = self._get_board_cards(round_state)
        hole = list(round_state.hands[active_player])
        my_pip = round_state.pips[active_player]
        opp_pip = round_state.pips[1 - active_player]
        my_stack = round_state.stacks[active_player]
        opp_stack = round_state.stacks[1 - active_player]
        pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)
        continue_cost = opp_pip - my_pip
        our_cruise = self._our_cruise_proximity(game_state)
        
        # Safety Override: Big Bets
        if continue_cost > 0:
            bet_analysis = self._analyze_bet(continue_cost, pot, my_stack, opp_stack)
            our_nuttedness = self._compute_our_nuttedness(hole, board) if len(board) >= 2 else 0
            
            if bet_analysis['shove'] and our_nuttedness < 7:
                return FoldAction() if FoldAction in legal else CheckAction()
            elif bet_analysis['type'] == 'MASSIVE_OVERBET' and our_nuttedness < 6:
                return FoldAction() if FoldAction in legal else CheckAction()
            elif bet_analysis['type'] == 'OVERBET' and our_nuttedness < 5:
                return FoldAction() if FoldAction in legal else CheckAction()
        
        # Cruise Proximity Override
        if our_cruise['status'] == 'ALMOST_THERE' and continue_cost > 0:
            our_nuttedness = self._compute_our_nuttedness(hole, board) if len(board) >= 2 else 0
            pot_odds = continue_cost / (pot + continue_cost)
            if our_nuttedness < 5 and pot_odds > 0.15:
                return FoldAction() if FoldAction in legal else CheckAction()
        
        # Try CFR Strategy
        cfr_action, skeleton_action, cfr_hit = self.pick_cfr_action(game_state, round_state, active_player)
        
        # If CFR missed, use FULL MC fallback
        if not cfr_hit:
            street_n = len(board)
            if street_n == 0:
                skeleton_action = self.mc_preflop_action(game_state, round_state, active_player)
            else:
                skeleton_action = self.mc_postflop_action(game_state, round_state, active_player)
            
            # Map back for history
            if isinstance(skeleton_action, FoldAction):
                cfr_action = FOLD
            elif isinstance(skeleton_action, (CheckAction, CallAction)):
                cfr_action = CHECK_CALL
            elif isinstance(skeleton_action, RaiseAction):
                cfr_action = RAISE_LARGE if skeleton_action.amount > pot * 0.7 else RAISE_SMALL
        
        self.betting_history.append((active_player, cfr_action))
        return skeleton_action

    # =====================
    # Framework Hooks
    # =====================

    def handle_new_round(self, game_state, round_state, active_player):
        self.total_hands += 1
        self.betting_history = []
        self.bb_discarded = False
        self.sb_discarded = False

    def handle_round_over(self, game_state, terminal_state, active_player):
        my_delta = terminal_state.deltas[active_player]
        self.cruise_mode = self.should_drain_to_one(game_state, active_player)
        if my_delta > 0 and my_delta <= 2:
            self.opponent_fold_count += 1
        
        # if self.total_hands % 100 == 0 and self.total_hands > 0:
        #     total = self.cfr_hits + self.cfr_misses
        #     if total > 0:
        #         print(f"[DEBUG] Hand {self.total_hands}: CFR hit rate = {self.cfr_hits/total*100:.1f}% ({self.cfr_hits}/{total})")
        #         print(self.cfr.debug_miss_summary(topk=3))
        # print(game_state.game_clock)

    def get_action(self, game_state, round_state, active_player):
        legal = round_state.legal_actions()
        street = round_state.street

        if self.cruise_mode:
            if FoldAction in legal:
                return FoldAction()
            if CheckAction in legal:
                return CheckAction()
            return CallAction()

        if DiscardAction in legal:
            idx = self.choose_discard_mc(game_state, round_state, active_player)
            if active_player == 1:
                self.bb_discarded = True
            else:
                self.sb_discarded = True
            return DiscardAction(idx)

        return self.get_betting_action(game_state, round_state, active_player)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
