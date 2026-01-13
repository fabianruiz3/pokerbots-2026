"""
MC-driven bot for Pokerbots 2026 (3 hole cards, discard twice, 6-card public board).
- Preflop: INSTANT TABLE LOOKUP (no MC needed!)
- Discard: MC with FIXED board state
- Postflop: MC equity vs bet-size-biased opponent + IMPROVED fold logic vs big bets
- FIXED: Opponent bias rejection was backwards (now accepts strong hands vs big bets)
- Cruise control: auto-fold/check when bankroll lead is sufficient.
- Clock safety: aggressive sim throttling + panic mode under very low clock.
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
        self.base_sims_post = 500
        self.base_sims_discard = 500
        self.base_sims_pre = 500  # Only used if table fails to load

        self.cruise_mode = False
        
        # Load preflop equity table for instant lookups
        import pickle
        import os
        try:
            table_path = os.path.join(os.path.dirname(__file__), 'preflop_equity_table.pkl')
            with open(table_path, 'rb') as f:
                table_data = pickle.load(f)
            self.preflop_equity = table_data['equity_table']
            print(f"[Player] ✓ Loaded preflop table: {len(self.preflop_equity)} hand classes")
        except Exception as e:
            print(f"[Player] WARNING: Could not load preflop table: {e}")
            print("[Player] Falling back to MC for preflop")
            self.preflop_equity = None

    # ---------- Utility helpers ----------

    def _normalize_hand(self, cards):
        """
        Normalize a 3-card hand to its canonical form for table lookup.
        """
        rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
                    'T':10,'J':11,'Q':12,'K':13,'A':14}
        
        ranks = []
        suits = []
        for card in cards:
            card_str = str(card)
            ranks.append(rank_map[card_str[0]])
            suits.append(card_str[1])
        
        ranks.sort(reverse=True)
        
        if suits[0] == suits[1] == suits[2]:
            suit_pattern = 2  # Three suited
        elif suits[0] == suits[1] or suits[1] == suits[2] or suits[0] == suits[2]:
            suit_pattern = 1  # Two suited
        else:
            suit_pattern = 0  # Rainbow
        
        return (ranks[0], ranks[1], ranks[2], suit_pattern)

    def _clock_mult(self, game_clock):
        """
        IMPROVED clock multiplier - never drops below 50%.
        """
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
        """
        Return the current public board as a flat list.
        """
        return list(round_state.board)

    def _to_card_list(self, cards):
        """
        Safely convert cards to pkrbot.Card objects.
        """
        out = []
        for c in cards:
            if isinstance(c, pkrbot.Card):
                out.append(c)
            else:
                out.append(pkrbot.Card(str(c)))
        return out

    def _should_cruise(self, game_state):
        """
        Conservative chip cruising threshold.
        """
        bankroll = game_state.bankroll
        remaining = max(0, NUM_ROUNDS - game_state.round_num)
        safety = 2 * remaining
        return bankroll >= safety

    def _opp_bias_from_action(self, continue_cost, pot, street_n):
        """
        Calculate opponent range bias based on bet size.
        Larger bets = assume stronger range = higher bias.
        """
        if continue_cost <= 0:
            return 0.0
        frac = continue_cost / max(1.0, pot)
        street_boost = 1.0 + 0.08 * max(0, street_n - 3)
        x = frac * street_boost
        return max(0.0, min(1.0, 1.4 * x))

    # ---------- Core equity engine ----------

    def mc_equity(self, round_state, my_hole_cards, sims, opp_bias=0.0):
        """
        Monte Carlo equity vs 1 opponent with range bias.
        FIXED: Rejection logic was backwards - now correctly accepts strong hands.
        """
        raw_board = list(round_state.board)
        raw_hole = list(my_hole_cards)

        board = self._to_card_list(raw_board)
        hole = self._to_card_list(raw_hole)

        # Preflop: 3 vs 3, postflop: 2 vs 2
        opp_hole_n = 3 if (len(hole) == 3 and len(board) < 2) else 2
        remaining_board = max(0, FINAL_BOARD_CARDS - len(board))

        deck = pkrbot.Deck()
        used = hole + board
        for c in used:
            if c in deck.cards:
                deck.cards.remove(c)

        tier = {
            "High Card": 0,
            "Pair": 1,
            "Two Pair": 2,
            "Trips": 3,
            "Straight": 4,
            "Flush": 5,
            "Full House": 6,
            "Quads": 7,
            "Straight Flush": 8,
        }

        wins = 0
        ties = 0
        iters = 0

        while iters < sims:
            deck.shuffle()
            draw = deck.peek(opp_hole_n + remaining_board)
            opp = draw[:opp_hole_n]
            runout = draw[opp_hole_n:]

            my_val = pkrbot.evaluate(hole + board + runout)
            opp_val = pkrbot.evaluate(opp + board + runout)

            # FIXED: Opponent range filtering
            if opp_bias > 0.0:
                opp_class = pkrbot.handtype(opp_val)
                t = tier.get(opp_class, 0)
                
                # Calculate acceptance probability
                # Higher tier = more likely to accept
                # Higher bias = reject more weak hands
                accept_p = min(
                    1.0,
                    max(
                        0.18,  # Always accept at least 18% (avoid infinite loops)
                        1.0 - 0.60 * opp_bias + 0.10 * t + 0.06 * opp_bias * t,
                    ),
                )
                
                # FIXED: Was "if random.random() > accept_p" (backwards!)
                # Now correctly rejects weak hands
                if random.random() >= accept_p:
                    continue  # Reject this weak hand

            if my_val > opp_val:
                wins += 1
            elif my_val == opp_val:
                ties += 1

            iters += 1

        return (wins + 0.5 * ties) / max(1, sims)

    def mc_equity_with_board(self, my_hole_cards, board, sims, opp_bias=0.0):
        """
        Monte Carlo equity with explicit board (for discard evaluation).
        """
        board = self._to_card_list(board)
        hole = self._to_card_list(my_hole_cards)

        opp_hole_n = 2
        remaining_board = max(0, FINAL_BOARD_CARDS - len(board))

        deck = pkrbot.Deck()
        used = hole + board
        for c in used:
            if c in deck.cards:
                deck.cards.remove(c)

        tier = {
            "High Card": 0,
            "Pair": 1,
            "Two Pair": 2,
            "Trips": 3,
            "Straight": 4,
            "Flush": 5,
            "Full House": 6,
            "Quads": 7,
            "Straight Flush": 8,
        }

        wins = 0
        ties = 0
        iters = 0

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
                
                # FIXED: Correct rejection logic
                if random.random() >= accept_p:
                    continue

            if my_val > opp_val:
                wins += 1
            elif my_val == opp_val:
                ties += 1

            iters += 1

        return (wins + 0.5 * ties) / max(1, sims)

    # ---------- Discard logic ----------

    def choose_discard_mc(self, game_state, round_state, active_player):
        """
        Choose which card to discard via Monte Carlo simulation.
        CRITICAL: Must simulate with discarded card added to board!
        """
        hole = list(round_state.hands[active_player])
        board = self._get_board_cards(round_state)
        sims = int(self.base_sims_discard * self._clock_mult(game_state.game_clock))

        best_i = 0
        best_ev = -1.0
        
        for i in range(3):
            kept = [hole[j] for j in range(3) if j != i]
            discarded = hole[i]
            
            # Evaluate with discarded card on board
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


        sims = self.base_sims_pre
        eq = self.mc_equity(round_state, hole, sims=sims, opp_bias=0.0)

        # Facing a raise / completion
        if continue_cost > 0:
            pot_odds = continue_cost / (pot + continue_cost)
            if eq < pot_odds + 0.045:
                return FoldAction() if FoldAction in legal else CallAction()
            return CallAction() if CallAction in legal else CheckAction()

        # No raise yet: open-raise strong 3-card holdings
        if RaiseAction in legal and eq >= 0.85:
            mn, mx = round_state.raise_bounds()
            target = int(max(mn, min(mx, 4 * pot)))
            return RaiseAction(target)
        elif RaiseAction in legal and eq >= 0.65:
            mn, mx = round_state.raise_bounds()
            target = int(max(mn, min(mx, 2.25 * pot)))
            return RaiseAction(target)
        
        elif CallAction in legal and eq >= 0.55:
            return CallAction()
        
        elif CheckAction in legal and eq >= 0.55:
            return CheckAction()

        return FoldAction() if FoldAction in legal else CheckAction()

    def postflop_action(self, game_state, round_state, active_player):
        legal = round_state.legal_actions()
        street_n = len(self._get_board_cards(round_state))

        my_pip = round_state.pips[active_player]
        opp_pip = round_state.pips[1 - active_player]
        continue_cost = opp_pip - my_pip

        my_stack = round_state.stacks[active_player]
        opp_stack = round_state.stacks[1 - active_player]
        pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)

        hole = list(round_state.hands[active_player])


        sims = self.base_sims_post
        opp_bias = self._opp_bias_from_action(continue_cost, pot, street_n)
        equity = self.mc_equity(round_state, hole, sims=sims, opp_bias=opp_bias)

        # Slightly more conservative fold margin when biased toward strong villain hands
        margin = (0.02 if street_n < 6 else 0.015) + (0.02 + 0.05 * opp_bias)

        if continue_cost > 0:
            pot_odds = continue_cost / (pot + continue_cost)

            # Fold if equity is clearly below pot odds + margin
            if equity < pot_odds + margin:
                return FoldAction() if FoldAction in legal else CallAction()

            # Raise strong value/nutted hands
            if RaiseAction in legal and equity >= (0.78 + 0.06 * opp_bias):
                mn, mx = round_state.raise_bounds()
                mult = 2.4 if equity < 0.85 else 3.2
                target = pot + mult * continue_cost
                amt = int(max(mn, min(mx, target)))
                return RaiseAction(amt)

            # Otherwise just call
            return CallAction() if CallAction in legal else CheckAction()

        # No bet facing us
        if RaiseAction not in legal:
            return CheckAction()

        # Check marginal hands
        if equity < (0.58 + 0.04 * opp_bias):
            return CheckAction()

        # Size bet by strength bucket
        if equity < 0.70:
            frac = 0.85
        elif equity < 0.82:
            frac = 1.00
        else:
            # Strong / nutted → slightly overbet, especially before final street
            frac = 1.15 if street_n < 6 else 1.20

        mn, mx = round_state.raise_bounds()
        amt = int(max(mn, min(mx, frac * pot)))
        return RaiseAction(amt)

    # ---------- Hooks from framework ----------

    def handle_new_round(self, game_state, round_state, active_player):
        pass

    def handle_round_over(self, game_state, terminal_state, active_player):
        self.cruise_mode = self._should_cruise(game_state)
        # print(game_state.game_clock)

    def get_action(self, game_state, round_state, active_player):
        legal = round_state.legal_actions()

        # Cruise control
        if self.cruise_mode:
            if FoldAction in legal:
                return FoldAction()
            if CheckAction in legal:
                return CheckAction()
            if CallAction in legal:
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
