"""
Simple example pokerbot, written in Python.
"""

from skeleton.actions import CallAction, CheckAction, FoldAction, RaiseAction, DiscardAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import (
    BIG_BLIND,
    NUM_ROUNDS,
    SMALL_BLIND,
    STARTING_STACK,
    GameState,
    RoundState,
    TerminalState,
)

# Set to True if you want to use GPT-4 to generate responses,
# and False if you want to manually input responses.
USE_GPT = False

if USE_GPT:
    import openai

    openai.api_key = "ENTER OPENAI API KEY!"


def chat(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview", messages=messages
    )
    return response.choices[0].message.content.strip()


ROLE = "You are an expert Poker player who is also good at playing different variants."

GAME_RULES = """
I want you to play a variant of Poker with me.
This is a 2-player variant similar to Texas Hold'em, but with key differences:
1. Each player is dealt 3 hole cards instead of 2.
2. The flop starts with only 2 community cards.
3. After the flop betting, players take turns discarding one card from their hand into the community cards.
   First player A discards (making it 3 community cards), then player B discards (making it 4 community cards).
4. After both discards, the turn is dealt (5 community cards total).
5. Then the river is dealt (6 community cards total).
6. At showdown, players use their remaining 2 hole cards and the 6 community cards to make the best 5-card hand.

The cards will be formatted as 'ab', where b is the suit (h = heart, d = diamond, s = spade, c = club),
and a is the rank (can be numbers from 2-9, or T (10), J (jack), Q (Queen), K (King), A (Ace)).
The starting stack for a round is 400, with the small blind being 1 and big blind equal 2.

Format your response as follows:
- To Fold, Call, or Check: just respond with that single word.
- To Raise: respond 'Raise x' where x is the total amount you want to raise to.
- To Discard: respond 'Discard x' where x is 0, 1, or 2 indicating which card from your hand to discard.

For example, if you want to call, respond with 'Call'. If you want to raise to 10, respond 'Raise 10'.
If you want to discard your second card (index 1), respond 'Discard 1'.
""".replace(
    "\n", " "
).strip()

ASSISTANT_AGREES = """
Of course, let's play this variant of Poker. Please provide me with the current game scenario,
including my hole cards, the visible community cards, my chip stack, my current
contribution to the pot, and the legal actions available to me.
""".replace(
    "\n", " "
).strip()


class Player(Bot):
    """
    A pokerbot.
    """

    def __init__(self):
        """
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        """
        self.messages = [
            {"role": "system", "content": ROLE},
            {"role": "user", "content": GAME_RULES},
            {"role": "assistant", "content": ASSISTANT_AGREES},
        ]
        self.new_message = ""
        self.is_gpt = False

    def handle_new_round(self, game_state, round_state, active):
        """
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        """
        game_clock = (
            game_state.game_clock
        )  # the total number of seconds your bot has left to play this game

        big_blind = bool(active)  # True if you are the big blind
        print(
            "================================NEW ROUND==================================="
        )
        print("You are", "big blind!" if big_blind else "small blind!")
        self.new_message = "You are " + ("big blind!" if big_blind else "small blind!")

    def handle_round_over(self, game_state, terminal_state, active):
        """
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        """
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        opp_cards = previous_state.hands[
            1 - active
        ]  # opponent's cards or [] if not revealed
        print()
        if opp_cards:
            print("Your opponent revealed", ", ".join(opp_cards))
            self.new_message += " Your opponent revealed " + ", ".join(opp_cards) + "."

        print("This round, your bankroll changed by", str(my_delta) + "!")

        self.new_message += (
            " This round, your bankroll changed by "
            + str(my_delta)
            + "! Onto the next round - Say yes to continue."
        )
        print()

        if self.is_gpt:
            self.messages.append({"role": "user", "content": self.new_message})
            response = chat(self.messages)
            self.messages.append({"role": "assistant", "content": response})

        ask = input("Press enter to continue, or q to quit!\n")
        if ask in ["q", "quit", "Quit"]:
            exit()

    def get_action(self, game_state, round_state, active):
        """
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        """
        # May be useful, but you may choose to not use.
        legal_actions = (
            round_state.legal_actions()
        )  # the actions you are allowed to take
        street = round_state.street
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.board  # the board cards
        my_pip = round_state.pips[
            active
        ]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[
            1 - active
        ]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[
            1 - active
        ]  # the number of chips your opponent has remaining
        continue_cost = (
            opp_pip - my_pip
        )  # the number of chips needed to stay in the pot
        my_contribution = (
            STARTING_STACK - my_stack
        )  # the number of chips you have contributed to the pot
        opp_contribution = (
            STARTING_STACK - opp_stack
        )  # the number of chips your opponent has contributed to the pot

        # Street description for display
        street_names = {
            0: "Pre-flop",
            2: "Flop (Discard Phase)",
            3: "Flop (Discard Phase)",
            4: "Post-Discard",
            5: "Turn",
            6: "River",
        }
        current_street = street_names.get(street, f"Street {street}")

        print()
        print(f"=== {current_street} ===")
        print("Your current cards are:", ", ".join(my_cards))
        self.new_message += " Your current cards are: " + ", ".join(my_cards) + "."
        if board_cards:
            print("The community cards are:", ", ".join(board_cards))
            self.new_message += (
                " The community cards are: " + ", ".join(board_cards) + "."
            )
        else:
            print("There are no community cards yet.")
            self.new_message += " There are no community cards yet."

        print("Your current contribution to the pot is", my_contribution)
        self.new_message += (
            " Your current contribution to the pot is " + str(my_contribution) + "."
        )
        print("Your remaining stack is", my_stack)
        self.new_message += " Your remaining stack is " + str(my_stack) + "."

        if my_contribution != 1 and continue_cost > 0:
            print("Your opponent raised by", continue_cost)
            self.new_message += " Your opponent raised by " + str(continue_cost) + "."

        poss_actions = "Your legal actions are: "

        if DiscardAction in legal_actions:
            poss_actions += "Discard, "
        if RaiseAction in legal_actions:
            poss_actions += "Raise, "
        if FoldAction in legal_actions:
            poss_actions += "Fold, "
        if CallAction in legal_actions:
            poss_actions += "Call, "
        if CheckAction in legal_actions:
            poss_actions += "Check, "
        print(poss_actions[:-2] + ".\n")
        self.new_message += " " + poss_actions[:-2] + "."

        if RaiseAction in legal_actions:
            min_raise, max_raise = (
                round_state.raise_bounds()
            )  # the smallest and largest numbers of chips for a legal bet/raise
            min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
            max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
            print(f"Raise bounds: {min_raise} to {max_raise}")

        if DiscardAction in legal_actions:
            print("You must discard one card. Cards are indexed 0, 1, 2.")
            for i, card in enumerate(my_cards):
                print(f"  {i}: {card}")

        if self.is_gpt:
            self.messages.append({"role": "user", "content": self.new_message})
            response = chat(self.messages)
            self.messages.append({"role": "assistant", "content": response})
            print("GPT-4:", response)
            response = response.split()
            if len(response) == 1:
                act = response[0]
            elif len(response) == 2:
                act, num = response
                num = int(num)
            else:
                print("Error: GPT gave too many words.")
                exit()
            self.new_message = ""
        else:
            user_input = input("Enter your move:\n")
            act = None
            num = None
            while act is None:
                parts = user_input.split(" ")
                if parts[0] in ["Quit", "quit", "q"]:
                    exit()
                if len(parts) != 1 and len(parts) != 2:
                    user_input = input("Too many words. Re-enter move: \n")
                elif len(parts) == 1:
                    act = parts[0].capitalize()
                    if act not in ["Check", "Fold", "Call"]:
                        act = None
                        user_input = input(
                            "One-word moves are only Check, Fold and Call. Re-enter move: \n"
                        )
                else:
                    act, num = parts
                    act = act.capitalize()
                    if act not in ["Raise", "Discard"]:
                        act = None
                        user_input = input(
                            "Two-word moves are only Raise and Discard. Re-enter move: \n"
                        )
                    else:
                        try:
                            num = int(num)
                        except ValueError:
                            act = None
                            user_input = input(
                                "Integer not entered for Raise/Discard. Enter new move: \n"
                            )

        if act == "Raise":
            return RaiseAction(num)
        elif act == "Discard":
            return DiscardAction(num)
        elif act == "Check":
            return CheckAction()
        elif act == "Call":
            return CallAction()
        else:
            return FoldAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
