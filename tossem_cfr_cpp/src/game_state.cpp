#include "game_state.h"
#include "hand_eval.h"

#include <algorithm>
#include <numeric>

namespace game {

static inline int rank(uint8_t c) { return c / 4; }
static inline int suit(uint8_t c) { return c % 4; }

static void make_full_deck(std::array<uint8_t,52>& out) {
  for (int r=0;r<13;++r) {
    for (int s=0;s<4;++s) {
      out[r*4+s] = static_cast<uint8_t>(r*4+s);
    }
  }
}

void GameState::reset(std::mt19937& rng) {
  // full deck
  std::array<uint8_t,52> full;
  make_full_deck(full);
  std::shuffle(full.begin(), full.end(), rng);

  // deal 3 each
  for (int p=0;p<2;++p) {
    hand_sizes[p]=3;
    for (int i=0;i<3;++i) hands[p][i] = full[p*3+i];
  }

  // remaining deck 46
  for (int i=0;i<46;++i) deck[i] = full[6+i];
  deck_idx = 0;

  board_size = 0;
  street = tossem_abs::STREET_PREFLOP;
  pips = {SMALL_BLIND, BIG_BLIND};
  stacks = {STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND};
  current_player = 0;
  history.clear();
  street_history.clear();
  bb_discarded = false;
  sb_discarded = false;
  is_terminal = false;
  payoffs = {0.0,0.0};
}

int GameState::pot() const {
  return (STARTING_STACK - stacks[0]) + (STARTING_STACK - stacks[1]);
}

int GameState::continue_cost() const {
  return pips[1-current_player] - pips[current_player];
}

int GameState::effective_stack() const {
  return std::min(stacks[0], stacks[1]);
}

bool GameState::is_discard_phase() const {
  if (street == tossem_abs::STREET_BB_DISCARD && !bb_discarded) return true;
  if (street == tossem_abs::STREET_SB_DISCARD && !sb_discarded) return true;
  return false;
}

std::vector<int> GameState::legal_actions() const {
  if (is_terminal) return {};

  if (is_discard_phase()) {
    // discarder has 3 cards at discard time
    return {DISCARD0, DISCARD1, DISCARD2};
  }

  std::vector<int> actions;
  int cost = continue_cost();
  if (cost == 0) {
    actions.push_back(tossem_abs::CHECK_CALL);
    if (stacks[0] > 0 && stacks[1] > 0) {
      actions.push_back(tossem_abs::RAISE_SMALL);
      actions.push_back(tossem_abs::RAISE_LARGE);
    }
  } else {
    actions.push_back(tossem_abs::FOLD);
    actions.push_back(tossem_abs::CHECK_CALL);
    if (cost < stacks[current_player] && stacks[1-current_player] > 0) {
      actions.push_back(tossem_abs::RAISE_SMALL);
      actions.push_back(tossem_abs::RAISE_LARGE);
    }
  }
  return actions;
}

bool GameState::should_advance_street() const {
  if (street_history.size() < 2) return false;
  if (pips[0] == pips[1]) {
    int last_action = street_history.back().second;
    if (last_action == tossem_abs::CHECK_CALL) return true;
  }
  return false;
}

void GameState::advance_street() {
  pips = {0,0};
  street_history.clear();

  if (street == tossem_abs::STREET_PREFLOP) {
    // deal 2-card flop
    board[0] = deck[deck_idx];
    board[1] = deck[deck_idx+1];
    board_size = 2;
    deck_idx += 2;
    street = tossem_abs::STREET_FLOP;
    current_player = 1; // BB acts first postflop
  } else if (street == tossem_abs::STREET_FLOP) {
    street = tossem_abs::STREET_BB_DISCARD;
    current_player = 1; // BB discards
  } else if (street == tossem_abs::STREET_TURN) {
    board[board_size++] = deck[deck_idx++];
    street = tossem_abs::STREET_RIVER;
    current_player = 1;
  } else if (street == tossem_abs::STREET_RIVER) {
    showdown();
  }
}

void GameState::apply_discard(int discard_idx) {
  // discard_idx 0..2
  if (street == tossem_abs::STREET_BB_DISCARD) {
    int p = 1;
    uint8_t card = hands[p][discard_idx];
    // remove from hand by swapping with last
    int hs = hand_sizes[p];
    hands[p][discard_idx] = hands[p][hs-1];
    hand_sizes[p] = hs-1;

    board[board_size++] = card;
    bb_discarded = true;
    street = tossem_abs::STREET_SB_DISCARD;
    current_player = 0;
  } else {
    int p = 0;
    uint8_t card = hands[p][discard_idx];
    int hs = hand_sizes[p];
    hands[p][discard_idx] = hands[p][hs-1];
    hand_sizes[p] = hs-1;

    board[board_size++] = card;
    sb_discarded = true;

    // deal turn immediately (board 4->5)
    board[board_size++] = deck[deck_idx++];
    street = tossem_abs::STREET_TURN;
    current_player = 1;
    pips = {0,0};
    street_history.clear();
  }
}

void GameState::showdown() {
  is_terminal = true;

  // each has 2 hole + 6 board = 8 cards (we enforce hand_sizes==2 here)
  std::vector<uint8_t> cards0;
  std::vector<uint8_t> cards1;
  cards0.reserve(8);
  cards1.reserve(8);

  for (int i=0;i<hand_sizes[0];++i) cards0.push_back(hands[0][i]);
  for (int i=0;i<hand_sizes[1];++i) cards1.push_back(hands[1][i]);
  for (int i=0;i<board_size;++i) {
    cards0.push_back(board[i]);
    cards1.push_back(board[i]);
  }

  eval::HandValue h0 = eval::evaluate_best(cards0);
  eval::HandValue h1 = eval::evaluate_best(cards1);

  int res = eval::compare(h0, h1);
  int p = pot();
  if (res > 0) {
    payoffs = {p/2.0, -p/2.0};
  } else if (res < 0) {
    payoffs = {-p/2.0, p/2.0};
  } else {
    payoffs = {0.0,0.0};
  }
}

void GameState::apply_action(int action, Undo& u) {
  // snapshot minimal
  u.street = street;
  u.current_player = current_player;
  u.pips = pips;
  u.stacks = stacks;
  u.bb_discarded = bb_discarded;
  u.sb_discarded = sb_discarded;
  u.is_terminal = is_terminal;
  u.payoffs = payoffs;
  u.history_size = history.size();
  u.street_history_size = street_history.size();
  u.deck_idx = deck_idx;
  u.hand_sizes = hand_sizes;
  u.board_size = board_size;

  if (is_terminal) return;

  if (is_discard_phase()) {
    int discard_idx = action - 4;
    apply_discard(discard_idx);
    return;
  }

  int cost = continue_cost();
  int pot_sz = pot();

  if (action == tossem_abs::FOLD) {
    is_terminal = true;
    int winner = 1 - current_player;
    int delta = STARTING_STACK - stacks[winner];
    payoffs[winner] = static_cast<double>(delta);
    payoffs[1-winner] = -static_cast<double>(delta);
    return;
  } else if (action == tossem_abs::CHECK_CALL) {
    if (cost > 0) {
      int actual = std::min(cost, stacks[current_player]);
      pips[current_player] += actual;
      stacks[current_player] -= actual;
    }
  } else if (action == tossem_abs::RAISE_SMALL || action == tossem_abs::RAISE_LARGE) {
    double mult = (action == tossem_abs::RAISE_SMALL) ? 0.55 : 1.0;
    int raise_amt = static_cast<int>(pot_sz * mult);
    int min_raise = cost + std::max(cost, BIG_BLIND);
    raise_amt = std::max(min_raise, raise_amt);
    raise_amt = std::min(raise_amt, stacks[current_player]);

    int total_contrib = cost + raise_amt;
    total_contrib = std::min(total_contrib, stacks[current_player]);

    pips[current_player] += total_contrib;
    stacks[current_player] -= total_contrib;
  }

  history.emplace_back(current_player, action);
  street_history.emplace_back(current_player, action);

  if (should_advance_street()) {
    advance_street();
  } else {
    current_player = 1 - current_player;
  }
}

void GameState::undo_action(const Undo& u) {
  street = u.street;
  current_player = u.current_player;
  pips = u.pips;
  stacks = u.stacks;
  bb_discarded = u.bb_discarded;
  sb_discarded = u.sb_discarded;
  is_terminal = u.is_terminal;
  payoffs = u.payoffs;

  // restore sizes
  history.resize(u.history_size);
  street_history.resize(u.street_history_size);

  deck_idx = u.deck_idx;
  hand_sizes = u.hand_sizes;
  board_size = u.board_size;
}

tossem_abs::InfoKey GameState::info_key(int player, const std::vector<int>& legal_actions_vec) const {
  // gather hole & board
  std::vector<uint8_t> hole;
  hole.reserve(hand_sizes[player]);
  for (int i=0;i<hand_sizes[player];++i) hole.push_back(hands[player][i]);
  std::vector<uint8_t> boardv;
  boardv.reserve(board_size);
  for (int i=0;i<board_size;++i) boardv.push_back(board[i]);

  uint16_t la_mask = 0;
  for (int a : legal_actions_vec) {
    if (a >= 0 && a < NUM_DISTINCT_ACTIONS) la_mask |= static_cast<uint16_t>(1u << a);
  }

  return tossem_abs::compute_info_key(
    player,
    street,
    hole,
    boardv,
    pot(),
    effective_stack(),
    history,
    bb_discarded,
    sb_discarded,
    la_mask
  );
}

} // namespace game
