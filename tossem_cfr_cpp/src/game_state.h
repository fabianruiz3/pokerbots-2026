#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "abstraction.h"

namespace game {

constexpr int STARTING_STACK = 400;
constexpr int SMALL_BLIND = 1;
constexpr int BIG_BLIND = 2;

// Distinct action ids: betting 0..3, discards 4..6 (discard index = action-4)
constexpr int DISCARD0 = 4;
constexpr int DISCARD1 = 5;
constexpr int DISCARD2 = 6;
constexpr int NUM_DISTINCT_ACTIONS = 7;

struct Undo {
  // scalar snapshot
  int street;
  int current_player;
  std::array<int,2> pips;
  std::array<int,2> stacks;
  bool bb_discarded;
  bool sb_discarded;
  bool is_terminal;
  std::array<double,2> payoffs;

  // history sizes
  std::size_t history_size;
  std::size_t street_history_size;

  // deck idx
  int deck_idx;

  // cards: we use fixed arrays; store sizes + changed cells
  std::array<int,2> hand_sizes;
  int board_size;
};

struct GameState {
  // hands: max 3 each
  std::array<std::array<uint8_t,3>,2> hands;
  std::array<int,2> hand_sizes{{0,0}};

  // board: max 6
  std::array<uint8_t,6> board;
  int board_size = 0;

  // deck: 46 after dealing 6
  std::array<uint8_t,46> deck;
  int deck_idx = 0;

  // street - uses 7-street system (0-6)
  // 0: PREFLOP, 1: FLOP_DEAL (skip), 2: BB_DISCARD, 3: SB_DISCARD, 4: FLOP_BET, 5: TURN, 6: RIVER
  int street = tossem_abs::STREET_PREFLOP;
  std::array<int,2> pips{{SMALL_BLIND, BIG_BLIND}};
  std::array<int,2> stacks{{STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND}};
  int current_player = 0; // SB acts first preflop

  // history (player, action)
  std::vector<std::pair<int,int>> history;
  std::vector<std::pair<int,int>> street_history;

  // discards
  bool bb_discarded = false;
  bool sb_discarded = false;

  // terminal
  bool is_terminal = false;
  std::array<double,2> payoffs{{0.0,0.0}};

  void reset(std::mt19937& rng);

  int pot() const;
  int continue_cost() const;
  int effective_stack() const;

  bool is_discard_phase() const;
  std::vector<int> legal_actions() const;

  void apply_action(int action, Undo& u);
  void undo_action(const Undo& u);

  // helper for CFR
  tossem_abs::InfoKey info_key(int player, const std::vector<int>& legal_actions) const;

private:
  bool should_advance_street() const;
  void advance_street();
  void apply_discard(int discard_idx);
  void showdown();
};

} // namespace game
