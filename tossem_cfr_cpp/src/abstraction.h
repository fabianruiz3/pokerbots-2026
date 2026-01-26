#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tossem_abs {

// Betting action ids (0..3)
constexpr int FOLD = 0;
constexpr int CHECK_CALL = 1;
constexpr int RAISE_SMALL = 2;
constexpr int RAISE_LARGE = 3;
constexpr int NUM_ACTIONS = 4;

// Streets
constexpr int STREET_PREFLOP = 0;
constexpr int STREET_FLOP = 1;
constexpr int STREET_BB_DISCARD = 2;
constexpr int STREET_SB_DISCARD = 3;
constexpr int STREET_TURN = 4;
constexpr int STREET_RIVER = 5;

constexpr int STARTING_STACK = 400;
constexpr int SMALL_BLIND = 1;
constexpr int BIG_BLIND = 2;

// Distinct action space for a *CFR state machine* that includes discards.
// We map discards to NUM_ACTIONS + {0,1,2}.
constexpr int DISCARD_BASE = NUM_ACTIONS; // 4
constexpr int NUM_DISCARD_ACTIONS = 3;
constexpr int NUM_DISTINCT_ACTIONS = NUM_ACTIONS + NUM_DISCARD_ACTIONS; // 7

// ===== Info key (tuple) for fast hashing =====
// Designed to mirror abstraction.compute_info_state(...) but without building strings.
struct InfoKey {
  uint8_t player = 0;
  uint8_t street = 0;
  uint16_t hole_bucket = 0;
  uint16_t board_bucket = 0;
  uint8_t pot_bucket = 0;
  uint8_t stack_bucket = 0;
  uint8_t hist_bucket = 0;
  uint8_t bb_discarded = 0;
  uint8_t sb_discarded = 0;
  uint8_t legal_mask = 0; // 7-bit mask over NUM_DISTINCT_ACTIONS (for OpenSpiel-style safety)

  bool operator==(const InfoKey& o) const noexcept {
    return player==o.player && street==o.street && hole_bucket==o.hole_bucket &&
           board_bucket==o.board_bucket && pot_bucket==o.pot_bucket && stack_bucket==o.stack_bucket &&
           hist_bucket==o.hist_bucket && bb_discarded==o.bb_discarded && sb_discarded==o.sb_discarded &&
           legal_mask==o.legal_mask;
  }
};

struct InfoKeyHash {
  std::size_t operator()(const InfoKey& k) const noexcept;
};

// Card helpers: card is uint8_t rank*4+suit where rank 0=2 .. 12=A, suit 0..3.
inline int rank(uint8_t c) { return c / 4; }
inline int suit(uint8_t c) { return c % 4; }

// Bucketing functions
uint16_t get_hole_bucket(const std::vector<uint8_t>& hole_cards);
uint16_t get_hole_bucket_2card(uint8_t c1, uint8_t c2);
uint16_t get_board_bucket(const std::vector<uint8_t>& board_cards);
uint8_t get_pot_bucket(int pot);
uint8_t get_stack_bucket(int eff_stack);
uint8_t get_history_bucket(const std::vector<std::pair<int,int>>& betting_history);

InfoKey compute_info_key(
  int player,
  int street,
  const std::vector<uint8_t>& hole_cards,
  const std::vector<uint8_t>& board_cards,
  int pot,
  int effective_stack,
  const std::vector<std::pair<int,int>>& betting_history,
  bool bb_discarded,
  bool sb_discarded,
  uint16_t legal_mask
);

// Optional: string version for debugging / exporting
std::string info_key_to_string(const InfoKey& k);

} // namespace tossem_abs
