#include "abstraction.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace tossem_abs {

static inline int rank_of(uint8_t c) { return static_cast<int>(c / 4); }
static inline int suit_of(uint8_t c) { return static_cast<int>(c % 4); }

std::size_t InfoKeyHash::operator()(const InfoKey& k) const noexcept {
  // FNV-1a mix
  std::size_t h = 1469598103934665603ull;
  auto mix = [&](std::size_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
  };
  mix(k.player); mix(k.street); mix(k.hole_bucket); mix(k.board_bucket);
  mix(k.pot_bucket); mix(k.hist_bucket);
  mix(k.bb_discarded); mix(k.sb_discarded); mix(k.legal_mask);
  return h;
}

static inline bool is_suited(uint8_t a, uint8_t b) { return suit_of(a) == suit_of(b); }

uint16_t get_hole_bucket_2card(uint8_t c1, uint8_t c2) {
  int r0 = rank_of(c1);
  int r1 = rank_of(c2);
  int hi = std::max(r0, r1);
  int lo = std::min(r0, r1);
  bool suited = is_suited(c1, c2);

  // pairs: 13 buckets
  if (hi == lo) {
    return static_cast<uint16_t>(hi); // 0..12
  }
  // non-pairs: base = 13 + (hi*(hi-1))//2 + lo; then +78 if suited
  int base = 13 + (hi * (hi - 1)) / 2 + lo;
  if (suited) base += 78;
  return static_cast<uint16_t>(base);
}

uint16_t get_hole_bucket(const std::vector<uint8_t>& hole_cards) {
  if (hole_cards.size() == 2) {
    return get_hole_bucket_2card(hole_cards[0], hole_cards[1]);
  }
  
  // 3-card hand: compute coarse features and map into 40 buckets
  std::array<int,3> ranks = {rank_of(hole_cards[0]), rank_of(hole_cards[1]), rank_of(hole_cards[2])};
  std::sort(ranks.begin(), ranks.end(), std::greater<int>());

  int a=ranks[0], b=ranks[1], c=ranks[2];
  bool trips = (a==b && b==c);
  bool pair = (a==b || b==c || a==c);

  // Suits
  std::array<int,4> suit_cnt{0,0,0,0};
  suit_cnt[suit_of(hole_cards[0])]++;
  suit_cnt[suit_of(hole_cards[1])]++;
  suit_cnt[suit_of(hole_cards[2])]++;
  int flush_count = *std::max_element(suit_cnt.begin(), suit_cnt.end());

  // Straight potential
  std::vector<int> uniq(ranks.begin(), ranks.end());
  std::sort(uniq.begin(), uniq.end(), std::greater<int>());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  int straight_potential = 0;
  if (uniq.size() >= 2) {
    for (size_t i=0; i+1<uniq.size(); ++i) {
      if (uniq[i] - uniq[i+1] <= 2) straight_potential++;
    }
  }

  // Heuristic strength score
  int strength = a*2 + b + c;
  if (trips) strength += 30;
  else if (pair) strength += 15;
  strength += (flush_count - 1) * 8;
  strength += straight_potential * 5;

  // Bucket into 40 bins (reduced from 60)
  int bucket = strength / 6;
  if (bucket < 0) bucket = 0;
  if (bucket > 39) bucket = 39;
  return static_cast<uint16_t>(bucket);
}

uint16_t get_board_bucket(const std::vector<uint8_t>& board) {
  if (board.empty()) {
    return 0;
  }

  std::vector<int> ranks;
  std::vector<int> suits;
  ranks.reserve(board.size());
  suits.reserve(board.size());
  for (auto c : board) {
    ranks.push_back(rank_of(c));
    suits.push_back(suit_of(c));
  }

  // Rank counts
  std::array<int,13> rc{};
  for (int r : ranks) rc[r]++;
  int max_rank_count = *std::max_element(rc.begin(), rc.end());

  // Suit counts
  std::array<int,4> sc{};
  for (int s : suits) sc[s]++;
  int max_suit_count = *std::max_element(sc.begin(), sc.end());

  // Straight potential
  std::vector<int> uniq = ranks;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  int straight_potential = 0;
  for (size_t i=0; i<uniq.size(); ++i) {
    for (size_t j=i+1; j<uniq.size(); ++j) {
      if (uniq[j] - uniq[i] <= 4) {
        straight_potential = std::max(straight_potential, static_cast<int>(j - i + 1));
      }
    }
  }

  int high_card = *std::max_element(ranks.begin(), ranks.end());

  // Simplified features into ~25 buckets
  int paired = (max_rank_count >= 2) ? 1 : 0;
  int flush_draw = std::min(2, max_suit_count - 1);
  int straight_draw = std::min(2, std::max(0, straight_potential - 2));
  int high = (high_card >= 10) ? 1 : 0;  // T=8, J=9, Q=10, K=11, A=12

  int bucket = paired * 12 + flush_draw * 4 + straight_draw * 2 + high;
  if (bucket > 24) bucket = 24;
  return static_cast<uint16_t>(bucket);
}

uint8_t get_pot_bucket(int pot) {
  if (pot <= 4) return 0;
  if (pot <= 10) return 1;
  if (pot <= 25) return 2;
  if (pot <= 60) return 3;
  if (pot <= 140) return 4;
  return 5;
}

uint8_t get_history_bucket(const std::vector<std::pair<int,int>>& hist) {
  // Simplified: 6 buckets
  if (hist.empty()) return 0;

  int raises = 0;
  int large_raises = 0;
  for (const auto& pa : hist) {
    if (pa.second == RAISE_SMALL) {
      raises++;
    } else if (pa.second == RAISE_LARGE) {
      raises++;
      large_raises++;
    }
  }

  if (raises == 0) return 1;  // passive
  if (raises == 1 && large_raises == 0) return 2;  // one small raise
  if (raises == 1 && large_raises == 1) return 3;  // one large raise
  if (raises == 2) return 4;  // two raises
  return 5;  // very aggressive
}

InfoKey compute_info_key(
  int player,
  int street,
  const std::vector<uint8_t>& hole_cards,
  const std::vector<uint8_t>& board_cards,
  int pot,
  int eff_stack,
  const std::vector<std::pair<int,int>>& betting_history,
  bool bb_discarded,
  bool sb_discarded,
  uint16_t legal_action_mask
) {
  (void)eff_stack;  // Not used in simplified version
  
  InfoKey k{};
  k.player = static_cast<uint8_t>(player);
  k.street = static_cast<uint8_t>(street);
  k.hole_bucket = get_hole_bucket(hole_cards);
  k.board_bucket = get_board_bucket(board_cards);
  k.pot_bucket = get_pot_bucket(pot);
  k.hist_bucket = get_history_bucket(betting_history);
  k.bb_discarded = bb_discarded ? 1 : 0;
  k.sb_discarded = sb_discarded ? 1 : 0;
  k.legal_mask = static_cast<uint8_t>(legal_action_mask & 0x7F);
  return k;
}

std::string info_key_to_string(const InfoKey& k) {
  std::ostringstream oss;
  oss << "P" << int(k.player)
      << "|S" << int(k.street)
      << "|H" << k.hole_bucket
      << "|B" << k.board_bucket
      << "|POT" << int(k.pot_bucket)
      << "|HIST" << int(k.hist_bucket)
      << "|BB" << int(k.bb_discarded)
      << "|SB" << int(k.sb_discarded)
      << "|LA" << int(k.legal_mask);
  return oss.str();
}

} // namespace tossem_abs
