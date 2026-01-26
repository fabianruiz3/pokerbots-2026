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
  mix(k.pot_bucket); mix(k.stack_bucket); mix(k.hist_bucket);
  mix(k.bb_discarded); mix(k.sb_discarded); mix(k.legal_mask);
  return h;
}

static inline bool is_suited(uint8_t a, uint8_t b) { return suit_of(a) == suit_of(b); }

uint16_t hole_bucket_2card(const std::array<uint8_t,2>& hole) {
  // Ported from abstraction.py:get_hole_bucket_2card
  int r0 = rank_of(hole[0]);
  int r1 = rank_of(hole[1]);
  int hi = std::max(r0,r1);
  int lo = std::min(r0,r1);
  bool suited = is_suited(hole[0], hole[1]);

  // pairs: 13 buckets at the top
  if (hi == lo) {
    return static_cast<uint16_t>(hi); // 0..12
  }
  // non-pairs: 78 combos x suited/offsuit => 156, but we compress similarly to python
  // python: base = 13 + (hi*(hi-1))//2 + lo; then +78 if suited
  int base = 13 + (hi * (hi - 1)) / 2 + lo;
  if (suited) base += 78;
  return static_cast<uint16_t>(base);
}

uint16_t hole_bucket_3card(const std::array<uint8_t,3>& hole) {
  // Ported from abstraction.py:get_hole_bucket (3-card).
  // We compute a few coarse features and map into 0..(NUM_HOLE_BUCKETS-1).
  // IMPORTANT: This is intentionally consistent with the Python version shipped in your zip.

  std::array<int,3> ranks = {rank_of(hole[0]), rank_of(hole[1]), rank_of(hole[2])};
  std::sort(ranks.begin(), ranks.end(), std::greater<int>());

  // Count rank duplicates
  int a=ranks[0], b=ranks[1], c=ranks[2];
  bool trips = (a==b && b==c);
  bool pair = (a==b || b==c || a==c);

  // Suits
  std::array<int,3> suits = {suit_of(hole[0]), suit_of(hole[1]), suit_of(hole[2])};
  int flush_count = 1;
  {
    std::array<int,4> suit_cnt{0,0,0,0};
    for (int s : suits) suit_cnt[s]++;
    flush_count = *std::max_element(suit_cnt.begin(), suit_cnt.end()); // 1..3
  }

  // Straight potential: count consecutive gaps on sorted unique ranks
  std::vector<int> uniq = {a,b,c};
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  int straight_potential = 0;
  if (uniq.size() >= 2) {
    for (size_t i=0;i+1<uniq.size();++i) {
      if (uniq[i] - uniq[i+1] <= 2) straight_potential++;
    }
  }

  int high = a;
  int mid = b;
  int low = c;

  // Heuristic score (matches Python weights)
  int strength = high*2 + mid + low;
  if (trips) strength += 30;
  else if (pair) strength += 15;
  strength += (flush_count - 1) * 8;
  strength += straight_potential * 5;

  // Bucket into 60 bins (same as Python NUM_HOLE_BUCKETS default)
  // Python: bucket = min(NUM_HOLE_BUCKETS-1, strength // 4)
  int bucket = strength / 4;
  if (bucket < 0) bucket = 0;
  if (bucket > 59) bucket = 59;
  return static_cast<uint16_t>(bucket);
}

static std::vector<int> compute_board_features(const std::vector<uint8_t>& board) {
  // Ported from abstraction.py:compute_board_features
  std::vector<int> feats;
  if (board.empty()) {
    feats.assign(5, 0);
    return feats;
  }

  std::vector<int> ranks;
  std::vector<int> suits;
  ranks.reserve(board.size());
  suits.reserve(board.size());
  for (auto c : board) {
    ranks.push_back(rank_of(c));
    suits.push_back(suit_of(c));
  }

  // rank counts
  std::array<int,13> rc{};
  for (int r : ranks) rc[r]++;
  int max_rank_count = 0;
  for (int v : rc) max_rank_count = std::max(max_rank_count, v);

  // flush potential
  std::array<int,4> sc{};
  for (int s : suits) sc[s]++;
  int max_suit_count = 0;
  for (int v : sc) max_suit_count = std::max(max_suit_count, v);

  // straight potential
  std::vector<int> uniq = ranks;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  std::sort(uniq.begin(), uniq.end());
  int straight_potential = 0;
  for (size_t i=0;i<uniq.size();++i) {
    for (size_t j=i+1;j<uniq.size();++j) {
      if (uniq[j] - uniq[i] <= 4) straight_potential = std::max(straight_potential, static_cast<int>(j - i + 1));
    }
  }

  int high_card = *std::max_element(ranks.begin(), ranks.end());

  // broadway count >=10 -> ranks 8..12? Wait rank 8 corresponds to T (2=0,...T=8)
  int broadway = 0;
  for (int r : ranks) if (r >= 8) broadway++;

  feats.push_back(max_rank_count);
  feats.push_back(max_suit_count);
  feats.push_back(straight_potential);
  feats.push_back(high_card);
  feats.push_back(broadway);
  return feats;
}

uint16_t board_bucket(const std::vector<uint8_t>& board) {
  auto feats = compute_board_features(board);
  int bucket = 0;
  bucket += (feats[0] - 1) * 20;
  bucket += (feats[1] - 1) * 8;
  bucket += std::max(0, feats[2] - 2) * 4;
  bucket += feats[4] * 2;
  bucket += feats[3] / 2;
  if (bucket < 0) bucket = 0;
  if (bucket > 79) bucket = 79;
  return static_cast<uint16_t>(bucket);
}

uint8_t pot_bucket(int pot) {
  if (pot <= 4) return 0;
  if (pot <= 10) return 1;
  if (pot <= 25) return 2;
  if (pot <= 60) return 3;
  if (pot <= 140) return 4;
  return 5;
}

uint8_t stack_bucket(int stack) {
  if (stack <= 50) return 0;
  if (stack <= 120) return 1;
  if (stack <= 220) return 2;
  if (stack <= 320) return 3;
  if (stack <= 400) return 4;
  return 5;
}

uint8_t history_bucket(const std::vector<std::pair<int,int>>& hist) {
  // Ported from abstraction.py:get_history_bucket
  const int L = static_cast<int>(hist.size());
  if (L == 0) return 0;
  if (L <= 2) {
    // use last action only bucketed
    int a = hist.back().second;
    return static_cast<uint8_t>(std::min<int>(3, a) + 1);
  }

  int raises = 0;
  for (auto& pa : hist) if (pa.second >= RAISE_SMALL) raises++;

  if (L <= 4) return static_cast<uint8_t>(4 + std::min(3, raises));
  if (raises == 0) return 8;
  if (raises == 1) return 9;
  if (raises == 2) return 10;
  return 11;
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
  InfoKey k{};
  k.player = player;
  k.street = street;

  if (hole_cards.size() >= 3) {
    std::array<uint8_t,3> h{hole_cards[0], hole_cards[1], hole_cards[2]};
    k.hole_bucket = hole_bucket_3card(h);
  } else {
    std::array<uint8_t,2> h{hole_cards[0], hole_cards[1]};
    k.hole_bucket = hole_bucket_2card(h);
  }

  k.board_bucket = board_bucket(board_cards);
  k.pot_bucket = pot_bucket(pot);
  k.stack_bucket = stack_bucket(eff_stack);
  k.hist_bucket = history_bucket(betting_history);
  k.bb_discarded = bb_discarded ? 1 : 0;
  k.sb_discarded = sb_discarded ? 1 : 0;
  k.legal_mask = legal_action_mask;
  return k;
}

std::string info_key_to_string(const InfoKey& k) {
  std::ostringstream oss;
  oss << "P" << int(k.player)
      << "|S" << int(k.street)
      << "|H" << k.hole_bucket
      << "|B" << k.board_bucket
      << "|POT" << int(k.pot_bucket)
      << "|STK" << int(k.stack_bucket)
      << "|HIST" << int(k.hist_bucket)
      << "|BB" << int(k.bb_discarded)
      << "|SB" << int(k.sb_discarded)
      << "|LA" << k.legal_mask;
  return oss.str();
}

} // namespace tossem_abs
