#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace eval {

// Card: uint8_t = rank*4 + suit, rank 0=2 .. 12=A, suit 0..3.

struct HandValue {
  int type = 0; // 0..8 (high card..straight flush)
  std::array<int,5> kickers{{0,0,0,0,0}};

  bool operator>(const HandValue& o) const {
    if (type != o.type) return type > o.type;
    for (size_t i = 0; i < kickers.size(); ++i) {
      if (kickers[i] != o.kickers[i]) return kickers[i] > o.kickers[i];
    }
    return false;
  }
};

// Evaluate best 5-card hand from N cards (N>=5 typical).
HandValue evaluate_best(const std::vector<uint8_t>& cards);

// Compare two hand values: 1 if a wins, -1 if b wins, 0 tie.
int compare(const HandValue& a, const HandValue& b);

} // namespace eval
