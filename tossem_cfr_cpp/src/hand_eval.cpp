#include "hand_eval.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>

namespace eval {

static inline int rank(uint8_t c) { return c / 4; }
static inline int suit(uint8_t c) { return c % 4; }

static HandValue eval_five(const std::array<uint8_t,5>& cards) {
  std::array<int,5> ranks;
  std::array<int,5> suits;
  for (int i=0;i<5;++i) { ranks[i]=rank(cards[i]); suits[i]=suit(cards[i]); }
  std::sort(ranks.begin(), ranks.end(), std::greater<int>());

  bool is_flush = std::all_of(suits.begin(), suits.end(), [&](int s){ return s==suits[0]; });

  // count ranks
  std::array<int,13> cnt{};
  for (int r: ranks) cnt[r]++;
  std::vector<std::pair<int,int>> groups; // (count, rank)
  groups.reserve(5);
  for (int r=12;r>=0;--r) if (cnt[r]>0) groups.push_back({cnt[r], r});
  std::sort(groups.begin(), groups.end(), [](auto a, auto b){
    if (a.first!=b.first) return a.first>b.first;
    return a.second>b.second;
  });

  // unique ranks for straight detection
  std::vector<int> uniq;
  uniq.reserve(5);
  for (int r: ranks) if (uniq.empty() || uniq.back()!=r) uniq.push_back(r);

  bool is_straight = false;
  int straight_high = 0;
  if (uniq.size()==5) {
    if (uniq[0]-uniq[4]==4) { is_straight=true; straight_high=uniq[0]; }
    else if (uniq[0]==12 && uniq[1]==3 && uniq[2]==2 && uniq[3]==1 && uniq[4]==0) {
      is_straight=true; straight_high=3; // 5-high straight
    }
  }

  HandValue hv;

  if (is_straight && is_flush) {
    hv.type = 8;
    hv.kickers = {straight_high,0,0,0,0};
    return hv;
  }
  if (groups[0].first==4) {
    hv.type = 7;
    int quad = groups[0].second;
    int kick = groups[1].second;
    hv.kickers = {quad,kick,0,0,0};
    return hv;
  }
  if (groups[0].first==3 && groups.size()>1 && groups[1].first==2) {
    hv.type = 6;
    hv.kickers = {groups[0].second, groups[1].second, 0,0,0};
    return hv;
  }
  if (is_flush) {
    hv.type = 5;
    hv.kickers = {ranks[0],ranks[1],ranks[2],ranks[3],ranks[4]};
    return hv;
  }
  if (is_straight) {
    hv.type = 4;
    hv.kickers = {straight_high,0,0,0,0};
    return hv;
  }
  if (groups[0].first==3) {
    hv.type = 3;
    int trip = groups[0].second;
    // remaining singles in desc order
    std::vector<int> singles;
    for (auto &g: groups) if (g.first==1) singles.push_back(g.second);
    std::sort(singles.begin(), singles.end(), std::greater<int>());
    hv.kickers = {trip, singles[0], singles[1], 0,0};
    return hv;
  }
  if (groups[0].first==2 && groups.size()>1 && groups[1].first==2) {
    hv.type = 2;
    int p1 = groups[0].second;
    int p2 = groups[1].second;
    int kick = -1;
    for (auto &g: groups) if (g.first==1) { kick=g.second; break; }
    hv.kickers = {std::max(p1,p2), std::min(p1,p2), kick, 0,0};
    return hv;
  }
  if (groups[0].first==2) {
    hv.type = 1;
    int pair = groups[0].second;
    std::vector<int> singles;
    for (auto &g: groups) if (g.first==1) singles.push_back(g.second);
    std::sort(singles.begin(), singles.end(), std::greater<int>());
    hv.kickers = {pair, singles[0], singles[1], singles[2], 0};
    return hv;
  }
  hv.type = 0;
  hv.kickers = {ranks[0],ranks[1],ranks[2],ranks[3],ranks[4]};
  return hv;
}

HandValue evaluate_best(const std::vector<uint8_t>& cards) {
  const int n = (int)cards.size();
  if (n < 5) {
    HandValue hv;
    hv.type = 0;
    std::vector<int> rs;
    rs.reserve(n);
    for (auto c: cards) rs.push_back(rank(c));
    std::sort(rs.begin(), rs.end(), std::greater<int>());
    for (size_t i=0;i<rs.size() && i<5;i++) hv.kickers[i]=rs[i];
    return hv;
  }

  HandValue best;
  best.type = -1;

  // brute choose 5 of n
  for (int a=0; a<n-4; ++a)
  for (int b=a+1; b<n-3; ++b)
  for (int c=b+1; c<n-2; ++c)
  for (int d=c+1; d<n-1; ++d)
  for (int e=d+1; e<n; ++e) {
    std::array<uint8_t,5> five{{cards[a],cards[b],cards[c],cards[d],cards[e]}};
    HandValue hv = eval_five(five);
    if (best.type<0 || hv > best) best = hv;
  }

  return best;
}

int compare(const HandValue& a, const HandValue& b) {
  if (a.type != b.type) return a.type > b.type ? 1 : -1;
  for (size_t i=0;i<a.kickers.size();++i) {
    if (a.kickers[i] != b.kickers[i]) return a.kickers[i] > b.kickers[i] ? 1 : -1;
  }
  return 0;
}

} // namespace eval
