#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "abstraction.h"
#include "game_state.h"

using Clock = std::chrono::steady_clock;

struct Node {
  std::array<double, tossem_abs::NUM_ACTIONS> regret{{0,0,0,0}};
  std::array<double, tossem_abs::NUM_ACTIONS> strat_sum{{0,0,0,0}};
};

using Table = std::unordered_map<tossem_abs::InfoKey, Node, tossem_abs::InfoKeyHash>;

static std::array<double, tossem_abs::NUM_ACTIONS> regret_match(const Node& node, const std::vector<int>& legal) {
  std::array<double, tossem_abs::NUM_ACTIONS> s{{0,0,0,0}};
  double norm = 0.0;
  for (int a : legal) {
    if (0 <= a && a < tossem_abs::NUM_ACTIONS) {
      s[a] = std::max(0.0, node.regret[a]);
      norm += s[a];
    }
  }
  if (norm > 0.0) {
    for (int a : legal) if (0 <= a && a < tossem_abs::NUM_ACTIONS) s[a] /= norm;
  } else {
    // uniform over legal
    double u = 1.0 / std::max<size_t>(1, legal.size());
    for (int a : legal) if (0 <= a && a < tossem_abs::NUM_ACTIONS) s[a] = u;
  }
  return s;
}

static double cfr_traverse(game::GameState& st,
                           int update_player,
                           double pi0, double pi1,
                           std::mt19937& rng,
                           Table& table) {
  if (st.is_terminal) {
    return st.payoffs[update_player];
  }

  int player = st.current_player;
  auto legal = st.legal_actions();

  // Discard phase: treat as uniform, do not learn regrets.
  if (st.is_discard_phase()) {
    if (player == update_player) {
      double total = 0.0;
      for (int a : legal) {
        game::Undo u;
        st.apply_action(a, u);
        total += cfr_traverse(st, update_player, pi0, pi1, rng, table) / static_cast<double>(legal.size());
        st.undo_action(u);
      }
      return total;
    } else {
      std::uniform_int_distribution<int> d(0, static_cast<int>(legal.size()) - 1);
      int a = legal[d(rng)];
      game::Undo u;
      st.apply_action(a, u);
      double v = cfr_traverse(st, update_player, pi0, pi1, rng, table);
      st.undo_action(u);
      return v;
    }
  }

  // Betting node
  tossem_abs::InfoKey key = st.info_key(player, legal);
  Node& node = table[key]; // default-inits if absent

  auto strat = regret_match(node, legal);

  // Update average strategy sum
  double reach = (player == 0) ? pi0 : pi1;
  for (int a : legal) {
    if (0 <= a && a < tossem_abs::NUM_ACTIONS) node.strat_sum[a] += reach * strat[a];
  }

  // Use FULL TRAVERSAL at preflop (street 0) for better coverage
  // Use external sampling for later streets to keep computation tractable
  bool use_full_traversal = (st.street == tossem_abs::STREET_PREFLOP);

  if (player == update_player || use_full_traversal) {
    // Full traversal: explore all actions
    std::array<double, tossem_abs::NUM_ACTIONS> action_values{{0,0,0,0}};
    for (int a : legal) {
      game::Undo u;
      st.apply_action(a, u);
      if (player == 0) action_values[a] = cfr_traverse(st, update_player, pi0 * strat[a], pi1, rng, table);
      else action_values[a] = cfr_traverse(st, update_player, pi0, pi1 * strat[a], rng, table);
      st.undo_action(u);
    }

    double node_value = 0.0;
    for (int a : legal) node_value += strat[a] * action_values[a];

    // Only update regrets if this is the update player
    if (player == update_player) {
      for (int a : legal) {
        if (0 <= a && a < tossem_abs::NUM_ACTIONS) node.regret[a] += (action_values[a] - node_value);
      }
    }

    return node_value;
  } else {
    // External sampling: sample opponent action (for non-preflop streets)
    std::vector<double> probs;
    probs.reserve(legal.size());
    double sum = 0.0;
    for (int a : legal) {
      double p = (0 <= a && a < tossem_abs::NUM_ACTIONS) ? strat[a] : 0.0;
      probs.push_back(p);
      sum += p;
    }
    if (sum <= 0.0) {
      for (double& p : probs) p = 1.0 / static_cast<double>(probs.size());
    } else {
      for (double& p : probs) p /= sum;
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int idx = dist(rng);
    int a = legal[idx];

    game::Undo u;
    st.apply_action(a, u);
    double v;
    if (player == 0) v = cfr_traverse(st, update_player, pi0 * strat[a], pi1, rng, table);
    else v = cfr_traverse(st, update_player, pi0, pi1 * strat[a], rng, table);
    st.undo_action(u);
    return v;
  }
}

struct WorkerResult {
  Table table;
  int64_t iters = 0;
};

static WorkerResult run_worker(int64_t iters, uint32_t seed) {
  WorkerResult r;
  r.iters = iters;
  std::mt19937 rng(seed);
  game::GameState st;
  for (int64_t i = 0; i < iters; ++i) {
    st.reset(rng);
    for (int p = 0; p < 2; ++p) {
      cfr_traverse(st, p, 1.0, 1.0, rng, r.table);
    }
  }
  return r;
}

static void merge_into(Table& dst, const Table& src) {
  for (const auto& kv : src) {
    Node& d = dst[kv.first];
    const Node& s = kv.second;
    for (int a = 0; a < tossem_abs::NUM_ACTIONS; ++a) {
      d.regret[a] += s.regret[a];
      d.strat_sum[a] += s.strat_sum[a];
    }
  }
}

// V2 binary format: 75 bytes per node (no stack_bucket)
// Header: magic(4) + version(4) + iterations(8) + num_nodes(8) = 24 bytes
// Per node: key(9 bytes) + regret(32) + strat_sum(32) + reserved(2) = 75 bytes
static void save_binary_v2(const std::string& path, const Table& table, int64_t iterations) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "ERROR: Could not open output file: " << path << std::endl;
    std::exit(1);
  }

  // Header
  const uint32_t magic = 0x544F5353; // 'TOSS'
  const uint32_t version = 2;  // V2 format
  out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char*>(&version), sizeof(version));
  out.write(reinterpret_cast<const char*>(&iterations), sizeof(iterations));
  uint64_t n = static_cast<uint64_t>(table.size());
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));

  // Rows - 75 bytes each
  for (const auto& kv : table) {
    const tossem_abs::InfoKey& k = kv.first;
    const Node& node = kv.second;

    // Key: 9 bytes (no stack_bucket)
    out.write(reinterpret_cast<const char*>(&k.player), sizeof(k.player));           // 1
    out.write(reinterpret_cast<const char*>(&k.street), sizeof(k.street));           // 1
    out.write(reinterpret_cast<const char*>(&k.hole_bucket), sizeof(k.hole_bucket)); // 2
    out.write(reinterpret_cast<const char*>(&k.board_bucket), sizeof(k.board_bucket)); // 2
    out.write(reinterpret_cast<const char*>(&k.pot_bucket), sizeof(k.pot_bucket));   // 1
    out.write(reinterpret_cast<const char*>(&k.hist_bucket), sizeof(k.hist_bucket)); // 1
    
    // Pack bb_discarded, sb_discarded, legal_mask into combined byte
    uint8_t flags = (k.bb_discarded ? 0x80 : 0) | (k.sb_discarded ? 0x40 : 0) | (k.legal_mask & 0x3F);
    out.write(reinterpret_cast<const char*>(&flags), sizeof(flags));                 // 1 = 9 total

    // Data: 64 bytes
    out.write(reinterpret_cast<const char*>(node.regret.data()), sizeof(double) * tossem_abs::NUM_ACTIONS);     // 32
    out.write(reinterpret_cast<const char*>(node.strat_sum.data()), sizeof(double) * tossem_abs::NUM_ACTIONS);  // 32

    // Reserved: 2 bytes for future use
    uint16_t reserved = 0;
    out.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));           // 2 = 75 total
  }
  
  std::cout << "Saved " << path << " (v2 format, " << table.size() << " nodes, " << iterations << " iters)\n";
}

static int64_t parse_i64(const char* s) {
  return static_cast<int64_t>(std::stoll(std::string(s)));
}

int main(int argc, char** argv) {
  int64_t iters = 1000000;
  int threads = std::max(1u, std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() - 1 : 1u);
  int64_t batch = 20000;
  int64_t checkpoint_interval = 500000;
  std::string out_path = "cfr_strategy.bin";

  for (int i=1;i<argc;++i) {
    std::string a(argv[i]);
    if ((a=="-i" || a=="--iters") && i+1<argc) iters = parse_i64(argv[++i]);
    else if ((a=="-t" || a=="--threads") && i+1<argc) threads = std::stoi(argv[++i]);
    else if ((a=="-b" || a=="--batch") && i+1<argc) batch = parse_i64(argv[++i]);
    else if ((a=="-c" || a=="--checkpoint") && i+1<argc) checkpoint_interval = parse_i64(argv[++i]);
    else if ((a=="-o" || a=="--out") && i+1<argc) out_path = argv[++i];
    else if (a=="-h" || a=="--help") {
      std::cout << "Usage: train_mccfr [-i iters] [-t threads] [-b batch] [-c checkpoint] [-o out.bin]\n";
      std::cout << "  -i, --iters       Total iterations (default: 1000000)\n";
      std::cout << "  -t, --threads     Number of threads (default: auto)\n";
      std::cout << "  -b, --batch       Batch size per thread (default: 20000)\n";
      std::cout << "  -c, --checkpoint  Checkpoint interval (default: 500000)\n";
      std::cout << "  -o, --out         Output file (default: cfr_strategy.bin)\n";
      return 0;
    }
  }

  std::cout << "Toss'em Hold'em MCCFR (C++ standalone) - V2 Format\n";
  std::cout << "Streets: 0=PREFLOP, 2=BB_DISCARD, 3=SB_DISCARD, 4=FLOP_BET, 5=TURN, 6=RIVER\n";
  std::cout << "iters=" << iters << " threads=" << threads << " batch=" << batch 
            << " checkpoint=" << checkpoint_interval << "\n";

  Table global;
  int64_t done = 0;
  int64_t last_checkpoint = 0;
  auto t0 = Clock::now();

  std::random_device rd;

  while (done < iters) {
    int64_t remaining = iters - done;
    int64_t per = std::max<int64_t>(1, std::min<int64_t>(batch, remaining / threads + 1));

    std::vector<std::thread> ts;
    std::vector<WorkerResult> results(threads);

    auto b0 = Clock::now();
    for (int w=0; w<threads; ++w) {
      uint32_t seed = static_cast<uint32_t>(rd()) ^ static_cast<uint32_t>(done + w*1337);
      results[w].iters = per;
      ts.emplace_back([&, w, seed](){ results[w] = run_worker(per, seed); });
    }
    for (auto& t : ts) t.join();

    int64_t batch_done = 0;
    for (const auto& r : results) {
      batch_done += r.iters;
      merge_into(global, r.table);
    }

    done += batch_done;

    auto b1 = Clock::now();
    double sec = std::chrono::duration<double>(b1 - b0).count();
    double rate = batch_done / std::max(1e-9, sec);
    double total_sec = std::chrono::duration<double>(b1 - t0).count();
    double total_rate = done / std::max(1e-9, total_sec);

    std::cout << "  " << done << "/" << iters << "  rate=" << static_cast<long long>(rate)
              << "/s total=" << static_cast<long long>(total_rate)
              << "/s states=" << global.size() << "\n";
    
    // Checkpoint
    if (done - last_checkpoint >= checkpoint_interval) {
      std::string cp_path = out_path + ".checkpoint_" + std::to_string(done/1000) + "k";
      save_binary_v2(cp_path, global, done);
      last_checkpoint = done;
    }
  }

  save_binary_v2(out_path, global, done);
  return 0;
}
