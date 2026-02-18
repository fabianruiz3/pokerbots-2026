# Camello 🐪 — MIT Pokerbots 2026

> *"The Three Wise Men arrive on three camels, and this year's poker variant starts with three cards. Camello was born."*

This repo documents the full journey of building **Camello**, a heads-up poker bot for MIT Pokerbots 2026, from a rough Monte Carlo baseline to a CFR-trained agent with opponent profiling. The game variant this year was **Toss'em Hold'em**: each player receives 3 hole cards pre-flop and must discard one face-up after the flop — a deceptively rich decision point that shapes the entire strategy.

---

## What is Toss'em Hold'em?

Standard Texas Hold'em heads-up, with one key twist: each player is dealt **3 hole cards** instead of 2. After the flop is revealed, each player must **discard one card face-up**, committing to a 2-card hand for the rest of the round.

This changes strategy significantly:
- The discard is public information — your opponent sees which card you let go, which leaks information about your range.
- Pre-flop hand selection now involves reasoning over 3-card combinations rather than 2.
- CFR must account for the discard as an explicit action in the game tree, not just a post-processing step.

---

## Narrative of iterations

### Week 1 Bots — Getting the loop working
The first bots established the basic engine integration (socket protocol, legal action handling, bet sizing). Hand strength estimation was minimal: a small Monte Carlo simulation over the 3-card variant with no discard modeling. These bots folded too much and had no awareness of what the discard decision revealed.

### Week 2 Bots — Richer heuristics and discard awareness
This generation added proper 3-card pre-flop hand evaluation, basic pot-odds/EV computation, and early discard logic. The bots started reasoning about which card to discard not just based on raw strength, but on what the revealed card would signal to the opponent. Bet sizing became dynamic by street.

### Camello 3.0.0 — Monte Carlo + opponent tracking
The first "named" version introduced a full Monte Carlo hand strength estimator tuned for Toss'em Hold'em, running simulations that enumerate possible opponent holdings and board runouts while accounting for the discard mechanic. It also added basic **opponent profiling**: tracking aggression factor, call frequency, and fold-to-continuation-bet rates to adjust thresholds at runtime.

### Camello 3.1.0 — Refined exploitability
Built on 3.0 with cleaner pre-flop tiering, improved discard EV logic, and a more principled exploitation layer. The opponent model was extended to categorize opponents as tight/loose/aggressive and gate bet sizing and bluff frequency accordingly. Monte Carlo accuracy improved with more rollouts and better card removal handling.

### Camello CFR — Final competition bot
The strongest version. Replaced the hand-evaluation core with a **CFR-trained policy** that treats the discard as an explicit action in the game tree. The bot uses CFR-derived strategies for pre-flop, discard, and post-flop decisions, falling back to Monte Carlo simulation for scenarios underrepresented in training. Opponent profiling from earlier iterations is retained and layered on top of the CFR strategy as a real-time exploitability adjustment.

---

## Repo layout

| Path | Description |
|---|---|
| `engine.py` | Local engine to run two bots head-to-head via socket protocol |
| `config.py` | Match configuration: players, stack sizes, blinds, timeouts |
| `Week_1_Bots/` | Initial baseline bots |
| `Week_2_Bots/` | Heuristic-improved generation with discard awareness |
| `Camello_3.0.0/` | Monte Carlo bot with opponent profiling |
| `Camello_3.1.0/` | Refined exploitability and opponent model |
| `Camello_cfr/` | **Final competition bot** — CFR-trained + Monte Carlo fallback + profiling |
| `Camello_cfr_old/` | Earlier CFR experiments |
| `tossem_cfr_cpp/` | C++ CFR training code for Toss'em Hold'em |
| `player_chatbot/` | Interactive human-in-the-loop client (optional GPT backend) |
| `python_skeleton/` | Provided Python skeleton with protocol helpers |
| `cpp_skeleton/` | Provided C++ skeleton |
| `java_skeleton/` | Provided Java skeleton |

---

## Run a match

**Prerequisites:** Python 3.8+, with dependencies managed via [`uv`](https://docs.astral.sh/uv/).

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After cloning the repo:

```bash
# Create virtual environment and install dependencies
uv venv
uv sync
```

Then configure the match in `config.py`:

```python
PLAYER_1_PATH = "./Camello_cfr"
PLAYER_2_PATH = "./python_skeleton"   # swap in any bot
```

And run:

```bash
python engine.py
```

Logs are written to `gamelog_summary.txt` and per-bot logs.

---

## The final bot: Camello CFR

Camello CFR is the competition-ready agent. Its decision-making combines three layers:

**1. CFR-trained strategy**
The discard decision and core betting lines were trained using Counterfactual Regret Minimization on an abstracted Toss'em Hold'em game tree. The CFR abstraction buckets hands by strength (pre-flop), flop texture, and discard value, producing near-Nash strategies for the most common game situations.

**2. Monte Carlo hand strength**
For post-discard streets where the exact opponent range is uncertain, the bot runs Monte Carlo rollouts to estimate equity against a range conditioned on the opponent's revealed discard and observed betting behavior.

**3. Opponent profiling**
Real-time opponent statistics (aggression factor, VPIP, fold-to-cbet) are tracked across hands and used to shift bet sizing and bluff frequency away from the baseline CFR strategy toward a more exploitative line when a clear pattern is detected.

---

## Building your own bot

1. Copy any bot directory (e.g. `Camello_3.1.0/`) to `my_bot/`.
2. Keep the `skeleton/` subdirectory — it handles the socket protocol.
3. Implement your logic in `player.py`.
4. Make sure `commands.json` exists and points to `python3 player.py`.
5. Set `PLAYER_1_PATH = "./my_bot"` in `config.py` and run `engine.py`.

---

## Setup: C++ bots

Requires `C++17`, `cmake >= 3.8`, and `boost`.

```bash
# Linux
sudo apt-get install -y libboost-all-dev

# macOS
brew install boost

# Windows (must use WSL)
wsl --install
sudo apt update && sudo apt install -y libboost-all-dev
```

Verify your compiler and cmake versions:

```bash
cmake --version      # should be 3.8+
g++ --version        # GCC 10+ or clang 10+
```

> On macOS, if boost isn't auto-detected after `brew install boost`, use:
> `cmake -DBOOST_ROOT=/opt/homebrew -DCMAKE_BUILD_TYPE=Debug ..`

---

## Setup: Java bots

Requires Java 8+. Verify with `java -version`.

**macOS:**
```bash
brew install --cask temurin
# If java isn't found after install:
echo 'export PATH="/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home/bin:$PATH"' >> ~/.zshrc
```

**Linux:**
```bash
sudo apt update && sudo apt install -y openjdk-17-jdk
```

**Windows:** Download the `.msi` installer from [Adoptium](https://adoptium.net) and ensure "Add to PATH" is checked.

---

*MIT Pokerbots 2026 — Happy Three Kings Day. 🐪🐪🐪*
sudo apt install -y openjdk-17-jdk
```

#### Windows

You can download manually from: [Adoptium](https://adoptium.net), install using the `.msi` installer, and make sure "Add to PATH" is checked.
