"""Hand-written opponent-model programs (sandbox-compatible) mirroring the bot families.

Used for parameter-recovery tests and as an optional starting library. The LLM-proposed programs
follow the same contract: `model(history, params, game) -> probabilities over opponent actions`.
"""
from __future__ import annotations

from .agents import ProgramSpec

UNIFORM = '''
def model(history, params, game):
    n = game["n_opp"]
    return np.ones(n) / n
'''

HABIT = '''
def model(history, params, game):
    """Opponent repeats its own recent action frequencies (Laplace-smoothed, window w)."""
    n = game["n_opp"]
    w = int(params["w"])
    counts = np.ones(n)
    for pair in history[-w:]:
        counts[pair[1]] += 1.0
    return counts / counts.sum()
'''

REACTIVE = '''
def model(history, params, game):
    """Opponent's action depends on the agent's previous action; learn that table from the history."""
    n = game["n_opp"]
    alpha = float(params["alpha"])
    if len(history) == 0:
        return np.ones(n) / n
    last = history[-1][0]
    counts = np.full(n, alpha)
    for t in range(1, len(history)):
        if history[t - 1][0] == last:
            counts[history[t][1]] += 1.0
    return counts / counts.sum()
'''

WSLS = '''
def model(history, params, game):
    """Win-stay lose-shift: repeat after a payoff above the mean payoff, otherwise switch uniformly."""
    n = game["n_opp"]
    eps = float(params["eps"])
    uni = np.ones(n) / n
    if len(history) == 0:
        return uni
    pay = np.array(game["payoff_opp"], dtype=float)
    a, o = history[-1]
    if pay[a][o] > pay.mean():
        det = np.zeros(n)
        det[o] = 1.0
    else:
        det = np.ones(n)
        det[o] = 0.0
        det = det / det.sum()
    return (1 - eps) * det + eps * uni
'''

KLEVEL = '''
def softmax(u, tau):
    z = (u - u.max()) / max(tau, 1e-6)
    e = np.exp(z)
    return e / e.sum()


def habit(actions, n, w):
    counts = np.ones(n)
    for x in actions[-w:]:
        counts[x] += 1.0
    return counts / counts.sum()


def model(history, params, game):
    """Level-k: level-0 = windowed habit; level-k softmax-best-responds to the other side at level k-1."""
    k = int(params["k"])
    w = int(params["w"])
    tau = float(params["tau"])
    pa = np.array(game["payoff_agent"], dtype=float)
    po = np.array(game["payoff_opp"], dtype=float)
    mine = [h[0] for h in history]
    theirs = [h[1] for h in history]

    def level(j, role):
        if role == "opp":
            if j == 0:
                return habit(theirs, game["n_opp"], w)
            return softmax(level(j - 1, "agent") @ po, tau)
        if j == 0:
            return habit(mine, game["n_agent"], w)
        return softmax(pa @ level(j - 1, "opp"), tau)

    return level(k, "opp")
'''

CYCLE = '''
def model(history, params, game):
    """Opponent cycles with a fixed period: predict its action from one period ago."""
    n = game["n_opp"]
    p = int(params["period"])
    eps = float(params["eps"])
    uni = np.ones(n) / n
    t = len(history)
    if t < p:
        return uni
    det = np.zeros(n)
    det[history[t - p][1]] = 1.0
    return (1 - eps) * det + eps * uni
'''

SEED_LIBRARY: list[ProgramSpec] = [
    ProgramSpec("habit", HABIT, {"w": [5, 20, 1000]}, "repeats its own recent action frequencies"),
    ProgramSpec("reactive", REACTIVE, {"alpha": [0.1, 0.5, 1.0]}, "responds to the agent's previous action"),
    ProgramSpec("wsls", WSLS, {"eps": [0.02, 0.1, 0.2, 0.4]}, "win-stay lose-shift"),
    ProgramSpec(
        "klevel", KLEVEL,
        {"k": [1, 2, 3], "w": [5, 10, 20, 40], "tau": [0.05, 0.1, 0.2, 0.35, 0.5, 1.0]},
        "level-k reasoning over windowed habits (k=1: fictitious play)",
    ),
    ProgramSpec("cycle", CYCLE, {"period": [2, 3, 4, 5], "eps": [0.05, 0.15, 0.3]}, "fixed-period cycle"),
]

MINIMAL_LIBRARY: list[ProgramSpec] = [ProgramSpec("uniform", UNIFORM, {}, "opponent plays uniformly at random")]


# ---------------------------------------------------------------- v5: true-class programs of the families outside the menu
# Used only by the oracle arm and scripts/gap_check.py (never shown to the LLM). Unknown constants of the bot
# (start action, response table, base distribution) are learned from the history or are grid parameters.

LOSE_K_SHIFT = '''
def model(history, params, game):
    """Holds an action; after k consecutive non-wins switches to its best reply to the agent's action that round."""
    n = game["n_opp"]
    k = int(params["k"])
    eps = float(params["eps"])
    pay = np.array(game["payoff_opp"], dtype=float)
    mean = pay.mean()
    cur = int(params["start"]) % n
    losses = 0
    for pair in history:
        a, o = pair
        losses = 0 if pay[a][o] > mean else losses + 1
        if losses >= k:
            cur = int(np.argmax(pay[a]))
            losses = 0
    det = np.zeros(n)
    det[cur] = 1.0
    return (1 - eps) * det + eps * np.ones(n) / n
'''

REACT2 = '''
def model(history, params, game):
    """Opponent responds to the agent's last two actions; learn that table from the history."""
    n = game["n_opp"]
    alpha = float(params["alpha"])
    t = len(history)
    if t < 2:
        return np.ones(n) / n
    key = (history[t - 2][0], history[t - 1][0])
    counts = np.full(n, alpha)
    for r in range(2, t):
        if (history[r - 2][0], history[r - 1][0]) == key:
            counts[history[r][1]] += 1.0
    return counts / counts.sum()
'''

SEQ_BR = '''
def model(history, params, game):
    """Softmax best response to the agent's next action predicted from its transition counts (Laplace)."""
    na = game["n_agent"]
    tau = float(params["tau"])
    if len(history) == 0:
        pred = np.ones(na) / na
    else:
        row = np.ones(na)
        last = history[-1][0]
        for t in range(1, len(history)):
            if history[t - 1][0] == last:
                row[history[t][0]] += 1.0
        pred = row / row.sum()
    u = pred @ np.array(game["payoff_opp"], dtype=float)
    z = np.exp((u - u.max()) / tau)
    return z / z.sum()
'''

BLOCK_DRIFT = '''
def model(history, params, game):
    """A base distribution whose labels rotate by one every `block` rounds; the base is learned un-rotated."""
    n = game["n_opp"]
    b = int(params["block"])
    alpha = float(params["alpha"])
    counts = np.full(n, alpha)
    for r in range(len(history)):
        counts[(history[r][1] - r // b) % n] += 1.0
    q = counts / counts.sum()
    return np.roll(q, len(history) // b)
'''

NOVEL_PROGRAMS: dict[str, ProgramSpec] = {
    "lose_k_shift": ProgramSpec("lose_k_shift", LOSE_K_SHIFT,
                                # dev and test eps values; start is cut to the game's actions by propose.oracle_specs
                                {"k": [2, 3, 4], "eps": [0.05, 0.1, 0.15, 0.2, 0.3], "start": [0, 1, 2, 3, 4]},
                                "holds an action and switches to its best reply after k consecutive non-wins"),
    "react2": ProgramSpec("react2", REACT2, {"alpha": [0.1, 0.5, 1.0]}, "responds to the agent's last two actions"),
    "seq_br": ProgramSpec("seq_br", SEQ_BR, {"tau": [0.05, 0.1, 0.2, 0.35, 0.5, 1.0]},
                          "softmax best response to the agent's predicted next action (transition counts)"),
    "block_drift": ProgramSpec("block_drift", BLOCK_DRIFT, {"block": [6, 8, 10, 12, 15, 20], "alpha": [0.1, 0.5]},
                               "a fixed distribution whose action labels rotate every `block` rounds"),
}


# ---------------------------------------------------------------- template proposer menu (no-LLM control)
# Generic programs for the kinds of structure the rich proposer prompt (propose.TASK_RICH) asks the LLM for.
# They are written once, without reference to any bot, and offered by `make_template_proposer` at the trigger.

PHASE_TABLE = '''
def model(history, params, game):
    """Learned phase table: P(opponent action | t mod p), Laplace-smoothed counts of past rounds in that phase."""
    n = game["n_opp"]
    p = int(params["period"])
    alpha = float(params["alpha"])
    t = len(history)
    counts = np.full(n, alpha)
    for r in range(t % p, t, p):
        counts[history[r][1]] += 1.0
    return counts / counts.sum()
'''

LAST2_TABLE = '''
def model(history, params, game):
    """Learned table P(opponent next | opponent's last two actions), Laplace-smoothed."""
    n = game["n_opp"]
    alpha = float(params["alpha"])
    counts = np.full(n, alpha)
    t = len(history)
    if t < 2:
        return counts / counts.sum()
    key = (history[t - 2][1], history[t - 1][1])
    for r in range(2, t):
        if (history[r - 2][1], history[r - 1][1]) == key:
            counts[history[r][1]] += 1.0
    return counts / counts.sum()
'''

SOFTMAX_BR = '''
def model(history, params, game):
    """Softmax best response to the agent's action frequencies over the last w rounds."""
    w = int(params["w"])
    tau = float(params["tau"])
    counts = np.ones(game["n_agent"])
    for pair in history[-w:]:
        counts[pair[0]] += 1.0
    u = (counts / counts.sum()) @ np.array(game["payoff_opp"], dtype=float)
    z = np.exp((u - u.max()) / tau)
    return z / z.sum()
'''

TEMPLATE_MENUS: dict[str, list[ProgramSpec]] = {
    # the structures the longer-range evidence views expose (lag-k repeats, t mod p tables, last-two table)
    "periodic": [
        ProgramSpec("tpl_lag", CYCLE, {"period": [2, 3, 4, 5, 6], "eps": [0.05, 0.15, 0.3]},
                    "repeats its own action from `period` rounds ago"),
        ProgramSpec("tpl_phase", PHASE_TABLE, {"period": [2, 3, 4, 5, 6], "alpha": [0.1, 0.5]},
                    "learned action distribution per phase t mod period"),
        ProgramSpec("tpl_last2", LAST2_TABLE, {"alpha": [0.1, 0.5, 1.0]},
                    "learned response to its own last two actions"),
    ],
}
# plus the strategic and outcome-rule items of the same prompt
TEMPLATE_MENUS["full"] = TEMPLATE_MENUS["periodic"] + [
    ProgramSpec("tpl_softbr", SOFTMAX_BR, {"w": [5, 10, 20, 40], "tau": [0.05, 0.1, 0.2, 0.5, 1.0]},
                "softmax best response to the agent's recent action frequencies"),
    ProgramSpec("tpl_wsls", WSLS, {"eps": [0.02, 0.1, 0.2, 0.4]}, "win-stay lose-shift"),
]
