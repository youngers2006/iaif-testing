import argparse
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from agent import AIfAgent
from bayesian_thermostat import BayesianThermometer, load_config

HERE = Path(__file__).resolve().parent


def simulate(agent_cfg, env_cfg, replan_mode, n_steps, seed, meta_params=None):
    """Run one closed-loop simulation. The plant's random stream does not depend on the actions, so runs with
    the same seed see the same process and sensor noise."""
    planner = {**agent_cfg["planner"], "replan_mode": replan_mode}
    agent = AIfAgent(agent_cfg["belief_prior"], agent_cfg["generative_model"], planner, seed=seed, meta_params=meta_params)
    env = BayesianThermometer(env_cfg["plant"], jax.random.key(seed + 1))
    x = env_cfg["x_init"]
    xs, x_obs, us, planned, wall = [], [], [], [], []
    for _ in range(n_steps):
        x_obs.append(np.asarray(x))                       # the state the agent observes and acts on at this step
        y = env.observe(x)
        t0 = time.perf_counter()
        u, info = agent.act(y)
        jax.block_until_ready(u)
        wall.append(time.perf_counter() - t0)
        planned.append(bool(info["replanned"]))
        x = env.step(x, u)
        xs.append(np.asarray(x))
        us.append(np.asarray(u))
    return {"agent": agent, "x": np.array(xs), "x_obs": np.array(x_obs), "u": np.array(us),
            "planned": np.array(planned), "wall": np.array(wall)}


def behaviour_metrics(run, burn_in):
    agent, n_steps = run["agent"], len(run["x"])
    y_true = run["x"] @ np.asarray(agent.generative_model.C).T              # noiseless sensor reading
    c = agent.compute
    return {
        "mean |y - y*|": np.mean(np.abs(y_true[burn_in:] - np.asarray(agent.y_star))),
        "mean u": np.mean(run["u"][burn_in:]),
        "plans / step": c["plan_calls"] / n_steps,
        "EFE step-evals / step": (c["plan_step_evals"] + c["trigger_step_evals"]) / n_steps,
        "ms / step": 1e3 * np.median(run["wall"][5:]),
    }


def closed_events(agent):
    """Plans whose fate is known: outcome index k-1 for a first trigger at depth k, H-1 for no trigger."""
    H, out = agent.H, []
    for ev in agent.events:
        if ev["status"] != "open":
            out.append((ev, ev["K_P"] - 1 if ev["K_P"] is not None else H - 1))
    return out


def predictor_metrics(events, method, H):
    P = np.array([np.r_[np.asarray(ev["predictions"][method].pmf), float(ev["predictions"][method].p_none)] for ev, _ in events])
    o = np.array([oi for _, oi in events])
    obs = np.eye(H)[o]
    lifetimes = np.arange(1, H + 1)                                        # executed actions: k, or H if no trigger
    rps = np.mean(np.sum((np.cumsum(P, 1) - np.cumsum(obs, 1))[:, :-1] ** 2, 1) / max(H - 1, 1))
    haz = []
    for k in range(1, H):
        reached = o >= k - 1
        pred_h = np.mean([float(ev["predictions"][method].hazard[k - 1]) for (ev, oi) in events if oi >= k - 1]) if reached.any() else np.nan
        emp_h = np.mean(o[reached] == k - 1) if reached.any() else np.nan
        haz.append((k, int(reached.sum()), pred_h, emp_h))
    return {
        "log score": np.mean(np.log(np.clip(P[np.arange(len(o)), o], 1e-12, None))),
        "RPS": rps,
        "E[lifetime] predicted": np.mean(P @ lifetimes),
        "mean pmf": P.mean(0),
        "hazards": haz,
    }, P


INK, INK_2, GRID, AXIS, CONTEXT = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#c9c9c9"
COLOURS = {"every_step": "#257de0", "efe_trigger": "#f47230"}          # categorical slots 1 and 2 (validated)
TITLES = {"every_step": "Replanning every step", "efe_trigger": "IAIF (EFE Error Trigger)"}


def plot_error(runs, seed, start, n_show, path):
    """Tracking error |C x - y*| against step for both modes, with a dot at every step where the mode planned.
    One panel per mode on a shared y-axis; the other mode's error is drawn in grey for comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("\n  plot skipped: matplotlib is not installed (pip install matplotlib)")
        return
    modes = ("every_step", "efe_trigger")
    err = {}
    for mode in modes:
        run, agent = runs[mode][seed], runs[mode][seed]["agent"]
        err[mode] = np.abs(run["x_obs"] @ np.asarray(agent.generative_model.C).T - np.asarray(agent.y_star))[:, 0]
    n_total = len(err[modes[0]])
    stop = n_total if not n_show else min(n_total, start + n_show)
    steps = np.arange(start, stop)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                         "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
                         "axes.labelcolor": INK_2})
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(11, 6.2), facecolor="white")
    for ax, mode, other in zip(axes, modes, modes[::-1]):
        planned = runs[mode][seed]["planned"][start:stop]
        e = err[mode][start:stop]
        ax.plot(steps, err[other][start:stop], color=CONTEXT, lw=1.0, zorder=1)
        ax.plot(steps, e, color=COLOURS[mode], lw=1.5, solid_joinstyle="round", solid_capstyle="round", zorder=2)
        ax.plot(steps[planned], e[planned], ls="none", marker="o", ms=6, mfc=COLOURS[mode], mec="white", mew=1.2, zorder=3)
        ax.set_title(f"{TITLES[mode]}: planned at {planned.sum()} of {len(steps)} steps",
                     loc="left", color=INK, fontsize=11, pad=8)
        handles = [Line2D([], [], color=COLOURS[mode], lw=1.5, label="tracking error"),
                   Line2D([], [], ls="none", marker="o", ms=6, mfc=COLOURS[mode], mec="white", mew=1.2, label="planned at this step"),
                   Line2D([], [], color=CONTEXT, lw=1.0, label="other mode, for comparison")]
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=3, frameon=False,
                  labelcolor=INK_2, fontsize=9, handlelength=1.8, columnspacing=1.4, borderaxespad=0.2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylabel("|y − y*|  (°C)")
    axes[0].set_ylim(bottom=0)
    axes[-1].set_xlabel("step")
    axes[-1].set_xlim(start - 0.5, stop - 0.5)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"\n  plot saved to {path} (seed {seed}, steps {start}-{stop - 1})")


def table(rows, cols, title):
    print(f"\n{title}")
    width = max(len(r) for r in rows) + 2
    print(" " * width + "".join(f"{c:>22s}" for c in cols))
    for r, vals in rows.items():
        print(f"{r:<{width}s}" + "".join(f"{v:>22s}" for v in vals))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=None, help="steps per run (default: n_steps in env.yaml)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--burn-in", type=int, default=20, help="steps excluded from the tracking metrics")
    p.add_argument("--mc-samples", type=int, default=None, help="override metacognition.mc_samples")
    p.add_argument("--plot", default="tracking_error.png", help="where to save the error plot ('' to skip)")
    p.add_argument("--plot-seed", type=int, default=0, help="which seed's runs to plot")
    p.add_argument("--plot-start", type=int, default=None, help="first step shown (default: --burn-in)")
    p.add_argument("--plot-steps", type=int, default=100, help="number of steps shown (0 = to the end of the run)")
    args = p.parse_args()

    agent_cfg, env_cfg = load_config(HERE / "agent.yaml"), load_config(HERE / "env.yaml")
    n_steps = args.steps or env_cfg["n_steps"]
    meta = dict(agent_cfg.get("metacognition") or {"methods": ["gaussian", "mc"]})
    if args.mc_samples:
        meta["mc_samples"] = args.mc_samples
    methods = meta.get("methods", [])
    H = agent_cfg["planner"]["H"]
    print(f"H = {H} (agent.yaml), action_rule = {agent_cfg['planner'].get('action_rule', 'mean')}, "
          f"{args.seeds} seeds x {n_steps} steps, predictors = {methods}")

    simulate(agent_cfg, env_cfg, "every_step", 6, 0)                     # warm-up: compile before timing
    simulate(agent_cfg, env_cfg, "efe_trigger", 6, 0, meta)

    # Part 1 runs without predictors so timing reflects the agent only; Part 2 reruns efe_trigger with the
    # predictors, which must reproduce exactly the same behaviour (they only log)
    runs = {mode: [simulate(agent_cfg, env_cfg, mode, n_steps, s) for s in range(args.seeds)]
            for mode in ("every_step", "efe_trigger")}
    pred_runs = [simulate(agent_cfg, env_cfg, "efe_trigger", n_steps, s, meta) for s in range(args.seeds)]
    unchanged = all(np.array_equal(a["u"], b["u"]) for a, b in zip(runs["efe_trigger"], pred_runs))

    names = list(behaviour_metrics(runs["every_step"][0], args.burn_in))
    rows = {}
    for name in names:
        rows[name] = []
        for mode in ("every_step", "efe_trigger"):
            v = np.array([behaviour_metrics(r, args.burn_in)[name] for r in runs[mode]])
            rows[name].append(f"{v.mean():.4g} ± {v.std():.2g}")
    table(rows, ["every_step", "efe_trigger"], "PART 1  replanning modes (mean ± sd over seeds; same plant noise)")
    if args.plot:
        start = args.burn_in if args.plot_start is None else args.plot_start
        plot_error(runs, min(args.plot_seed, args.seeds - 1), start, args.plot_steps, args.plot)

    agents = [r["agent"] for r in pred_runs]
    events = [e for a in agents for e in closed_events(a)]
    if H < 2 or not events or not methods:
        print("\nPART 2 skipped: needs H >= 2, closed plans and at least one predictor in metacognition.methods.")
        return
    o = np.array([oi for _, oi in events])
    emp = np.bincount(o, minlength=H) / len(o)
    se = lambda p, n: np.sqrt(p * (1 - p) / n) if n else np.nan           # binomial standard error
    labels = [f"K_P={k}" for k in range(1, H)] + ["no trigger"]
    print(f"\nPART 2  plan lifetimes: {len(events)} completed plans "
          f"({sum(e['status'] == 'open' for a in agents for e in a.events)} censored at the end of a run); "
          f"predictors left behaviour unchanged: {unchanged}")

    res, pmfs = {}, {}
    for m in methods:
        res[m], pmfs[m] = predictor_metrics(events, m, H)
    rows = {lab: [f"{res[m]['mean pmf'][i]:.3f}" for m in methods] + [f"{emp[i]:.3f} ± {se(emp[i], len(o)):.3f}"]
            for i, lab in enumerate(labels)}
    rows["E[lifetime] (actions)"] = ([f"{res[m]['E[lifetime] predicted']:.3f}" for m in methods]
                                     + [f"{np.mean(o + 1):.3f} ± {np.std(o + 1) / np.sqrt(len(o)):.3f}"])
    rows["log score (higher=better)"] = [f"{res[m]['log score']:.3f}" for m in methods] + ["-"]
    rows["RPS (lower=better)"] = [f"{res[m]['RPS']:.4f}" for m in methods] + ["-"]
    table(rows, methods + ["realised"], "  q(K_P) averaged over plans vs realised frequencies (± 1 standard error)")

    rows = {}
    for i, k in enumerate(range(1, H)):
        n_reached, emp_h = res[methods[0]]["hazards"][i][1], res[methods[0]]["hazards"][i][3]
        rows[f"k={k} (plans reaching: {n_reached})"] = ([f"{res[m]['hazards'][i][2]:.3f}" for m in methods]
                                                       + [f"{emp_h:.3f} ± {se(emp_h, n_reached):.3f}"])
    table(rows, methods + ["realised"], "  hazard h_k = q(trigger at k | no trigger before k), predicted vs realised")

    if "gaussian" in methods and "mc" in methods:
        tv = 0.5 * np.abs(pmfs["gaussian"] - pmfs["mc"]).sum(1)
        print(f"\n  gaussian vs mc, per plan: total-variation distance mean {tv.mean():.3f}, max {tv.max():.3f}")

    plan_cost = agents[0].n * H
    for m in methods:
        per_plan = sum(a.compute["meta_step_evals"].get(m, 0) for a in agents) / sum(len(a.events) for a in agents)
        print(f"  cost of '{m}' per plan: {per_plan:.0f} EFE step-evals ({per_plan / plan_cost:.2%} of planning, n*H = {plan_cost})")

    # diagnostic: the predictors assume the agent's posterior covariance equals the exact filter covariance
    worst = 0.0
    for a in agents:
        kf, starts = a.generative_model, {ev["t"]: ev for ev in a.events}
        for chk in a.trigger_log:
            ev = starts.get(chk["t"] - chk["depth"])
            if ev is None:
                continue
            S = ev["P"]
            for _ in range(chk["depth"]):
                Sp = kf.A @ S @ kf.A.T + kf.Q
                Sy = kf.C @ Sp @ kf.C.T + kf.R
                K = jnp.linalg.solve(Sy, kf.C @ Sp).T
                S = Sp - K @ Sy @ K.T
            worst = max(worst, float(jnp.max(jnp.abs(chk["P"] - S) / jnp.abs(S))))
    print(f"  diagnostic: agent posterior covariance vs exact filter at trigger checks, max relative error {worst:.1e}")


if __name__ == "__main__":
    main()
