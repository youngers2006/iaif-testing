import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from utils import vfe, init_params, make_L, efe, PolicyPosterior
from metacognition import suffix_references, run_predictor
from typing import NamedTuple

class AIfAgent():
    def __init__(self, belief_prior: dict, generative_model_params: dict, planner_params: dict, seed: int = 0, meta_params: dict = None):
        self.generative_model = KalmanFilter.from_params(generative_model_params)
        self.state_belief = belief_prior
        self.u_prev = jnp.zeros(self.generative_model.B.shape[1])

        self.H = planner_params["H"]
        self.n = planner_params["n"]
        self.gamma = planner_params["gamma"]
        self.y_star = planner_params["y_star"]
        self.L_star = jnp.linalg.cholesky(planner_params["Sigma_star"])
        self.action_rule = planner_params.get("action_rule", "mean")
        self.key = jax.random.key(seed)

        # replanning: "every_step" (receding horizon) or "efe_trigger" (IAIF: keep the plan until the EFE Error Trigger fires)
        self.replan_mode = planner_params.get("replan_mode", "every_step")
        if self.replan_mode not in ("every_step", "efe_trigger"):
            raise ValueError(f"unknown replan_mode {self.replan_mode!r} (available: every_step, efe_trigger)")
        # lifetime predictors for the selected plan: logged only, with their own random stream so behaviour is unchanged
        self.meta_params = dict(meta_params or {})
        self.meta_methods = list(self.meta_params.get("methods", []))
        self.meta_key = jax.random.fold_in(jax.random.key(seed), 1)

        self.t = 0 # number of observations processed
        self.current_plan, self.G_ref, self.depth, self.q_policy = None, None, 0, None
        self.events, self.trigger_log = [], [] # planning events (with predictions) and trigger checks
        self.compute = {"plan_calls": 0, "plan_step_evals": 0, "trigger_step_evals": 0, "meta_step_evals": {}}

    @staticmethod
    @jax.jit()
    def _inference(y: jax.Array, m_prior: jax.Array, L_prior: jax.Array, C: jax.Array, L_R: jax.Array, kappa_m=0.03, kappa_L=0.03, tol=1e-6, max_iter=5000):
        """Run inference to compute current belief."""
        args = (y, m_prior, L_prior, C, L_R)
        value_and_grad = jax.value_and_grad(vfe)

        def cond(carry):
            _, F_prev, F_curr, _, i = carry
            return (i < max_iter) & (jnp.abs(F_curr - F_prev) >= tol)

        def loop(carry):
            params, F_prev, F_curr, scale, i = carry
            _, grads = value_and_grad(params, *args)
            candidate = {"m": params["m"] - scale * kappa_m * grads["m"],
                    "L_raw": params["L_raw"] - scale * kappa_L * grads["L_raw"]}
            F_cand = vfe(candidate, *args)
            accept = (F_cand <= F_curr)
            params = {"m": jnp.where(accept, candidate["m"], params["m"]),
                    "L_raw": jnp.where(accept, candidate["L_raw"], params["L_raw"])}
            F_prev = jnp.where(accept, F_curr, F_prev)
            F_curr = jnp.where(accept, F_cand, F_curr)
            scale = jnp.where(accept, scale * 1.5, scale * 0.5)
            return params, F_prev, F_curr, scale, i + 1

        params0 = init_params(m_prior, L_prior)
        F0 = vfe(params0, *args)
        params, F_prev, F_curr, _, n_iter = lax.while_loop(cond, loop, (params0, jnp.inf, F0, 1.0, 0))

        info = {"iterations": n_iter, "F": F_curr, "dF": jnp.abs(F_curr - F_prev)}
        return params["m"], make_L(params["L_raw"]), info

    def inference(self, y: jax.Array, m_prior: jax.Array, L_prior: jax.Array, C: jax.Array, L_R: jax.Array, kappa_m=0.03, kappa_L=0.03, tol=1e-6, max_iter=5000):
        """Run inference to compute current belief."""
        return self._inference(y, m_prior, L_prior, C, L_R, kappa_m, kappa_L, tol, max_iter)

    def plan(self, m: jax.Array, P: jax.Array, H: int, n: int):
        """Replan based off current information."""
        kf = self.generative_model
        self.key, k_sample, k_pick = jax.random.split(self.key, 3)
        return self._plan(
            m, P, self.y_star, self.L_star, kf, self.gamma, n, H, k_sample
        )

    @staticmethod
    @jax.jit(static_argnames=("n", "H"))
    def _plan(
            m: jax.Array, 
            P: jax.Array, 
            y_star: jax.Array, 
            L_star: jax.Array, 
            kf: "KalmanFilter", 
            gamma: float, 
            n: int, 
            H: int, 
            k_sample: jax.Array
        ):
        U = jax.random.uniform(k_sample, (n, H, kf.B.shape[1]), minval=kf.u_lower, maxval=kf.u_upper)
        G = jax.vmap(efe, (0, None, None, None, None, None))(U, m, P, kf, y_star, L_star)
        log_w = jax.nn.log_softmax(- gamma * G)
        return PolicyPosterior(U, G, log_w)

    def act(self, y: jax.Array):
        """Do all computation for the current timestep: inference, then either replanning or executing the
        next action of the committed plan ("every_step" mode replans at every step)."""
        m_hat, P_hat = self.generative_model.predict(
            self.u_prev, self.state_belief["m"], self.state_belief["P"]
        )
        m, L, info = self.inference(
            y, m_hat, jnp.linalg.cholesky(P_hat), self.generative_model.C, self.generative_model.L_R
        )
        P = L @ L.T

        self.state_belief = {"m": m, "P": P}

        replanned = self._needs_replan(m, P)
        if replanned:
            self._replan(m, P)

        u = self.current_plan[self.depth]
        self.depth += 1
        self.t += 1

        self.u_prev = u
        info = {**info, "ess": self.q_policy.ess(), "replanned": replanned}
        return u, info

    def _select_plan(self, q_policy: PolicyPosterior):
        """Commit to one full plan from q(pi): the expected plan ("mean"), one sampled policy ("sample") or the
        most probable sampled policy ("map"). Its first action is the action applied now."""
        if self.action_rule == "mean":
            return q_policy.expected_plan()
        elif self.action_rule == "sample":
            self.key, k = jax.random.split(self.key)
            return q_policy.U[jax.random.categorical(k, q_policy.log_w)]
        elif self.action_rule == "map":
            return q_policy.U[jnp.argmax(q_policy.log_w)]
        raise ValueError(f"unknown action_rule {self.action_rule!r}")

    def _needs_replan(self, m: jax.Array, P: jax.Array):
        """Every step in "every_step" mode. In "efe_trigger" mode (IAIF), when the EFE Error Trigger fires, i.e. the
        remaining suffix re-evaluated from the current posterior is worse than its planning-time reference,
        or when the plan has been executed to the end."""
        if self.current_plan is None or self.replan_mode == "every_step":
            return True
        k = self.depth
        if k >= self.H:
            self._close_plan(None)
            return True
        G_now = efe(self.current_plan[k:], m, P, self.generative_model, self.y_star, self.L_star)
        fired = bool(G_now > self.G_ref[k - 1])
        self.compute["trigger_step_evals"] += self.H - k
        self.trigger_log.append({"t": self.t, "depth": k, "G": float(G_now), "G_ref": float(self.G_ref[k - 1]),
                                 "fired": fired, "P": P})
        if fired:
            self._close_plan(k)
        return fired

    def _replan(self, m: jax.Array, P: jax.Array):
        """Plan and commit to a plan. In "efe_trigger" mode also store its suffix references Ghat^(k) and log the
        lifetime predictions of the configured predictors."""
        kf = self.generative_model
        q_policy = self.plan(m, P, self.H, self.n)
        plan = self._select_plan(q_policy)
        self.current_plan, self.depth, self.q_policy = plan, 0, q_policy
        self.compute["plan_calls"] += 1
        self.compute["plan_step_evals"] += self.n * self.H

        if self.replan_mode == "efe_trigger":
            self.G_ref = suffix_references(plan, m, P, kf, self.y_star, self.L_star)
            event = {"t": self.t, "plan": plan, "m": m, "P": P, "G_ref": self.G_ref,
                     "status": "open", "K_P": None, "predictions": {}}
            for method in self.meta_methods:
                self.meta_key, key = jax.random.split(self.meta_key)
                pred, cost = run_predictor(method, plan, m, P, kf, self.y_star, self.L_star, self.G_ref, key, self.meta_params)
                event["predictions"][method] = pred
                self.compute["meta_step_evals"][method] = self.compute["meta_step_evals"].get(method, 0) + cost
            self.events.append(event)

    def _close_plan(self, K_P):
        """Record how the current plan ended: first triggered depth K_P, or None if it was executed to the end.
        A plan still running when the simulation stops keeps status "open" (censored)."""
        if self.events and self.events[-1]["status"] == "open":
            self.events[-1]["status"] = "triggered" if K_P is not None else "exhausted"
            self.events[-1]["K_P"] = K_P

class KalmanFilter(NamedTuple):
    """Linear-Gaussian generative model. A NamedTuple is already a JAX pytree, so it can be passed into jit/vmap."""
    A: jax.Array
    B: jax.Array
    c: jax.Array
    Q: jax.Array
    C: jax.Array
    R: jax.Array
    L_R: jax.Array
    u_lower: jax.Array
    u_upper: jax.Array

    @classmethod
    def from_params(cls, params: dict):
        return cls(A=params["A"], B=params["B"], c=params["c"], Q=params["Q"], C=params["C"], R=params["R"],
                   L_R=jnp.linalg.cholesky(params["R"]), u_lower=params["u_lower"], u_upper=params["u_upper"])

    def predict(self, u: jax.Array, m: jax.Array, P: jax.Array):
        m_next = self.A @ m + self.B @ u + self.c
        P_next = self.A @ P @ self.A.T + self.Q
        return m_next, P_next

    def predict_obs(self, mu: jax.Array, P: jax.Array):
        return self.C @ mu, self.C @ P @ self.C.T + self.R