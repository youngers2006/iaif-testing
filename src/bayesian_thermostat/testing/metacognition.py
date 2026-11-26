from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from utils import efe, efe_terms


class LifetimePrediction(NamedTuple):
    """Predicted distribution of the first EFE-triggered replanning depth K_P of one plan."""
    pmf: jax.Array        # (H-1,) q(K_P = k), k = 1..H-1
    p_none: jax.Array     # ()     q(no trigger before the plan is exhausted)
    hazard: jax.Array     # (H-1,) q(trigger at k | no trigger at depths < k)
    marginal: jax.Array   # (H-1,) q(trigger at k), ignoring earlier checks


def suffix_references(plan, m, P, kf, y_star, L_star):
    """Ghat^(k), k = 1..H-1: the EFE assigned at planning time to the suffix plan[k:] (tail sums of the rollout)."""
    terms = efe_terms(plan, m, P, kf, y_star, L_star)
    tails = jnp.cumsum(terms[::-1])[::-1]          # tails[j] = sum(terms[j:])
    return tails[1:]


def _nominal_update(kf, u, mu, Sbar):
    """One step of the within-posterior belief (eqs. 62-71): predict under action u, then update with the
    certainty-equivalent observation, so the mean stays at the prediction and the covariance shrinks as in
    the filter. Returns the predicted mean, the updated covariance, the gain K and S_cond."""
    mu_pred, Sbar_pred = kf.predict(u, mu, Sbar)
    _, S_cond = kf.predict_obs(mu_pred, Sbar_pred)           # conditional innovation covariance (excludes Omega)
    K = jnp.linalg.solve(S_cond, kf.C @ Sbar_pred).T         # Sbar_pred C^T S_cond^-1
    return mu_pred, Sbar_pred - K @ S_cond @ K.T, K, S_cond


def _sigma_points(mu, cov, kappa):
    """Unscented-transform sigma points and weights for N(mu, cov). A symmetric square root is used because
    Omega can be rank deficient (e.g. fewer observations than states)."""
    n = mu.shape[0]
    evals, evecs = jnp.linalg.eigh((n + kappa) * cov)
    root = evecs * jnp.sqrt(jnp.clip(evals, 0.0))
    pts = jnp.concatenate([mu[None, :], mu + root.T, mu - root.T])
    w = jnp.concatenate([jnp.array([kappa / (n + kappa)]), jnp.full(2 * n, 1.0 / (2 * (n + kappa)))])
    return pts, w


def _ut_suffix_efe(suffix, mu, Omega, Sbar, kf, y_star, L_star, kappa):
    """Push M ~ N(mu, Omega) through g_k(m) = EFE(suffix; N(m, Sbar)) (eqs. 79-85, 95)."""
    pts, w = _sigma_points(mu, Omega, kappa)
    G = jax.vmap(efe, in_axes=(None, 0, None, None, None, None))(suffix, pts, Sbar, kf, y_star, L_star)
    mu_G = w @ G
    var_G = w @ (G - mu_G) ** 2
    C_MG = (pts - mu).T @ (w * (G - mu_G))
    return mu_G, var_G, C_MG


@jax.jit
def _gaussian_core(plan, m, P, kf, y_star, L_star, G_ref, eps_G, kappa):
    H = plan.shape[0]
    zero = jnp.zeros_like(P)
    mu_s, Om_s = m, zero                 # survivor metabelief; Omega_t = 0 because the current posterior is known
    mu_u, Om_u = m, zero                 # unconditioned metabelief, only for the marginal trigger probabilities
    Sbar = P
    surv = jnp.asarray(1.0, dtype=P.dtype)
    pmf, hazard, marginal = [], [], []
    for k in range(1, H):
        u = plan[k - 1]
        # (a)-(d): propagate the within-posterior belief and both metabeliefs under a_k (eqs. 72, 77)
        mu_s, Sbar_new, K, S_cond = _nominal_update(kf, u, mu_s, Sbar)
        mu_u = kf.predict(u, mu_u, Sbar)[0]
        Om_s = kf.A @ Om_s @ kf.A.T + K @ S_cond @ K.T
        Om_u = kf.A @ Om_u @ kf.A.T + K @ S_cond @ K.T
        Sbar = Sbar_new
        suffix, G_hat = plan[k:], G_ref[k - 1]

        # marginal trigger probability, no survivor conditioning (eq. 87)
        mG, vG, _ = _ut_suffix_efe(suffix, mu_u, Om_u, Sbar, kf, y_star, L_star, kappa)
        sG = jnp.sqrt(vG)
        marginal.append(jnp.where(sG < eps_G, (mG > G_hat).astype(P.dtype), norm.cdf((mG - G_hat) / jnp.maximum(sG, eps_G))))

        # (e)-(h): hazard under the survivor metabelief (eqs. 96-98) with the degenerate case (eq. 108)
        mG, vG, C_MG = _ut_suffix_efe(suffix, mu_s, Om_s, Sbar, kf, y_star, L_star, kappa)
        sG = jnp.sqrt(vG)
        degenerate = sG < eps_G
        a = (G_hat - mG) / jnp.maximum(sG, eps_G)
        h = jnp.where(degenerate, (mG > G_hat).astype(P.dtype), norm.sf(a))

        # (j): condition on survival G <= Ghat and moment-match back to a Gaussian (eqs. 100-103)
        lam = jnp.exp(norm.logpdf(a) - norm.logcdf(a))
        delta = lam * (lam + a)
        cond = (~degenerate) & (h < 1.0)
        mu_s = jnp.where(cond, mu_s - C_MG / jnp.maximum(sG, eps_G) * lam, mu_s)
        Om_s = jnp.where(cond, Om_s - jnp.outer(C_MG, C_MG) / jnp.maximum(vG, eps_G ** 2) * delta, Om_s)

        # (i): first-trigger probability and survival recursion (eqs. 112-113)
        pmf.append(surv * h)
        hazard.append(h)
        surv = surv * (1.0 - h)

    stack = lambda xs: jnp.stack(xs) if xs else jnp.zeros((0,), dtype=P.dtype)
    return LifetimePrediction(stack(pmf), surv, stack(hazard), stack(marginal))


def predict_lifetime_gaussian(plan, m, P, kf, y_star, L_star, G_ref, eps_G=1e-9, kappa=None):
    """Gaussian metabelief predictor of q(K_P | plan) (notes, sections 4-8). kappa defaults to 3 - n, which
    matches the Gaussian fourth moment and makes the UT variance exact for a quadratic EFE in one dimension."""
    kappa = 3.0 - m.shape[0] if kappa is None else kappa
    return _gaussian_core(plan, m, P, kf, y_star, L_star, G_ref, eps_G, kappa)


@partial(jax.jit, static_argnames=("n_samples",))
def predict_lifetime_mc(plan, m, P, kf, y_star, L_star, G_ref, key, n_samples):
    """Posterior-predictive Monte Carlo reference (notes, section 9.2): sample states and observations from the
    agent's own generative model, run the exact filter update, re-evaluate every suffix and record the first
    triggered depth of each sampled history."""
    H, n_x, n_y = plan.shape[0], m.shape[0], kf.C.shape[0]
    L_Q = jnp.linalg.cholesky(kf.Q)
    key, k0 = jax.random.split(key)
    x = m + jax.random.normal(k0, (n_samples, n_x)) @ jnp.linalg.cholesky(P).T       # x_t ~ N(m, P)
    mm, PP = jnp.broadcast_to(m, (n_samples, n_x)), P
    first = jnp.zeros(n_samples, dtype=jnp.int32)                                      # 0 = no trigger (yet)
    pmf, hazard, marginal = [], [], []
    for k in range(1, H):
        u = plan[k - 1]
        key, kw, kv = jax.random.split(key, 3)
        x = x @ kf.A.T + kf.B @ u + kf.c + jax.random.normal(kw, (n_samples, n_x)) @ L_Q.T
        y = x @ kf.C.T + jax.random.normal(kv, (n_samples, n_y)) @ kf.L_R.T
        m_pred, P_pred = mm @ kf.A.T + kf.B @ u + kf.c, kf.A @ PP @ kf.A.T + kf.Q
        S = kf.C @ P_pred @ kf.C.T + kf.R
        K = jnp.linalg.solve(S, kf.C @ P_pred).T
        mm, PP = m_pred + (y - m_pred @ kf.C.T) @ K.T, P_pred - K @ S @ K.T
        G = jax.vmap(efe, in_axes=(None, 0, None, None, None, None))(plan[k:], mm, PP, kf, y_star, L_star)
        fired = G > G_ref[k - 1]
        at_risk = first == 0
        new = at_risk & fired
        frac = lambda b: jnp.mean(b.astype(P.dtype))       # jnp.mean of a bool array is float32 even with x64
        pmf.append(frac(new))
        hazard.append(jnp.sum(new) / jnp.maximum(jnp.sum(at_risk), 1))
        marginal.append(frac(fired))
        first = jnp.where(new, k, first)
    stack = lambda xs: jnp.stack(xs).astype(P.dtype) if xs else jnp.zeros((0,), dtype=P.dtype)
    return LifetimePrediction(stack(pmf), jnp.mean((first == 0).astype(P.dtype)), stack(hazard), stack(marginal))


def run_predictor(method, plan, m, P, kf, y_star, L_star, G_ref, key, params):
    """Dispatch to a lifetime predictor. Returns (prediction, cost), where cost counts single-step EFE
    evaluations so the predictor's computation can be compared with planning (n * H per plan)."""
    H, n_x = plan.shape[0], m.shape[0]
    suffix_steps = sum(H - k for k in range(1, H))
    if method == "gaussian":
        pred = predict_lifetime_gaussian(plan, m, P, kf, y_star, L_star, G_ref, eps_G=params.get("eps_G", 1e-9))
        return pred, (2 * n_x + 1) * suffix_steps       # survivor UT only; the marginal diagnostics double this
    if method == "mc":
        n_samples = int(params.get("mc_samples", 20000))
        return predict_lifetime_mc(plan, m, P, kf, y_star, L_star, G_ref, key, n_samples), n_samples * suffix_steps
    raise ValueError(f"unknown lifetime predictor {method!r} (available: gaussian, mc)")
