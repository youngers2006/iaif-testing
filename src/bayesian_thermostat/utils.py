import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from agent import KalmanFilter
from typing import NamedTuple

class PolicyPosterior(NamedTuple):
    U: jax.Array
    G: jax.Array
    log_w: jax.Array

    def weights(self):
        return jnp.exp(self.log_w)

    def ess(self):
        return jnp.exp(-jax.nn.logsumexp(2 * self.log_w))

    def expected_plan(self):
        return self.weights() @ self.U

def make_L(L_raw: jax.Array):
    return jnp.tril(L_raw, -1) + jnp.diag(jax.nn.softplus(jnp.diag(L_raw)))

def init_params(m_hat: jax.Array, L_hat: jax.Array) -> dict:
    L_raw = jnp.tril(L_hat, -1) + jnp.diag(jnp.log(jnp.expm1(jnp.diag(L_hat))))
    return {"m": m_hat, "L_raw": L_raw}

def frobenius_norm_squared(A: jax.Array) -> jax.Array:
    """Get Frobenius norm of a matrix A and L2 norm of vector A."""
    f_norm = jnp.sum(A ** 2)
    return f_norm

def kl_div_chol(m_P: jax.Array, L_P: jax.Array, m_Q: jax.Array, L_Q: jax.Array):
    """Get the KL divergence between two gaussians where the covariance matrices are decomposed into Cholesky components."""
    n = L_Q.shape[-1]
    A = solve_triangular(L_Q, L_P, lower=True)
    B = solve_triangular(L_Q, (m_P - m_Q), lower=True)
    lndet_Q = 2 * jnp.sum(jnp.log(jnp.diag(L_Q)))
    lndet_P = 2 * jnp.sum(jnp.log(jnp.diag(L_P)))
    D_kl = 0.5 * (frobenius_norm_squared(A) + frobenius_norm_squared(B) - n + lndet_Q - lndet_P)
    return D_kl

def get_ambiguity(L_R: jax.Array) -> jax.Array:
    """Get ambiguity term."""
    D = L_R.shape[-1]
    return 0.5 * D * (1.0 + jnp.log(2 * jnp.pi)) + jnp.sum(jnp.log(jnp.diag(L_R)))

def log_prob_chol(y: jax.Array, m_y: jax.Array, L_R: jax.Array):
    """Get the log prob of some observation y for a gaussian where the covariance matrix is decomposed into Cholesky components"""
    D = y.shape[0]
    z = solve_triangular(L_R, y - m_y, lower=True)
    lndet_R = 2 * jnp.sum(jnp.log(jnp.diag(L_R)))
    return -0.5 * (D * jnp.log(2 * jnp.pi) + lndet_R + frobenius_norm_squared(z))

def expected_log_lik(y: jax.Array, m_q: jax.Array, L_q: jax.Array, C: jax.Array, L_R: jax.Array) -> jax.Array:
    """Get the expected log probability of observation."""
    B = solve_triangular(L_R, C @ L_q, lower=True)
    return log_prob_chol(y, C @ m_q, L_R) - 0.5 * frobenius_norm_squared(B)

@jax.jit
def vfe(params: dict, y: jax.Array, m_pred: jax.Array, L_pred: jax.Array, C: jax.Array, L_R: jax.Array):
    """VFE where L is the cholesky decomposition of P."""
    # Extract parameter terms
    m_prior = params["m"]
    L_prior = make_L(params["L_raw"])

    # Divergence term    
    complexity = kl_div_chol(m_prior, L_prior, m_pred, L_pred)

    # Log prob calc
    accuracy = expected_log_lik(y, m_prior, L_prior, C, L_R)
    return complexity - accuracy

@jax.jit()
def efe(u_seq: jax.Array, m: jax.Array, P: jax.Array, kf: KalmanFilter, y_star: jax.Array, L_star: jax.Array):
    mu = m ; sigma = P ; efe_0 = 0
    def scan_step(carry, u):
        mu, P, efe = carry
        mu_, P_ = kf.predict(u, mu, P)

        mu_y, P_y = kf.predict_obs(mu_)
        L_y = jnp.linalg.cholesky(P_y)

        risk = kl_div_chol(mu_y, L_y, y_star, L_star)
        ambiguity = get_ambiguity(kf.L_R)
        efe_t = risk + ambiguity

        efe = efe + efe_t
        return (mu_, P_, efe), efe_t

    (mu_T, P_T, efe_T), efe_per_timestep = jax.lax.scan(scan_step, (mu, sigma, efe_0), u_seq)
    return efe_T