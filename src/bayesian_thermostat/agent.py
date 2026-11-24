import jax
import jax.numpy as jnp
from jax import lax
from utils import vfe, init_params, make_L

from utils import vfe

class AIfAgent():
    def __init__(self, belief_prior: dict, generative_model_params: dict):
        self.generative_model = KalmanFilter(generative_model_params)
        self.state_belief = belief_prior
        self.u_prev = 0

    def inference(self, y: jax.Array, m_prior: jax.Array, L_prior: jax.Array, C: jax.Array, L_R: jax.Array, kappa_m=0.03, kappa_L=0.03, tol=1e-10, max_iter=5000):
        """Run inference to compute current belief."""
        args = (y, m_prior, L_prior, C, L_R)
        value_and_grad = jax.value_and_grad(vfe)

        def cond(carry):
            _, F_prev, F_curr, _, i = carry
            return (i < max_iter) & (jnp.abs(F_curr - F_prev) >= tol)

        def body(carry):
            params, F_prev, F_curr, scale, i = carry
            _, grads = value_and_grad(params, *args)
            candidate = {"m": params["m"] - scale * kappa_m * grads["m"],
                    "L_raw": params["L_raw"] - scale * kappa_L * grads["L_raw"]}
            F_cand = vfe(candidate, *args)
            accept = (F_cand <= F_curr)
            params = {"m":     jnp.where(accept, candidate["m"],     params["m"]),
                    "L_raw": jnp.where(accept, candidate["L_raw"], params["L_raw"])}
            F_prev = jnp.where(accept, F_curr, F_prev)
            F_curr = jnp.where(accept, F_cand, F_curr)
            scale = jnp.where(accept, scale * 1.5, scale * 0.5)
            return params, F_prev, F_curr, scale, i + 1

        params0 = init_params(m_prior, L_prior)
        F0 = vfe(params0, *args)
        params, F_prev, F_curr, _, n_iter = lax.while_loop(cond, body, (params0, jnp.inf, F0, 1.0, 0))

        info = {"iterations": n_iter, "F": F_curr, "dF": jnp.abs(F_curr - F_prev)}
        return params["m"], make_L(params["L_raw"]), info

    def plan(self, m: jax.Array, P: jax.Array):
        """Replan based off current information."""
        return u

    def act(self, y: jax.Array):
        """Do all computation for the current timestep (inference and planning)."""
        m_hat, P_hat = self.generative_model.predict(
            self.u_prev, self.state_belief["m"], self.state_belief["P"]
        )
        m, P, info = self.inference(
            y, m_hat, jnp.linalg.cholesky(P_hat), self.generative_model.C, self.generative_model.L_R
        )

        return u

class KalmanFilter():
    def __init__(self, generative_model_params: dict):
        for (k, v) in generative_model_params.items():
            setattr(self, k, v)

    def predict(self, u: jax.Array, m: jax.Array, P: jax.Array):
        m_next = self.A @ m + self.B @ u + self.c
        P_next = self.A @ P @ self.A.T + self.Q
        return m_next, P_next