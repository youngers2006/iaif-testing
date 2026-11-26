import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from utils import vfe, init_params, make_L

class AIfAgent():
    def __init__(self, belief_prior: dict, generative_model_params: dict):
        self.generative_model = KalmanFilter(generative_model_params)
        self.state_belief = belief_prior
        self.u_prev = jnp.zeros(self.generative_model.B.shape[1])

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
        u_bounds = [self.generative_model.u_lower, self.generative_model.u_upper]
        
        return 

    def rollout(self, m: jax.Array, P: jax.Array, u_seq: jax.Array):
        m_hist = [] ; P_hist = []
        for i in range(u_seq[0]):
            m, P = self.generative_model.predict(u_seq[i], m, P)
        return

    def act(self, y: jax.Array):
        """Do all computation for the current timestep (inference and planning)."""
        m_hat, P_hat = self.generative_model.predict(
            self.u_prev, self.state_belief["m"], self.state_belief["P"]
        )
        m, L, info = self.inference(
            y, m_hat, jnp.linalg.cholesky(P_hat), self.generative_model.C, self.generative_model.L_R
        )
        P = L @ L.T

        self.state_belief = {"m": m, "P": P}
        u = self.plan(m, P)
        self.u_prev = u
        return u, info

class KalmanFilter():
    def __init__(self, generative_model_params: dict):
        for (k, v) in generative_model_params.items():
            setattr(self, k, v)
        self.L_R = jnp.linalg.cholesky(self.R)

    def predict(self, u: jax.Array, m: jax.Array, P: jax.Array):
        m_next = self.A @ m + self.B @ u + self.c
        P_next = self.A @ P @ self.A.T + self.Q
        return m_next, P_next