import jax
import jax.numpy as jnp
from agent import AIfAgent

class BayesianThermometer():
    def __init__(self, params: dict, key: jax.Array):
        self.alpha = params["alpha"]
        self.T_amb = params["T_amb"]
        self.beta = params["beta"]
        self.q_std = params["q_std"]
        self.r_std = params["r_std"]
        self.key = key

    @staticmethod
    @jax.jit()
    def _step(x: jax.Array, u: jax.Array, alpha: float, T_amb: float, beta: float):
        """Compute next state with closed form solution where the state is the temperature."""
        x_next = alpha * x + (1 - alpha) * T_amb + beta * u
        return x_next

    def step(self, x: jax.Array, u: jax.Array):
        self.key, k = jax.random.split(self.key)
        w = self.q_std * jax.random.normal(k, jnp.shape(x))
        return self._step(x, u, self.alpha, self.T_amb, self.beta) + w

    def observe(self, x: jax.Array):
        self.key, k = jax.random.split(self.key)
        return x + self.r_std * jax.random.normal(k, jnp.shape(x))

def run_sim(agentparamfilename: str, envparamfilename: str, seed: int = 0):
    agent_params, belief_prior = ...
    env_params, x_init, N = ...

    agent = AIfAgent(belief_prior, agent_params)
    env = BayesianThermometer(env_params, jax.random.key(seed))

    x = x_init
    x_hist = [x_init] ; y_hist = [] ; u_hist = []
    for i in range(N):
        y = env.observe(x)
        u = agent.act(y)
        x = env.step(x, u)

        x_hist.append(x) ; y_hist.append(y) ; u_hist.append(u)
    return x_hist, y_hist, u_hist