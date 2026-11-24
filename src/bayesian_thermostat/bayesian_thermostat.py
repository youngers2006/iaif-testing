import jax
import jax.numpy as jnp
from agent import AIfAgent

class BayesianThermometer():
    def __init__(self, params: dict):
        self.alpha = params.alpha
        self.T_amb = params.T_amb
        self.beta = params.beta

    @jax.jit()
    def _step(x: jax.Array, u: jax.Array, alpha: float, T_amb: float, beta: float):
        """Compute next state with closed form solution where the state is the temperature."""
        x_next = alpha * x + (1 - alpha) * T_amb + beta * u
        return x_next

    def step(self, x: jax.Array, u: jax.Array):
        return self._step(x, u, self.alpha, self.T_amb, self.beta)


def run_sim(agentparamfilename: str, envparamfilename: str):
    params = ...

    agent = AIfAgent()
    env = BayesianThermometer()

    x = x_init
    for i in range(N):
        u = agent.act()
        x_ = self.step(x, u)