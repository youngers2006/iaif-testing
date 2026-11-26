import jax
import jax.numpy as jnp
import yaml
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

def load_config(path) -> dict:
    """Read a YAML file. Every list becomes a float64 jax array; scalars and strings stay Python values."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    def convert(v):
        if isinstance(v, dict):
            return {k: convert(x) for k, x in v.items()}
        if isinstance(v, list):
            return jnp.asarray(v, dtype=jnp.float64)
        return v

    return convert(cfg)

def run_sim(agentparamfilename: str, envparamfilename: str, seed: int = 0):
    agent_config = load_config(agentparamfilename)
    env_config = load_config(envparamfilename)

    agent = AIfAgent(
        agent_config["belief_prior"], agent_config["generative_model"], agent_config["planner"], seed=seed
    )
    env = BayesianThermometer(env_config["plant"], jax.random.key(seed + 1))

    x = env_config["x_init"]
    x_hist = [x] ; y_hist = [] ; u_hist = []
    for i in range(env_config["n_steps"]):
        y = env.observe(x)
        u, info = agent.act(y)
        x = env.step(x, u)

        x_hist.append(x) ; y_hist.append(y) ; u_hist.append(u)
    return x_hist, y_hist, u_hist

if __name__ == "__main__":
    x, y, u = run_sim(agentparamfilename="agent.yaml", envparamfilename="env.yaml", seed=0)