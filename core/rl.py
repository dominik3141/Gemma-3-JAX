r"""
Some simple RL roughly using the DeepSeek R1 Zero approach.

Our first objective is to make the Gemma model very good at systematically calculating square roots.
Without incentivizing any particular approach, it will be interesting to see what kind of approach will emerge.
Optimally we would remove all knowledge of human methods for calculating square roots first (Babylonian etc.)
to see if these can be rediscovered using RL.

Plan:
1.  Prompt model N times to calculate square root of random number (N different seeds),
    instruct model to put the final result in <result> tag
2.  Assign reward with baseline (GRPO)
3.  Loss
4.  Backpropagate

Stuff to fix:
-   We need a forward function that is optimized for inference
"""

MAX_ROOT = 90000000
MIN_ROOT = 1000
SAMPLE_TEMP = 1  # as suggested by R1 paper
GROUP_SIZE = 16  # as suggested by R1 paper
MAX_RESPONSE_LENGTH = 1024

import jax
import jax.numpy as jnp
from core.gemma_forward import Params, forward_single
from utils.inspect_weights import load_weights_as_dict
from utils.tokenize import tokenize_text


def sample_with_temp(
    key: jax.random.PRNGKey, xs: jax.Array, temperature: float, params: Params
) -> jax.Array:
    """
    Only here to provide a function with the right signature for now, will later on be relocated
    to our new inference optimized forward function.

    This function should sample according to temperature until we reach the EOS token.

    CURRENT PROBLEMS:
        - no KV caching
    """

    def sampling_loop(init, carry) -> jax.Array:
        key = init
        xs, i = carry

        next_token_logits = forward_single(xs, params, i)

        next_token = jax.random.categorical(key, next_token_logits / temperature)

        xs[i] = next_token  # doesn't work in JAX, but correct logic

        return xs

    # padding
    input_length = xs.shape[0]
    padding_tokens = max(0, MAX_RESPONSE_LENGTH - input_length)
    xs = jnp.concatenate([xs, jnp.zeros_like(xs, shape=(padding_tokens,))])

    # forward scan
    key, *keys = jax.random.split(key, MAX_RESPONSE_LENGTH + 1)
    xs = jax.lax.scan(sampling_loop, (xs, 0), jnp.array(keys))

    return xs


def get_prompt(n: int) -> jax.Array:
    r"""
    Returns the tokens of a prompt to calculate the square root of n,
    wrapped in the DeepSeek-R1-Zero system template.
    """
    prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: Calculate the square root of {n} up to three decimal places. Assistant:"""

    return jnp.array(tokenize_text(prompt))


def reward(output_tokens) -> jax.Array:
    r"""
    Calculate a reward (both for correctness and for existence of thinking tags)
    given the models output.
    """
    pass


def objective_function() -> jax.Array:
    r"""
    The GRPO objective function that we can then differentiate with repect to the policy parameters \theta.

    Let G = {o_1,..., o_n} be a set (called group) of outputs for a given prompt q.
    The objective function is given by
        J(\theta) = E_{q \sim P(Q), {o_i}_{i=1}^G \sim \pi_{\theta_old}} \\
            [
                1/G \Sum_{i = 1}^G min
                (
                    r_i(\theta), A_i),
                    clip(r_i(\theta), 1-\epsilon, 1+\epsilon )
                ) A_i - \beta KL(\pi_\theta, \pi_{theta_old})
            ]

    In order to make this a little more readable, we break it up into a couple of helper functions.
    """
    pass


def advantage(r_t: jax.Array) -> jax.Array:
    r"""
    Defined as
        A_i = (r_i - mean({r_1, ..., r_G})) / std({r_1, ..., r_G})

    where r_i is the reward assigned to output o_i.

    Needs global information from the whole group.
    """
    mean = jnp.mean(r_t)
    std = jnp.std(r_t)

    return r_t * (-mean / std)


def ratio() -> jax.Array:
    r"""
    The good old PPO ratio defined as
        r_i(\theta) = ( \pi_\theta (o_i | q) ) / ( \pi_{\theta_old} (o_i | q))

    Local to a specific group element (so we vmap over the group).
    """
    pass


def prop_of_trajectory():
    r"""
    The probability of a given trajectory o_i is defined as the (conditional) probability
    of every token, so
        \pi_\theta (o | q) = \prod_{t+1}^L \pi(o_t | q, t_{<t})
    """
    pass


def KL() -> jax.Array:
    r"""
    Calculates KL(\pi_\theta, \pi_{\theta_old}).
    """
    pass


def get_group(key: jax.random.PRNGKey, group_size: int, params: Params) -> jax.Array:
    """
    Samples a group of responses.
    """
    key, subkey = jax.random.split(key)
    int_to_radicate = jax.random.randint(subkey, 1, MIN_ROOT, MAX_ROOT)

    prompt = get_prompt(int_to_radicate)  # prompt is the same for the whole group

    all_keys = jax.random.split(key, group_size + 1)
    key, traj_keys = all_keys[0], all_keys[1:]
    group = jax.vmap(lambda key: sample_with_temp(key, prompt, SAMPLE_TEMP, params))(
        traj_keys
    )

    return group


key = jax.random.PRNGKey(42)
params = load_weights_as_dict("data/model_stacked_pt.safetensors")
grp = get_group(key, 16, params)
print(grp)
