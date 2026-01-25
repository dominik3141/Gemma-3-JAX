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
MAX_RESPONSE_LENGTH = 250

import re
import math
import jax
import jax.numpy as jnp
from core.gemma_forward import Params
from core.gemma_forward_inference import forward_single, get_KV
from utils.inspect_weights import load_weights_as_dict
from utils.tokenize import tokenize_text, detokenize_ids


def sample_with_temp(
    key: jax.random.PRNGKey,
    params: Params,
    final_prompt_token: jax.Array,
    pos: int,  # position of the last prompt token <=> length of prompt
    K_cache: jax.Array,
    V_cache: jax.Array,
    temperature: float,
    sample_length: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Sample according to a given temperature for a fixed length.
    We return both the autoregressively completed sequence and the probability of the trajectory at
    sampling time.

    For efficency, this function already takes the K,V cache of the prompt and only the last prompt token.
    This way we can calculate the KV of the prompt only once and reuse this for every group element.
    """

    def sample_loop(carry, scans):
        x, K_cache, V_cache = carry
        pos, key = scans

        logits, K_cache, V_cache = forward_single(x, params, pos, K_cache, V_cache)

        # sample next token
        scaled_logits = logits / jnp.maximum(
            temperature, 1e-8
        )  # to support temperature=0
        x = jax.random.categorical(key, scaled_logits)
        log_prop_of_next_token = jax.nn.log_softmax(scaled_logits)[x]

        return (x, K_cache, V_cache), (x, log_prop_of_next_token)

    poss = jnp.arange(pos, pos + sample_length)
    keys = jax.random.split(key, sample_length)
    _, (xs, log_probs) = jax.lax.scan(
        sample_loop, (final_prompt_token, K_cache, V_cache), (poss, keys)
    )

    return xs, log_probs


def get_prompt(n: int) -> jax.Array:
    r"""
    Returns the tokens of a prompt to calculate the square root of n,
    wrapped in the DeepSeek-R1-Zero system template.
    """
    prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: Calculate the square root of {n} up to three decimal places. Assistant:"""

    return jnp.array(tokenize_text(prompt))


def reward(output_tokens: list[int], int_to_radicate: int) -> float:
    r"""
    Calculate a reward (both for correctness and for existence of thinking tags)
    given the models output.

    Would be very nice if we could do this in pure JAX, but so far I have no idea how
    one could practically do so.
    """
    text = detokenize_ids(output_tokens)

    # Tuning Parameters
    FORMAT_WEIGHT = 0.1
    CORRECTNESS_WEIGHT = 1.0

    # Initialize scores
    format_score = 0.0
    correctness_score = 0.0

    # 1. Check Format
    # We look for the specific structure: <think> ... </think> <answer> ... </answer>
    match = re.search(r"<think>.*?</think>\s*<answer>(.*?)</answer>", text, re.DOTALL)

    if match:
        format_score = 1.0

        # 2. Check Correctness (only possible if format is valid)
        try:
            prediction_str = match.group(1).strip()
            prediction = float(prediction_str)
            target = math.sqrt(int_to_radicate)

            # Check if close enough
            if abs(prediction - target) < 1e-3:
                correctness_score = 1.0

        except ValueError:
            # The number inside <answer> wasn't a valid float
            correctness_score = 0.0

    # Combine
    return (format_score * FORMAT_WEIGHT) + (correctness_score * CORRECTNESS_WEIGHT)


def objective_function() -> jax.Array:
    r"""
    The GRPO objective function that we can then differentiate with respect to the policy parameters \theta.

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

    return (r_t - mean) / (std + 1e-8)


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


def get_group(
    key: jax.random.PRNGKey, group_size: int, params: Params
) -> tuple[jax.Array, jax.Array]:
    """
    Samples a group of responses.

    TODO:
        -   We should make sure to only once per group calculate the KV cache for the prompt
            that can save us a little compute.
    """
    key, subkey = jax.random.split(key)
    int_to_radicate = jax.random.randint(subkey, 1, MIN_ROOT, MAX_ROOT)

    prompt = get_prompt(int_to_radicate)  # prompt is the same for the whole group

    # calculate the KV cache of the prompt
    K_cache, V_cache = get_KV(prompt, params, MAX_RESPONSE_LENGTH)

    all_keys = jax.random.split(key, group_size + 1)
    key, group_keys = all_keys[0], all_keys[1:]
    responses, log_probs = jax.vmap(
        lambda key: sample_with_temp(
            key,
            params,
            prompt[-1],
            len(prompt) - 1,
            K_cache,
            V_cache,
            1,
            MAX_RESPONSE_LENGTH - len(prompt),
        )
    )(group_keys)

    return responses, log_probs  # doesn't contain values for prompt


key = jax.random.PRNGKey(42)
params = load_weights_as_dict("data/model_stacked_pt.safetensors")

grp, log_probs = get_group(key, 8, params)

for i, response_tokens in enumerate(grp):
    text = detokenize_ids(response_tokens.tolist())
    print(f"Response {i}:\n{text}\n{'-' * 20}")
