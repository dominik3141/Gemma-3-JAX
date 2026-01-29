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
"""

MAX_ROOT = 90000000
MIN_ROOT = 1000
SAMPLE_TEMP = 1  # as suggested by R1 paper
GROUP_SIZE = 8  # as suggested by R1 paper
MAX_RESPONSE_LENGTH = 250
EPSILON = 0.1
BETA = 0.001  # as suggested by R1 paper
NUM_GROUPS_PER_UPDATE = 32  # as suggested by R1 paper
LEARNING_RATE = (
    (GROUP_SIZE / 16) * (NUM_GROUPS_PER_UPDATE / 32) * 3e-6
)  # as suggested by R1 paper

import re
import math
import jax
import jax.numpy as jnp
import optax
from core.gemma_forward import Params, forward
from core.gemma_forward_inference import forward_single, get_KV
from utils.inspect_weights import load_weights_as_dict
from utils.tokenize import tokenize_text, detokenize_ids
import functools


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

    For efficiency, this function already takes the K,V cache of the prompt and only the last prompt token.
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

    poss = jnp.arange(sample_length) + pos  # [pos+0, pos+1,...,pos+same_length]
    keys = jax.random.split(key, sample_length)
    _, (xs, log_probs) = jax.lax.scan(
        sample_loop, (final_prompt_token, K_cache, V_cache), (poss, keys)
    )

    return xs, log_probs


def get_prompt(n: int) -> jax.Array:
    r"""
    Returns the tokens of a prompt to calculate the square root of n,
    wrapped in the DeepSeek-R1-Zero system template.

    This function is implemented purely in JAX to avoid `jax.pure_callback` or host-device transfers,
    ensuring it can be JIT-compiled and traced efficiently.

    Mechanism:
    1.  We decompose the integer `n` into its decimal digits using basic arithmetic:
        `digits = (n // 10^i) % 10` for i in [7..0].
    2.  We map these digits [0-9] to their corresponding token IDs using a lookup table.
        We verified that the tokenizer treats digits as individual tokens (no merging like "12" -> one token)
        and that they don't merge with the prefix/suffix.
    3.  We concatenate [BOS] + prefix + digit_tokens + suffix.

    This guarantees a fixed shape output (112 tokens) and ensures that we don't break the JAX tracer.
    """
    prefix_str = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The answer should only include the numerical result and nothing else.
User: Calculate the square root of """
    suffix_str = " up to three decimal places. Assistant"

    # These tokenizations happen during tracing (constant folding) or eagerly (fast enough)
    prefix_tokens = jnp.array(tokenize_text(prefix_str), dtype=jnp.int32)
    suffix_tokens = jnp.array(tokenize_text(suffix_str), dtype=jnp.int32)

    # Tokens for digits 0-9: ["0", "1", ..., "9"]
    # We use hardcoded IDs to ensure stability, verified to be single tokens.
    digit_tokens_map = jnp.array(
        [
            236771,
            236770,
            236778,
            236800,
            236812,
            236810,
            236825,
            236832,
            236828,
            236819,
        ],
        dtype=jnp.int32,
    )

    # Calculate digits for 8-digit zero-padded number
    # n is a JAX scalar integer
    powers = 10 ** jnp.arange(7, -1, -1)  # [10000000, ..., 1]
    digits = (n // powers) % 10  # shape (8,)

    n_tokens = digit_tokens_map[digits]

    return jnp.concatenate(
        [
            jnp.array([2], dtype=jnp.int32),
            prefix_tokens,
            n_tokens,
            suffix_tokens,
        ]
    )


def _impure_reward_fn(
    output_tokens: jax.Array, int_to_radicate: int
) -> tuple[float, float, float, int]:
    r"""
    Calculate a reward (both for correctness and for existence of thinking tags)
    given the models output.

    Would be very nice if we could do this in pure JAX, but so far I have no idea how
    one could practically do so.

    The code in this function is AI maintained and not exactly beautiful.
    """
    # default end position is the end of the sequence
    end_pos = len(output_tokens)
    text = detokenize_ids(output_tokens.tolist())

    # Tuning Parameters
    # same coefficients for format and correctness according to R1 paper (p.4)
    FORMAT_WEIGHT = 1.0
    CORRECTNESS_WEIGHT = 1.0

    # 1. Strict Tag Count Check
    if (
        text.count("<think>") != 1
        or text.count("</think>") != 1
        or text.count("<answer>") != 1
        or text.count("</answer>") != 1
    ):
        return 0.0, 0.0, 0.0, end_pos

    # 2. Strict Format Check
    # Must start with <think>, allow newlines in thinking, no newlines in answer.
    # We ignore everything after the first </answer> by not anchoring to the end.
    match = re.search(
        r"^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>", text, re.DOTALL
    )

    if not match:
        return 0.0, 0.0, 0.0, end_pos

    # find the end position in tokens
    tokens = output_tokens.tolist()
    # We search for the sequence </answer>.
    # From our tests, answer> is [14433, 236813] and </ is 954 or 1454 (with space).
    for i in range(len(tokens) - 1):
        if tokens[i : i + 2] == [14433, 236813]:
            if i > 0 and tokens[i - 1] in [954, 1454]:
                end_pos = i + 2
                break

    format_score = 1.0
    correctness_score = 0.0

    # 3. Content Validation
    answer_raw = match.group(2)

    # Enforce no newlines in the raw answer content
    if "\n" in answer_raw:
        return 0.0, 0.0, 0.0, end_pos

    prediction_str = answer_raw.strip()

    # 4. Correctness Check
    try:
        prediction = float(prediction_str)
        target = math.sqrt(float(int_to_radicate))

        # Check if close enough
        if abs(prediction - target) < 1e-3:
            correctness_score = 1.0

    except ValueError:
        correctness_score = 0.0

    # Combine
    reward = (format_score * FORMAT_WEIGHT) + (correctness_score * CORRECTNESS_WEIGHT)
    return float(reward), float(format_score), float(correctness_score), int(end_pos)


def reward_fn(
    output_tokens: jax.Array, int_to_radicate: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return jax.pure_callback(
        _impure_reward_fn,
        (
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.int32),
        ),
        output_tokens,
        int_to_radicate,
        vmap_method="sequential",
    )


def objective_function(
    params: Params,
    group: jax.Array,
    int_to_radicate: int,
    prompt: jax.Array,
    theta_old_log_probs: jax.Array,
    params_ref: Params,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    r"""
    The GRPO objective function that we can then differentiate with respect to the policy parameters \theta.

    Let G = {o_1,..., o_n} be a set (called group) of outputs for a given prompt q.
    The objective function is given by
        J(\theta) = E_{ q \sim P(Q), {o_i}_{i=1}^G \sim \pi_{\theta_old} } \\
            [
                1/n \Sum_{i = 1}^n (
                    min (
                        \rho_i A_i,
                        clip(\rho_i , 1-\epsilon, 1+\epsilon ) * A_i
                    )
                    - \beta KL(\pi_\theta, \pi_{ref})
                )
            ]

    In order to make this a little more readable, we break it up into a couple of helper functions.
    """
    prompt_len = prompt.shape[0]

    (
        rewards,
        format_scores,
        correctness_scores,
        end_of_answer_pos_relative,
    ) = jax.vmap(reward_fn, in_axes=(0, None))(
        group, int_to_radicate
    )  # shape [n,], where n is group size
    end_of_answer_pos = prompt_len + end_of_answer_pos_relative

    # calculate trajectory probabilites under current parameters
    theta_log_probs = jax.vmap(log_prop_of_trajectory, in_axes=(None, 0, None))(
        params, group, prompt
    )  # shape [group_size, traj_len]

    # we add the prompt token probabilities to our sample time log probs
    # just so they have the same shape as the new log probs
    theta_old_log_probs = jax.vmap(
        lambda x: jnp.concatenate([jnp.zeros((prompt_len - 1,)), x])
    )(theta_old_log_probs)

    # masking of both the prompt tokens and the post-answer tokens
    theta_log_probs = jax.vmap(mask_fn, in_axes=(0, 0, None))(
        theta_log_probs, end_of_answer_pos, prompt_len
    )
    theta_old_log_probs = jax.vmap(mask_fn, in_axes=(0, 0, None))(
        theta_old_log_probs, end_of_answer_pos, prompt_len
    )

    # now we can reduce the log_probs to a single log_prob per trajectory
    theta_traj_log_probs = jax.vmap(jnp.sum)(theta_log_probs)
    theta_old_traj_log_probs = jax.vmap(jnp.sum)(theta_old_log_probs)

    # calculate advantage (needs full group)
    advantage = advantage_fn(rewards)

    # calculate ratio
    ratios = jax.vmap(ratio_fn)(theta_traj_log_probs, theta_old_traj_log_probs)

    # clipped objective
    # min(rho * A, clip(rho, 1-eps, 1+eps) * A)
    unclipped = ratios * advantage
    clipped = jnp.clip(ratios, 1.0 - EPSILON, 1.0 + EPSILON) * advantage

    # KL penalty
    kl_penalty = KL(params_ref, group, prompt, theta_log_probs, end_of_answer_pos)
    kl_penalty = jnp.sum(kl_penalty, axis=-1)  # sum over sequence length

    # We take the minimum of the two
    # J_theta = mean(min(unclipped, clipped) - beta * KL)
    J_theta = jnp.mean(jnp.minimum(unclipped, clipped) - BETA * kl_penalty)

    # Calculate metrics
    mean_format = jnp.mean(format_scores)
    mean_correctness = jnp.mean(correctness_scores)

    return -J_theta, (mean_format, mean_correctness)


def mask_fn(
    log_probs: jax.Array, answer_end_pos: jax.Array, prompt_len: int
) -> jax.Array:
    """
    Mask out prompt tokens as well as post answer tokens.
    """
    log_probs_len = log_probs.shape[0]

    prompt_mask = jnp.arange(log_probs_len) >= (prompt_len - 1)  # [0,0,...,1,1,...]
    post_answer_mask = jnp.arange(log_probs_len) < (answer_end_pos - 1)
    mask = prompt_mask & post_answer_mask

    masked_log_probs = jnp.where(mask, log_probs, 0.0)

    return masked_log_probs


def advantage_fn(rewards: jax.Array) -> jax.Array:
    r"""
    Defined as
        A_i = (r_i - mean({r_1, ..., r_G})) / std({r_1, ..., r_G})

    where r_i is the reward assigned to output o_i.

    Needs global information from the whole group.
    """
    mean = jnp.mean(rewards)
    std = jnp.std(rewards)

    return (rewards - mean) / (std + 1e-8)


def ratio_fn(
    theta_traj_log_prob: jax.Array, theta_old_traj_log_prob: jax.Array
) -> jax.Array:
    r"""
    The good old PPO ratio defined as
        \rho_i(\theta) = ( \pi_\theta (o_i | q) ) / ( \pi_{\theta_old} (o_i | q)) ,
    where i is a group element and not a single trajectory step.
    The paper is quite vocal about calculating the ratios per trajectory and not per token (as
    PPO would do).

    Local to a specific group element (so we vmap over the group).
    """
    return jnp.exp(theta_traj_log_prob - theta_old_traj_log_prob)


def log_prop_of_trajectory(params: Params, trajectory: jax.Array, prompt: jax.Array):
    r"""
    The probability of a given trajectory o_i is defined as the (conditional) probability
    of every token, so
        \pi_\theta (o | q) = \prod_{t+1}^L \pi(o_t | q, t_{<t})

    The calculation of this probability must be differentiable.
    For this we can use the forward function that we originally had build for pretraining as we
    need to calculate the (probability of a) next token for a sequence with ground truth.
    """
    xs = jnp.concatenate([prompt, trajectory])

    # get the logits for the full trajectory
    logits = forward(xs, params)

    # use same temperature as at sample time
    logits = logits / SAMPLE_TEMP

    # do the gather (expensive!)
    # logits[:-1] predicts xs[1:]
    log_probs = jax.nn.log_softmax(logits[:-1])
    targets = jnp.expand_dims(xs[1:], axis=-1)  # shape (seq_len, 1)
    log_probs_taken = jnp.take_along_axis(log_probs, targets, axis=-1).squeeze(-1)

    # we return the list of probabilities of every transition between tokens
    # we have to make sure to later mask out the prompt tokens which should have
    # probability 1
    return log_probs_taken


def KL(
    params_ref: Params,
    group: jax.Array,
    prompt: jax.Array,
    theta_log_probs: jax.Array,
    end_of_answer_pos: jax.Array,
) -> jax.Array:
    r"""
    Calculates KL(\pi_{ref} || \pi_\theta) per token.

    We use the approximation derived from f-divergence:
        KL(p || q) \approx p/q - log(p/q) - 1
    where p is \pi_{ref} and q is \pi_{\theta}.
    """
    prompt_len = prompt.shape[0]

    # we need to calculate the log probs of the reference model
    ref_log_probs = jax.vmap(log_prop_of_trajectory, in_axes=(None, 0, None))(
        params_ref, group, prompt
    )
    # same masking as for theta_log_probs
    ref_log_probs = jax.vmap(mask_fn, in_axes=(0, 0, None))(
        ref_log_probs, end_of_answer_pos, prompt_len
    )

    log_ratio = ref_log_probs - theta_log_probs
    ratio = jnp.exp(log_ratio)

    return ratio - log_ratio - 1


def get_group(
    key: jax.random.PRNGKey, group_size: int, params: Params
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Samples a group of responses.
    """
    key, subkey = jax.random.split(key)
    int_to_radicate = jax.random.randint(subkey, (), MIN_ROOT, MAX_ROOT)

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
            SAMPLE_TEMP,
            MAX_RESPONSE_LENGTH - len(prompt),
        )
    )(group_keys)

    return (
        responses,
        log_probs,
        int_to_radicate,
        prompt,
    )


def train_inner_loop(
    key: jax.random.PRNGKey, params: Params, params_ref: Params
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # sample a group
    grp, theta_old_log_probs, int_to_radicate, prompt = get_group(
        key, GROUP_SIZE, params
    )

    # put into objective function
    (loss, (mean_format, mean_correctness)), grads = jax.value_and_grad(
        objective_function, has_aux=True
    )(params, grp, int_to_radicate, prompt, theta_old_log_probs, params_ref)

    return loss, mean_format, mean_correctness, grads


@jax.jit
def train_loop(
    key: jax.random.PRNGKey,
    params: Params,
    params_ref: Params,
    optimizer_state: optax.OptState,
) -> tuple[Params, jax.Array, jax.Array, jax.Array, optax.OptState]:
    keys = jax.random.split(key, NUM_GROUPS_PER_UPDATE)
    inner_loop_partial = functools.partial(
        train_inner_loop, params=params, params_ref=params_ref
    )
    # Use scan to accumulate gradients sequentially to save memory
    grads_accum = jax.tree_util.tree_map(jnp.zeros_like, params)

    def scan_body(accum, key):
        loss, fmt, cor, grads = inner_loop_partial(key)
        new_accum = jax.tree_util.tree_map(jnp.add, accum, grads)
        return new_accum, (loss, fmt, cor)

    accumulated_grads, (losses, format_scores, correctness_scores) = jax.lax.scan(
        scan_body, grads_accum, keys
    )

    # Average the grads
    accumulated_grads = jax.tree_util.tree_map(
        lambda g: g / NUM_GROUPS_PER_UPDATE, accumulated_grads
    )

    # update parameters
    grad_updates, new_optimizer_state = optax.adam(LEARNING_RATE).update(
        accumulated_grads, optimizer_state, params
    )
    new_params = jax.tree_util.tree_map(lambda p, u: p + u, params, grad_updates)

    return (
        new_params,
        jnp.mean(losses),
        jnp.mean(format_scores),
        jnp.mean(correctness_scores),
        new_optimizer_state,
    )


from utils.save_params import save_params


def main():
    key = jax.random.PRNGKey(42)
    params = load_weights_as_dict("data/model_stacked_pt.safetensors")

    # initial adam state
    optimizer_state = optax.adam(LEARNING_RATE).init(params)

    params_ref = params
    i = 0
    while True:
        params, loss, format_pct, correct_pct, optimizer_state = train_loop(
            key, params, params_ref, optimizer_state
        )
        key, _ = jax.random.split(key)
        print(
            f"{i}, Loss: {loss}, Format: {format_pct * 100:.2f}%, Correct: {correct_pct * 100:.2f}%"
        )
        i += 1

        if i % 100 == 0:
            save_params(params)

        if i % 400 == 0:
            params_ref = params


if __name__ == "__main__":
    main()
