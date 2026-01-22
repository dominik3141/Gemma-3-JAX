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

import jax


def get_prompt(n: int) -> jax.Array:
    r"""
    Should return the tokens of a prompt to calculate the square root of
    the given integer.
    The model should be instructed to return the final results in a specific format
    so we can later easily check for correctness.
    """
    pass


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
    mean = jax.numpy.mean(r_t)
    std = jax.numpy.std(r_t)

    return r_t * (-mean / std)


def ratio() -> jax.Array:
    r"""
    The good old PPO ratio defined as
        r_i(\theta) = ( \pi_\theta (o_i | q) ) / ( \pi_{\theta_old} (o_i | q))

    Local to a specific group element (so we vmap over the group).
    """
    pass


def prop_of_grp_elem():
    r"""
    The probability of a given trajectory o_i is defined as the (conditional) probability
    of every token, so
        \pi_\theta (o | q) = \prod_{t+1}^L \pi(o_t | q, t_{<t})
    """
    pass


def rl_iteration(group_size):
    pass
