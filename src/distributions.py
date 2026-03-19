import distrax, chex, jax
from jax.experimental import checkify
import jax.numpy as jnp
from functools import partial
# from typing import Optional, Array, Union, Any

class JointCategoricalPair():
    """Represents a joint distribution over two categorical variables.

    The joint distribution is stored internally as a flat `distrax.Categorical`
    with `var1_num_categories * var2_num_categories` outcomes, laid out in
    row-major order (var1 = rows, var2 = columns):

        2D view (var1=3, var2=4):       flat view (length 12):
        ┌──────────────────────┐        ┌────────────────────────────────────────────┐
        │            var2      │        │ idx:  0    1    2    3    4    5    6  ... │
        │       0   1   2   3  │        │      p00  p01  p02  p03  p10  p11  p12 ... │
        │ v 0 [p00 p01 p02 p03]│        └────────────────────────────────────────────┘
        │ a 1 [p10 p11 p12 p13]│
        │ r 2 [p20 p21 p22 p23]│        decode: flat index i
        │ 1                    │                → var1 = i // var2_num_categories
        └──────────────────────┘                → var2 = i %  var2_num_categories

    This class provides JIT-compiled helpers for marginalization and sampling,
    designed to be used as a static object (passed as `static_argnums`) inside
    JAX-traced code.

    Example:
        # Two variables: var1 with 3 categories, var2 with 5 categories.
        joint = JointCategoricalPair(vars_num_categories=(3, 5))
        probs = jnp.zeros(15).at[7].set(1.0)  # all mass on outcome (1, 2)
        flat_dist = distrax.Categorical(probs=probs)

        marginal_var2 = joint.marginalize_var1(flat_dist)  # shape (5,)
        sample = joint.sample_joint_distribution(flat_dist, key)  # [1, 2]
    """

    def __init__(self, vars_num_categories):
        """
        Args:
            vars_num_categories: A tuple `(var1_num_categories, var2_num_categories)`
                giving the number of categories for each variable. The flat
                probability vector must have length equal to their product.
        """
        self.var1_num_categories, self.var2_num_categories = vars_num_categories

    @partial(jax.jit, static_argnums=(0,))
    def marginalize_var1(self, probs_flat_distribution):
        """Return the marginal distribution of var2 by summing out var1.

        Reshapes the flat joint probability vector into a
        `(var1_num_categories, var2_num_categories)` grid and sums over the
        var1 axis (axis 0), yielding a `distrax.Categorical` over var2.

        Args:
            probs_flat_distribution: A `distrax.Categorical` whose `.probs`
                has shape `(var1_num_categories * var2_num_categories,)`.

        Returns:
            A `distrax.Categorical` over var2 with shape `(var2_num_categories,)`.
        """
        probs_2d = probs_flat_distribution.probs.reshape((self.var1_num_categories, self.var2_num_categories))
        marginalized_probs = jnp.sum(probs_2d, axis=0)
        return distrax.Categorical(probs=marginalized_probs)

    @partial(jax.jit, static_argnums=(0,))
    def marginalize_var2(self, probs_flat_distribution):
        """Return the marginal distribution of var1 by summing out var2.

        Reshapes the flat joint probability vector into a
        `(var1_num_categories, var2_num_categories)` grid and sums over the
        var2 axis (axis 1), yielding a `distrax.Categorical` over var1.

        Args:
            probs_flat_distribution: A `distrax.Categorical` whose `.probs`
                has shape `(var1_num_categories * var2_num_categories,)`.

        Returns:
            A `distrax.Categorical` over var1 with shape `(var1_num_categories,)`.
        """
        probs_2d = probs_flat_distribution.probs.reshape((self.var1_num_categories, self.var2_num_categories))
        marginalized_probs = jnp.sum(probs_2d, axis=1)
        return distrax.Categorical(probs=marginalized_probs)

    @partial(jax.jit, static_argnums=(0,))
    def sample_joint_distribution(self, key: chex.PRNGKey, probs_flat_distribution):
        """Draw one sample from the joint distribution and decode it to (var1, var2).

        Samples a flat index from the joint `distrax.Categorical`, then decodes
        it back to a `(var1, var2)` pair using the row-major convention:
            var1 = flat_index // var2_num_categories
            var2 = flat_index %  var2_num_categories

        Args:
            probs_flat_distribution: A `distrax.Categorical` whose `.probs`
                has shape `(var1_num_categories * var2_num_categories,)`.
            key: A JAX PRNG key used for sampling.

        Returns:
            A JAX array of shape `(2,)` containing `[var1_category, var2_category]`.
        """
        flattened_sample = probs_flat_distribution.sample(seed=key)
        var_1_cat = flattened_sample // self.var2_num_categories
        var_2_cat = flattened_sample % self.var2_num_categories
        return jnp.stack((var_1_cat, var_2_cat))


if __name__ == "__main__":
    # Joint distribution over var1 (3 categories) and var2 (5 categories).
    # The flat probability vector has length 3 * 5 = 15.
    factory = JointCategoricalPair(vars_num_categories=(3, 5))

    key = jax.random.PRNGKey(8)

    # Place equal mass on flat indices 1 and 2, which correspond to
    # (var1=0, var2=1) and (var1=0, var2=2) in the 3x5 grid.
    initial_probs = jnp.zeros(15).at[1:3].set(jnp.array([0.5, 0.5]))

    # Marginalize out var1: sum each column of the 3x5 grid to get P(var2).
    # Expected result: [0, 0.5, 0.5, 0, 0] since all mass is in var2 columns 1 and 2.
    marg_dist = factory.marginalize_var1(distrax.Categorical(probs=initial_probs))
    print(marg_dist.probs)

    # Sample a (var1, var2) pair from the joint. Should return [0, 1] or [0, 2].
    print(factory.sample_joint_distribution(key, distrax.Categorical(probs=initial_probs)))

