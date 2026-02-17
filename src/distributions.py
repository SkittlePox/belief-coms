import distrax, chex, jax
from jax.experimental import checkify
import jax.numpy as jnp
from functools import partial
# from typing import Optional, Array, Union, Any

class JointCategoricalPair():
    def __init__(self, vars_num_categories):
        self.var1_num_categories, self.var2_num_categories = vars_num_categories
    
    @partial(jax.jit, static_argnums=(0,))
    def marginalize_var1(self, probs_flat_distribution):
        checkify.check(jnp.sum(probs_flat_distribution.probs) == 1.0, 'Probabilities do not sum to 1.0')
        probs_2d = probs_flat_distribution.probs.reshape((self.var1_num_categories, self.var2_num_categories))
        marginalized_probs = jnp.sum(probs_2d, axis=0)
        return distrax.Categorical(probs=marginalized_probs)

    @partial(jax.jit, static_argnums=(0,))
    def marginalize_var2(self, probs_flat_distribution):
        checkify.check(jnp.sum(probs_flat_distribution.probs) == 1.0, 'Probabilities do not sum to 1.0')
        probs_2d = probs_flat_distribution.probs.reshape((self.var1_num_categories, self.var2_num_categories))
        marginalized_probs = jnp.sum(probs_2d, axis=1)
        return distrax.Categorical(probs=marginalized_probs)
    
    @partial(jax.jit, static_argnums=(0,))
    def sample_joint_distribution(self, probs_flat_distribution, key: chex.PRNGKey):
        checkify.check(jnp.sum(probs_flat_distribution.probs) == 1.0, 'Probabilities do not sum to 1.0')
        flattened_sample = probs_flat_distribution.sample(seed=key)
        var_1_cat = flattened_sample // self.var1_num_categories
        var_2_cat = flattened_sample % self.var2_num_categories
        return jnp.stack((var_1_cat, var_2_cat))


# let's imagine two variables with 3 and 5 categories respectively. A grid with 3 rows and 5 cols.

if __name__ == "__main__":
    factory = JointCategoricalPair(vars_num_categories=(3, 5))
    
    key = jax.random.PRNGKey(8)

    initial_probs = jnp.zeros(15).at[1:3].set(jnp.array([0.5, 0.5]))

    marg_dist = factory.marginalize_var1(distrax.Categorical(probs=initial_probs))

    print(marg_dist.probs)

    print(factory.sample_joint_distribution(distrax.Categorical(probs=initial_probs), key))

