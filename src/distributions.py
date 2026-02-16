import distrax, chex, jax
import jax.numpy as jnp
# from typing import Optional, Array, Union, Any

class JointCategoricalPairFactory():
    def __init__(self, var1_num_categories, var2_num_categories):
        self.var1_num_categories = var1_num_categories
        self.var2_num_categories = var2_num_categories

        self.underlying_probs = jnp.zeros((var1_num_categories, var2_num_categories))
        self.flat_joint_distribution = distrax.Categorical(logits=jnp.ones(var1_num_categories * var2_num_categories))
        self.internal_distribution_is_stale = False
    
    def set_probs(self, var1_idx, var2_idx, probability):
        self.underlying_probs = self.underlying_probs.at[var1_idx, var2_idx].set(probability)
        self.internal_distribution_is_stale = True
    
    def add_probs(self, var1_idx, var2_idx, probability):
        self.underlying_probs = self.underlying_probs.at[var1_idx, var2_idx].set(
            probability + self.underlying_probs[var1_idx][var2_idx])
        self.internal_distribution_is_stale = True
    
    def make_distribution(self):
        # Need to do something like this:
        # jax.debug.assert(jnp.sum(self.underlying_probs) == 1.0)
        self.flat_joint_distribution = distrax.Categorical(probs=self.underlying_probs.flatten())
        self.internal_distribution_is_stale = False

    def marginalize_var1(self):
        # Need to do something like this:
        # jax.debug.assert(jnp.sum(self.underlying_probs) == 1.0)
        marginalized_probs = jnp.sum(self.underlying_probs, axis=0)
        return distrax.Categorical(probs=marginalized_probs)
    
    def marginalize_var2(self):
        # Need to do something like this:
        # jax.debug.assert(jnp.sum(self.underlying_probs) == 1.0)
        marginalized_probs = jnp.sum(self.underlying_probs, axis=1)
        return distrax.Categorical(probs=marginalized_probs)
    
    def sample_joint_distribution(self, key: chex.PRNGKey):
        # Assert self.flat_joint_distribution is not None and not internal_distribution_is_stale
        flattened_sample = self.flat_joint_distribution.sample(seed=key)
        var_1_cat = flattened_sample // self.var1_num_categories
        var_2_cat = flattened_sample % self.var2_num_categories
        return jnp.stack((var_1_cat, var_2_cat))
    
    def __str__(self) -> str:
        return str(self.underlying_probs)

# let's imagine two variables with 3 and 5 categories respectively. A grid with 3 rows and 5 cols.

if __name__ == "__main__":
    factory = JointCategoricalPairFactory(var1_num_categories=3, var2_num_categories=5)
    
    key = jax.random.PRNGKey(8)

    factory.set_probs(0, 1, 0.5)
    factory.set_probs(0, 2, 0.5)

    factory.make_distribution()

    print(factory)

    print(factory.sample_joint_distribution(key))
