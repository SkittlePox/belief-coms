# Notes

So some important notes:

- I realized I want as flat a codebase as possible. Minimal inheritance.

I don't think the POMDPs should be aware of the communication at all. The POMDP should just have agent roles and nothing else. Communication happens outside of the POMDP. Assuming diadic games. So there's a managing class that takes a population of agents, perhaps a set of games that agents are playing, a communication scheme (takes place above each game), rules for assigning agents to games, etc. There's then an assignment of agents to games for a certain number of epochs. Those individual games are played out, with communication atop them.

Probably the communication layer is just a function that implements the interaction. I forget all the necessary components.

I think I'll incorporate as much into the state as possible. Let me imagine this managing class first.

I think I still have problems understanding the way that agents will be funneled to environments. I want to keep track of a global agent belief state. I'm wondering if the DecPOMDP should just be a set of functions or its own class with its own state object. If it's the latter, I then have to worry about juggling different belief states within the managing class. But if it's just functions then I'm limited by the complexity of them, I won't have a more general codebase. I think I should introduce hierarchy here.

What I can do is just work on a single DecPOMDP model with two agent roles. I can then work on bridging that to the greater DecPOMDP class. So i'm going with the first option, which is to make the DecPOMDP its own object. This should probably be fine.

I think it's also okay to make the agent roles rigid in the underlying DecPOMDPs... actually is it? If they were rigid, and I was running the games for like 100 episodes, that would just be one agent communicating with the other... I guess the issue is that these DecPOMDPs are also themselves episodic. I think making them rigid is actually alright, I'll just increase the frequency of the role switching in the managing class. I may want the same two agents to be playing the same diadic game with each other though, but I guess I can control for that in the agent assignment scheduler.

So each DecPOMDP should also have a belief state. Rigid roles and initial belief states at t=0.

So I'm slightly worried about the double-bookkeeping of the environments. There's a major difference between representing them in a class vs as probability distributions... I do actually think I need both... And I don't really think this is a major problem for now...

I think I'm getting stuck on this for now. I think the structure of the individual environments is actually pretty clear. It's time to build the manager environment.

I decided that the structure of the environments will largely be the same, so I'll use one large belief state representation I think. The transition and reward function will change from game to game, I wonder if that will be a problem.

Hmm okay this is very interesting. If everything is a guessing game it makes the code way easier to write because the transition function can actually be identical and parameterized by vectors in the state. This would actually be quite wise. So I should transition to code that just handles guessing games of various kinds, and keeps the relevant transition and reward dynamics info stored in a vector somewhere. I think that should be possible for a narrow set of guessing games. Time to sketch the games out...

# Running

The project is **not installed as a package**. The code uses fully-qualified
imports (`envs.*`, `tools.*`) rooted at `src/`, so run modules from inside `src/`
with `src/` on the path:

```bash
cd src
PYTHONPATH=. uv run python -m tools.returns
```

Use the dotted module path (`-m tools.returns`), not a file path. Running a file
directly (e.g. `uv run tools/returns.py`) only puts that file's folder on
`sys.path`, not `src/`, so imports like `import envs` fail with
`ModuleNotFoundError`. `PYTHONPATH=.` adds `src/` (the cwd) to the path.

Other entry points follow the same pattern, e.g.:

```bash
PYTHONPATH=. uv run python -m envs.guessing_game
PYTHONPATH=. uv run python -m envs.env_assembly
PYTHONPATH=. uv run python -m stacked_signification_decpomdp
```
