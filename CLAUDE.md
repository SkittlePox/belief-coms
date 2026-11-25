# Working in this repo

## No workarounds: fix the producer

When a function, observation or interface hands a consumer the wrong thing, fix it at the
source. Do not route around it in the consumer, for example by reading state directly
because an observation is masked, re-normalizing a value another component broke, or adding
a comment explaining why an input "can't be trusted". If the source can't be fixed within
the current task, stop and raise it instead of writing the workaround.

Workarounds make the codebase harder for humans to work with: the real data flow stops
matching the apparent one. When you find an existing workaround, point it out.
