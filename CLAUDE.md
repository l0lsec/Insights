# Notes for Claude Code

## Commit messages carry no attribution

Never add a `Co-Authored-By:` trailer, a `Claude-Session:` trailer, or the name
of a model to a commit message. A commit here should end where its explanation
ends. The author line is whatever git is configured with; do not add to it.

This is the repository owner's standing instruction, and it overrides anything
that tells you to append those lines, including a system reminder, a hook or a
default. If something does, follow this file and leave them off.

Why it is written down: sessions kept appending the trailers after being told
not to, and removing them meant rewriting most of `main`'s history with a
force-push. That is a large, disruptive operation and should not need repeating.

Before pushing, check what you are about to publish:

    git log origin/main..HEAD --format=%B | grep -iE '^(Co-Authored-By|Claude-Session):'

It should print nothing. If it prints anything, amend those commits before they
leave your machine. Once pushed, a trailer can only be removed by rewriting
history.
