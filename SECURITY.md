# Security

## Credential handling

Real credentials must never be committed to this repository, placed in example
commands, stored in plain-text configuration, or embedded in notebooks, logs,
screenshots, or generated artifacts.

If Weights & Biases is legitimately required in the future:

- supply authentication through secure environment configuration such as
  `WANDB_API_KEY`;
- keep local environment files untracked;
- use `WANDB_MODE=disabled` when tracking is not required;
- use `WANDB_MODE=offline` only when local run artifacts are intentionally
  required and can be handled safely;
- never pass a credential as a command-line argument.

## 2026-07-29 incident status

An unknown W&B credential was discovered in the public repository and its Git
history. Salih does not have a W&B account. The credential owner and validity
are unknown.

The credential was removed from this isolated corrected tree without being
displayed, tested, used, authenticated, submitted to a service, or attributed
to an account.

This isolated correction does not contain the live public repository. Provider
revocation is not confirmed. Historical removal is not complete on GitHub until
the rewritten branches and tags are published and GitHub Support completes any
required pull-request-reference, cached-view, garbage-collection, and LFS
cleanup.
