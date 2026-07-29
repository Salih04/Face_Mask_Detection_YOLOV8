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

On 2026-07-29, the credential was removed from the current public tree and the
reachable `main` history without being displayed, tested, used, authenticated,
submitted to a service, or attributed to an account. W&B Security was notified
without the credential value. A separate, unused W&B GitHub OAuth authorization
was revoked.

Provider revocation of the unknown credential is not yet confirmed. Historical
removal is not complete until GitHub Support finishes cached-view, hosted-
reference, and server-side garbage-collection cleanup. Copies outside GitHub's
control, including pre-existing clones, cannot be recalled by this repository
rewrite.
