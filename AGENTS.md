# PRISM-FAS-B operating invariants

- Raw CASIA-FASD, MSU-MFSD, and SiW-Mv2 directories are strictly read-only. Never rename, move, delete, modify, or write frames, manifests, or caches there.
- All generated data belongs below this project root; large generated directories are ignored by Git.
- Preprocessing is local-only; never upload raw datasets to Modal. Use a single `TrainerCore` in later milestones.
- CASIA-FASD and MSU-MFSD are source only. SiW-Mv2 is target-test only and must never select a checkpoint or tune any training decision.
- Training-facing target APIs must not expose labels, attack types, taxonomies, or attack-wise metadata. Evaluation opens private labels only after predictions exist.
- Adapters use explicit YAML rules or official metadata only. Missing mapping data is a concrete failure/blocker, never a guessed layout.
- Decide splits at video/subject level before frame work. Never split frames from one video across splits.
- Use `pathlib.Path` for all Python paths, atomic writes for manifests/config/checkpoints, and explicit exception handling.
- Each milestone needs implementation, unit/integration tests, run command, acceptance report, and `PROJECT_STATUS.md` update. Do not claim completion when tests fail.
- Implement M0–M2 only when explicitly assigned. M3+ is out of scope.
