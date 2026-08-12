# PRISM-FAS-C-LLM — project invariants

LLM-guided executable artifact planning for cross-domain face anti-spoofing.
This file holds stable invariants only. Dynamic state lives in
`docs/PROJECT_STATE.md`; procedures live in `.claude/skills/`.

## Authoritative spec

    docs/PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_FullPipeline_Spec_2026.docx
    sha256 ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5

Read only the sections you need. Read it in full at v1.5 reconciliation, or when a
scientific requirement cannot be resolved from a scoped section. Never paste the
document into context wholesale.

## Authority order (higher wins; never silently reconcile a conflict — record it)

1. Frozen Version-B Git/evidence — authoritative for Version-B history only
2. The v1.5 spec — governing contract for Version C
3. Immutable Version-C locks and decision records
4. Git history and machine-readable run evidence
5. `docs/PROJECT_STATE.md` — derived handoff state
6. `CLAUDE.md`, skills, rules — execution procedure only
7. Claude auto-memory and conversation — convenience only, never scientific authority

## Hard invariants

**Version B is immutable.** `D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project` at
`7799f7decd35db6987ce4578824e5bd8d9eab4ae`, tag `m10-blind-evaluation-checkpoint`.
Read-only. Never write, never push, never let a Version-C output path point inside it.
Re-verify HEAD/tag/clean at the start and end of any milestone.

**Target firewall.** SiW-Mv2 is the fixed P3 held-out target. TRAIN / LLM / SYNTHESIS
code may never resolve SiW labels or target metrics. C11 may resolve label-free SiW
features only; C12 may resolve locked predictions plus labels but may not mutate any
model, checkpoint or calibration artifact. No prompt, recipe, architecture,
quality-gate, checkpoint or calibration choice may be informed by target results.

**C3 scientific generation is prohibited until explicitly authorized.** The 12×32
schedule producing 384 raw candidate slots per arm is the scientific bank. Disposable
pilots (C2/C2B/C2C) are not C3. Before C3 the BANK_LOCK must bind
`C3_BANK_CONTRACT_IDENTITY` = SHA256 over generation **and** selection contract
identities (§7.8.4).

**Smoke is not scientific completion.** Every milestone carries two statuses:
`engineering_status` (NOT_TESTED | RUNNING | SMOKE_PASS | SMOKE_FAIL | BLOCKED) and
`scientific_status` (NOT_RUN | RUNNING | PASS | FAIL | BLOCKED). Only
`scientific_status=PASS` under `--profile full` is scientific completion. Every smoke
artifact serializes `execution_profile="smoke"` and `scientific_eligible=false`. A
smoke number may never select a winner, change a hypothesis or support a claim.

**No silent science changes.** Do not alter scientific constants, ontology, prompt,
schema, route policy, quotas, selection rule, seed family or acceptance criteria to
make a test pass or a result look better. Frozen values change only by explicit user
decision, recorded with old identity, new identity, reason and affected-request count.

**No target-guided tuning.** Source-only adaptation is permitted strictly inside the
envelopes the spec authorizes, before source freeze.

**Secrets.** Check presence only (`PRESENT`/`MISSING`). Never print, log, serialize or
commit a key. `GEMINI_API_KEY` is read only by the provider at call time.

## Git safety

Never rewrite history, force push, rebase or squash historical milestone commits.
Never merge to `main`. Never delete a historical lock or artifact. Work on a milestone
branch; commit and push normally only when acceptance conditions are satisfied.

## Tests

Milestone suites are offline: they block sockets and delete ambient credentials, and
use mock/replay providers. Before running a broad suite, confirm it cannot call a live
provider, launch Modal/GPU, or read SiW labels. Record exact passed/failed/skipped.
Do not hide the documented inherited failures; do not edit scientific code to turn them
green.

## Resume and idempotency

`--resume` is identity-aware. A completed artifact is skipped only after its parent
identities, config identity, content hash and acceptance state validate. A valid frozen
C3 archive or GPAT checkpoint must never be regenerated merely because the orchestrator
restarted. Writes to archives, checkpoints, locks and pipeline state are atomic. If an
expected identity changed, fail closed and compute the invalidation subtree.

## Canonical entrypoint (planned; see `docs/V15_PIPELINE_RESTRUCTURE_PLAN.md`)

    python train.py --profile validate
    python train.py --profile smoke --resume
    python train.py --profile full --resume

`train.py` orchestrates C0→C13 and delegates to `src/prism_fas/`. It must not absorb
recipe, GPAT, synthesis, detector, evaluation or lock logic.

## Pointers

- `docs/PROJECT_STATE.md` — current milestone, statuses, identities, next action
- `.claude/skills/prism-milestone/SKILL.md` — execute one authorized milestone
- `.claude/skills/prism-handoff/SKILL.md` — session handoff and context recovery
- `reports/v15_reconciliation/` — v1.5 audit, matrices, blockers

## STOP conditions

Stop and report rather than proceeding when:

- a live provider call, GPU job or target-label access is attempted by a path that
  should not do it;
- Version B is not at the expected HEAD/tag, or its tree is dirty;
- the spec SHA-256 does not match;
- an identity or lock disagrees with the code that claims to produce it;
- a gate needs a scientific decision outside a preregistered envelope
  (`NEEDS_SCIENTIFIC_DECISION`);
- the action is not the single `next_authorized_action` in `PROJECT_STATE.md`.
