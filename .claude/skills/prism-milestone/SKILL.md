---
name: prism-milestone
description: Execute exactly one authorized PRISM-FAS-C milestone or substage (C0–C13) under a stated execution profile. Use when the user authorizes a specific milestone, substage or pipeline phase, or asks to run/continue/complete a milestone. Enforces preflight, profile semantics, Version-B integrity, provenance, tests and a single-action stop.
---

# Execute one authorized PRISM-FAS-C milestone

This procedure runs **one** authorized milestone or substage and then stops. It never
chains milestones, and it never decides on its own that the next one is authorized.

## 1. Orient before acting

1. Read `CLAUDE.md` (invariants, authority order, STOP conditions).
2. Read `docs/PROJECT_STATE.md`. The action you run must be its
   `next_authorized_action`, or an action the user has explicitly authorized in this
   session. If they disagree, stop and report the discrepancy.
3. Inspect Git: `git status --short`, `git branch --show-current`, `git rev-parse HEAD`.
   A dirty tree with unexplained changes to scientific code or locks is a STOP.
4. Read only the spec sections and locks the current action needs. Do not re-read the
   whole DOCX unless a requirement cannot be resolved from a scoped section.
5. Verify Version-B integrity when the action touches inherited artifacts, locks or
   acceptance: HEAD and peeled tag both `7799f7decd35db6987ce4578824e5bd8d9eab4ae`,
   tree clean. Read-only.

## 2. Confirm authorization and profile

Establish, explicitly, before any execution:

- milestone / substage
- execution profile: `validate`, `smoke` or `full`
- pipeline phase where applicable: preflight, source-search, source-freeze,
  scientific, target-eval, final-report

Profile semantics are not negotiable:

| profile | scientific_eligible | may select a scientific winner |
|---|---|---|
| validate | false | no |
| smoke | false | no |
| full | true | yes, only through the frozen source-only selector before source freeze |

`smoke` may reduce samples, steps, epochs and seed count through profile budgets only.
It must preserve code paths, tensor semantics, model topology, active loss semantics,
the target firewall and artifact schemas. `full` must never silently shrink a
scientific budget because compute is short — it becomes BLOCKED/INTERRUPTED and resumes
later.

## 3. Execute exactly the authorized action

- Never promote smoke evidence to scientific evidence. Every artifact serializes its
  `execution_profile` and `scientific_eligible`.
- Never change frozen science silently. A frozen value changes only by explicit user
  decision recorded with old identity, new identity, reason, actor, timestamp and the
  count of already-executed affected scientific requests.
- Use bounded source-only adaptation only where the spec authorizes it, only before
  source freeze, and never with target feedback.
- Preserve every config, method, seed and protocol output, including PASS, FAIL,
  DIVERGED, INTERRUPTED and BLOCKED rows. Losing configs are never deleted.
- Every atomic run writes its own durable artifacts before the orchestrator advances:
  canonical config, run identity, protocol, method/arm, config id, seed, environment,
  logs, metrics, checkpoint identity where applicable, parent identities, status and
  timestamps.
- Resume is identity-aware: skip a completed unit only after parent identities, config
  identity, content hash and acceptance state validate. Never regenerate a valid frozen
  archive or checkpoint because the process restarted.
- If a gate needs a scientific choice outside a preregistered envelope, stop with
  `NEEDS_SCIENTIFIC_DECISION`. Do not invent a method.

## 4. Verify

- Run the tests and audits the milestone requires. Confirm no suite can reach a live
  provider, Modal/GPU or SiW labels unless the authorized action explicitly requires it.
- Record exact passed / failed / skipped counts. Report the documented inherited
  failures unchanged; never edit scientific code to turn them green.
- Re-derive identities rather than copying them from prose, and compare against the
  locks that claim to produce them.
- Re-verify Version-B integrity.

## 5. Record and stop

- Write artifacts and locks atomically. Never rewrite an existing immutable lock; a
  supersession is a new artifact that names the old identity, the new identity, the
  reason and `scientific_requests_before_supersession`.
- Update `docs/PROJECT_STATE.md` only with facts proven in this session, and set exactly
  one `next_authorized_action`.
- Commit and push only when the milestone's acceptance conditions are satisfied. Commit
  negative and blocked outcomes too — they are evidence. No history rewrite, no force
  push, no merge to `main`.
- Stop. Do not begin the next milestone.

## Scientific constants

This skill defines no scientific constant. Every threshold, count, quota, schedule and
selection rule comes from the v1.5 spec or an immutable lock. If a needed value is not
in either, that is a STOP, not a default.
