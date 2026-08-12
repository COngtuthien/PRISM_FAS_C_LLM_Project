# v1.5 reconciliation report

Read-only audit of the PRISM-FAS-C-LLM repository against
`PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_FullPipeline_Spec_2026.docx`.
No scientific generation, GPU work, target access or lock supersession was performed.

Machine-readable companions: `V15_PREFLIGHT.json`, `V15_HISTORICAL_MILESTONE_AUDIT.json`,
`V15_RECONCILIATION_MATRIX.json`, `V15_CLAUDE_CONTEXT_AUDIT.json`.

---

## A. Preflight

| item | value |
|---|---|
| repository | `D:\AI on IOT\Anti_spoofing\PRISM_FAS_C_LLM_Project` |
| branch at start | `c3-generation-bank-lock` |
| HEAD at start | `50514cef4b1105bfaf6b3a97c44d1e25588c6448` |
| working tree | clean — no unexplained modifications to scientific code or locks |
| origin | `https://github.com/COngtuthien/PRISM_FAS_C_LLM_Project.git` (Version-C) |
| version-b remote | fetch only; push URL `DISABLED_NO_PUSH_TO_VERSION_B` |
| reconciliation branch | `v15-spec-reconciliation`, created from `50514ce` |

Branches present: `main`, `c0-spec-reconciliation`, `c1-llm-provider-contract`,
`c2-llm-pilot`, `c2b-batch-shape-validation`, `c2c-route-contract-freeze`,
`c3-generation-bank-lock`, all with matching `origin/` refs. Tags are the inherited
Version-B set (`m2…m10`); Version C has created no tag.

## B. v1.5 spec verification

| item | value |
|---|---|
| expected SHA-256 | `ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5` |
| actual SHA-256 | `ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5` |
| match | **yes** |
| copied to | `docs/PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_FullPipeline_Spec_2026.docx` |
| copy SHA-256 | identical |
| source file | preserved, not moved |

The source file was locked by another process; it was read with shared access rather than
requiring it to be closed. **Read in full**: 741 blocks, 684 paragraphs, 57 tables,
169,586 characters, 211 headings indexed.

> **Deviation recorded.** Appendix M.1 places the spec at `docs/specs/`; the bootstrap
> instruction directed `docs/`. Followed the instruction and recorded the conflict rather
> than silently reconciling it. Move scheduled in the restructure plan.

## C. Version-B integrity

| check | result |
|---|---|
| `B_HEAD_MATCH` | **true** — `7799f7decd35db6987ce4578824e5bd8d9eab4ae` |
| `B_TAG_MATCH` | **true** — `m10-blind-evaluation-checkpoint` peels to the same commit |
| `B_TREE_CLEAN` | **true** |
| writes performed | none — read-only throughout |

Tag object is `b78e3e076bdd4078db98b348c07a322dc0912e7e` (annotated), peeling to the
expected commit. Re-verified at the end of the task: unchanged.

## D. Version-C current state

Linear history from the Version-B base:

```
50514ce  feat(c3): freeze and verify the immutable C3 generation BANK_LOCK
c2c84ae  fix(c2c): align recipe route validation with synthesis contract
969639c  feat(c2b): validate 32-recipe Gemini batch generation
a7f56be  feat(c2): complete disposable Gemini recipe pilot
bf73fbb  chore(c2): fix the Version-C environment; C2 BLOCKED_NO_API_KEY
cdf1594  feat(c1): add strict LLM recipe provider contract
955a834  docs(c0): record the C0 commit, push and Version-B re-verification
ee7975a  chore(c0): initialize PRISM-FAS-C specification baseline
7799f7d  (tag: m10-blind-evaluation-checkpoint) ← Version-B base
```

All expected historical commits verified present. Every expectation given in the task
brief matched the repository.

## E. Historical milestone audit

Classified from Git, artifacts, locks and reruns — not from conversation.

| milestone | branch | accepted commit | tests rerun | classification |
|---|---|---|---|---|
| C0 | `c0-spec-reconciliation` | `955a8344182b4eaa7af70b80ee2783a792bc5d41` | 32 passed | ACCEPTED_WITH_DOCUMENTED_DEVIATION |
| C1 | `c1-llm-provider-contract` | `cdf1594ca6892ca05c168fb2f6b4494236981222` | 138 passed | ACCEPTED_WITH_DOCUMENTED_DEVIATION |
| C2 | `c2-llm-pilot` | `a7f56be7109f149be18bbd4c4907edf4b04b17f8` | 43 passed | ACCEPTED |
| C2B | `c2b-batch-shape-validation` | `969639cc1a72690ae276afdb6e42487721b04c04` | 41 passed | ACCEPTED_WITH_DOCUMENTED_DEVIATION |
| C2C | `c2c-route-contract-freeze` | `c2c84aecc7e6fce84a18f5dc3ab32d531feed2c5` | 54 passed | ACCEPTED |
| C3-prep | `c3-generation-bank-lock` | `50514cef4b1105bfaf6b3a97c44d1e25588c6448` | 29 passed | **PARTIAL** |
| C4–C13 | — | — | — | MISSING |

All expected commit candidates in the brief verified exactly, including the full hashes
behind the `969639c` prefix.

**C0** froze against spec v1.1. Its scientific content remains compatible; the v1.5
Appendix L/M execution and persistent-context requirements are new scope C0 never
evaluated. Its recorded limitation (7 inherited failures / 101 skips) still holds.

**C1** carries the naming error corrected at C2C: `7afc3abd…` was recorded under a name
reading as a single-recipe schema identity but is the 32-object **envelope** identity.
Its provider implementation was later corrected twice against the live wire.

**C2** satisfies §24 C2 hard acceptance: no final bank, invalid/duplicate/retry/quota
statistics documented, model and prompt frozen before C3. The singleton shape (1 recipe
per request) is not the C3 batch shape; the coverage collapse it showed was attributed to
batch size at C2B, not to the prompt.

**C2B** remains **BATCH_SHAPE_FAIL** and is preserved unchanged. Two findings stand: the
C1-recorded bounded envelope is rejected by the provider (`400 INVALID_ARGUMENT`), and 10
of 32 accepted recipes could not compile because they omitted the physics route.

**C2C** is EXACT against §7.3.1: scientific `generator_route` is exactly
`["physics","gpat"]`, physics-only and gpat-only both rejected, no GPAT-only class, no
silent repair. Its live batch returned 32/32 accepted and compiled with zero compiler
failures.

**C3-preparation is PARTIAL** — see §H.

## F. C3 scientific-request audit

| phase | archived attempts | semantic responses | scientific? |
|---|---|---|---|
| C1 | 0 | 0 | no — contract tests only |
| C2 smoke | 2 | 2 | no — disposable, `C2_SMOKE_ONLY` |
| C2 pilot | 42 | 32 | no — disposable pilot |
| C2B | 1 (+1 documented) | 1 | no — batch-shape diagnostic |
| C2C | 1 | 1 | no — route-contract validation |
| **C3 scientific** | **0** | **0** | — |

```
historical_live_provider_calls_before_C3 = 47   (46 archived + 1 documented-but-overwritten)
c3_scientific_logical_requests            = 0
c3_scientific_candidate_slots             = 0
```

`reports/c3/` contains only `C3_BANK_LOCK.json`, `C3_BANK_LOCK_VERIFICATION.json` and
`.gitkeep`. No raw candidate archive, no selection audit, no recipe bank lock, no
`C3_ACCEPTANCE`. No file anywhere references a C3 batch id or a C3 scientific phase.

**Verdict: no C3 scientific generation has occurred.** The expected count of 0 holds, so
the stop condition in §8 of the brief is not triggered.

## G. v1.5 reconciliation

### Scientific core — highlights

EXACT: target label firewall (§5.4), Version-B immutability (§4), LLM offline-planner role
(§7.1), Gemini provider contract (§7.2), scientific route (§7.3.1), batch envelope
(§7.6.1).

PARTIAL: datasets/protocols (§5.3 — protocol manifests and the leakage=0 proof are not
materialized in Version C); generation/selection/bank identity hierarchy (§7.8.4, §21.3).

MISSING: 384 raw candidates per arm, ≥320 unique valid pool, 256 final recipes,
`prism_c3_selection_v1`, the §7.8.2 coverage quota table, RND/DET schedules (§7.8.5), arm
fairness (§8.2), neutral GPAT (§8.3), 2048/1024 renders, q-once (§11.2), Track-G/Track-R
decision identities (§13.4), PromptHead exclusion, manifold-OFF policy, source-only
envelope (§15.2.2), video aggregation (§20.2), C-H4 rule (§3.1.1), statistics (§20.3).

### v1.5 execution closure

EXACT: `CLAUDE.md`, `PROJECT_STATE.md`, both required skills — all created by this task.

PARTIAL: resume/idempotency (real identity-aware resume exists in the C2/C2B/C2C runners
and `write_lock_once`, but is not centralized); stage-level acceptance (acceptance JSONs
exist but predate L.9 and lack profile/eligibility fields); all-output preservation (the
principle is honoured — C2 preserves every attempt, C2B's negative result is preserved —
but there is no systematic run tree).

MISSING: validate/smoke/full profiles, dual `engineering_status`/`scientific_status`,
`train.py`, method/config/seed run isolation, source-search phase, deterministic source
selector, selected-config lock, `MASTER_RUN_INDEX.json`, `PIPELINE_STATE.json`, recovery
ladder, scientific-eligibility flags.

DEVIATED: spec location (`docs/` vs `docs/specs/`).

## H. C3-preparation audit — the blocking finding

The preliminary lock at `reports/c3/C3_BANK_LOCK.json` is internally sound and verifies
against live code:

```
bank_lock_identity               7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba
C3_GENERATION_CONTRACT_IDENTITY  884bce03b4f40a4ffbbef30f14c2216a6166a0ee1e8a6f6facb163f8bb3cdd85
```

Both match the historical candidates in the brief. All 18 components re-derive from the
repository, and the 29 lock tests pass.

**It is incomplete under v1.5.** §7.8.4 requires a BANK_LOCK to bind

```
C3_BANK_CONTRACT_IDENTITY = SHA256(canonical_json({generation_contract_identity,
                                                   selection_contract_identity}))
```

The lock binds **no** `C3_SELECTION_CONTRACT_IDENTITY` — the word "selection" appears only
as prose in the request schedule. Four things are absent from the repository:

1. `prism_c3_selection_v1` (§7.8.3) — the five-stage lexicographic selection with exact
   integer arithmetic and canonical SHA-256 tie-break.
2. The §7.8.2 frozen coverage table — medium 32/80, geometry 24/64, illumination 24/64,
   artifact 8/128 (preferred 32), region 8/128 (preferred 24).
3. Frozen RND and DET 384-slot schedules (§7.8.5).
4. `C3_SELECTION_CONTRACT_IDENTITY` and `C3_BANK_CONTRACT_IDENTITY`.

**Status: `SUPERSESSION_REQUIRED_BEFORE_C3_SCIENTIFIC_GENERATION`.**

§7.8.4 permits exactly this: *"A preliminary immutable lock that omitted a scientifically
required selection identity may be superseded only before the first affected scientific
request, with an explicit supersedes identity, reason, and
`scientific_requests_before_supersession=0`; the old lock bytes remain immutable
historical evidence."* Since `c3_scientific_logical_requests = 0`, supersession is
permitted now. **No superseding lock was created in this task**, and the preliminary lock
bytes are untouched.

This also resolves the open question raised at the end of the C3 lock session: the
384→256 selection algorithm is not unspecified after all — v1.5 §7.8.3 defines it
completely.

## I. CLAUDE.md

Created at the repository root; none existed. Concise (~95 lines), invariants only:
authority order, Version-B immutability, target firewall, C3 prohibition, smoke≠scientific,
no silent science changes, secret policy, Git safety, test policy, resume semantics,
canonical commands, pointers, STOP conditions. The spec is referenced by path and hash,
never pasted.

## J. PROJECT_STATE.md

Created at `docs/PROJECT_STATE.md`, populated only from evidence verified in this task,
following the M.4 schema. Records branch/HEAD, Version-B integrity, per-milestone
classifications, all contract identities, the C3 blocker, provider-call counts, exact test
counts, known deviations and exactly one `next_authorized_action`.

## K. Project skills

`.claude/skills/prism-milestone/SKILL.md` — execute one authorized milestone/substage:
orient → confirm authorization and profile → execute → verify → record and stop. Defines
no scientific constant.

`.claude/skills/prism-handoff/SKILL.md` — session handoff from evidence, never memory;
explicitly forbidden from making scientific decisions, calling providers, running GPU
work, opening target labels or changing configs/locks.

Parent-scope audit: `D:\AI on IOT\Anti_spoofing\.claude` contains only permission
allowlists (`settings.json`, `settings.local.json`). No parent `CLAUDE.md` exists. Nothing
there conflicts with project science. Left unmodified.

## L. Pipeline restructure plan

`docs/V15_PIPELINE_RESTRUCTURE_PLAN.md` — target layout, per-file disposition
(KEEP / MOVE_LATER / WRAP / REFACTOR_LATER / HISTORICAL_ONLY), execution semantics to
preserve, and a six-step migration order. Nothing was moved. It flags one hazard: several
identities are computed over config file bytes, so any future move must re-derive every
affected identity and stop if one changes.

## M. Tests

```
python -m pytest -q --no-header -p no:cacheprovider --continue-on-collection-errors
→ 7 failed, 1310 passed, 101 skipped, 1 warning in 421.34s
```

Milestone suites: C0 32 · C1 138 · C2 43 · C2B 41 · C2C 54 · C3 29 — all passed (337).

The 7 failures are exactly the inherited set documented in `reports/c0/C0_TEST_SUITE.json`
(M2 validation ×2, M8 pair-plan ×2, M10 closure ×2, M10 target-evaluation ×1) — each
asserts against a frozen Version-B evidence file absent from a fresh Version-C clone.
**0 new unexplained failures.** No scientific code was modified to turn them green.

Capability check before running: every milestone conftest blocks `socket.connect`,
`connect_ex` and `create_connection`, and deletes `GEMINI_API_KEY`/`GOOGLE_API_KEY` (16
guard sites). Suites use mock/replay providers. No live provider call, Modal job, GPU
workload or SiW label access occurred.

## N. Blockers

1. **C3 scientific generation is blocked.** `prism_c3_selection_v1`, the §7.8.2 quota
   table, the §7.8.5 RND/DET schedules, `C3_SELECTION_CONTRACT_IDENTITY` and
   `C3_BANK_CONTRACT_IDENTITY` are all absent, so no lock can satisfy §7.8.4.
2. **The v1.5 execution layer does not exist** — no `train.py`, no execution profiles, no
   dual status, no `state/` files, no run tree.
3. **Protocol manifests (§5.3) are not materialized** in Version C, and the leakage=0
   proof required before any scientific training has not been produced.

None of these is a defect in previously accepted work; all are v1.5 scope not yet built.

## O. Next authorized action

> Implement the v1.5 C3 selection contract — `prism_c3_selection_v1` per §7.8.3 with the
> §7.8.2 quota table and the §7.8.5 RND/DET schedules — then compute
> `C3_SELECTION_CONTRACT_IDENTITY` and `C3_BANK_CONTRACT_IDENTITY` and prepare a
> superseding C3 BANK_LOCK for user approval, preserving the preliminary lock bytes
> unchanged. No live C3 generation.
