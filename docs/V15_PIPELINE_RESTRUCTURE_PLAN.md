# v1.5 pipeline restructure plan

How the existing repository migrates to the Appendix M.1 layout and the Appendix L
execution semantics. **This is a plan. Nothing is migrated, moved or deleted by it.**
Every historical artifact, lock and report stays exactly where it is until a separate
authorized task performs the move.

## 1. Target layout (Appendix M.1)

```
PRISM_FAS_C_LLM_Project/
├── train.py                          # canonical C0→C13 orchestrator CLI
├── CLAUDE.md
├── .claude/skills/{prism-milestone,prism-handoff}/SKILL.md
├── docs/
│   ├── specs/PRISM_FAS_C_LLM_v1_5_...docx
│   └── PROJECT_STATE.md
├── configs/
│   ├── execution/{validate,smoke,full}.yaml
│   ├── search/  llm/  recipes/  synthesis/  detector/  experiments/  target/
├── src/prism_fas/
│   ├── pipeline/{orchestrator,state,registry,resume,budget}.py
│   │   └── stages/c0.py … c13.py
│   ├── recipes/ synthesis/ detector/ train/ evaluation/ locks/
├── state/{PIPELINE_STATE.json,MASTER_RUN_INDEX.json}
├── runs/{validate,smoke,full}/
├── reports/{validate,smoke,full}/
└── tests/
```

## 2. Disposition of existing files

`KEEP` = stays as-is · `MOVE_LATER` = relocate unchanged · `WRAP` = keep logic, add a
stage adapter · `REFACTOR_LATER` = restructure when the stage is built ·
`HISTORICAL_ONLY` = immutable evidence, never edited or moved.

### Scientific source (`src/prism_fas/`)

| Path | Disposition | Note |
|---|---|---|
| `llm/config.py`, `contracts.py`, `firewall.py`, `provenance.py` | KEEP | frozen C1 contract |
| `llm/json_schema.py`, `prompt.py`, `pipeline.py` | KEEP | carry frozen identities |
| `llm/route_policy.py` | KEEP | §7.3.1, identity-bearing |
| `llm/coverage_quotas.py` | KEEP | §7.8 request-side quotas |
| `llm/bank_lock.py` | REFACTOR_LATER | move to `locks/`; extend for selection + bank identity |
| `llm/providers/{gemini,mock,replay}.py` | KEEP | live/offline provider triad |
| `llm/pilot_audit.py` | WRAP | audit helpers reused by C3 coverage reporting |
| `recipes/*` | KEEP | inherited compiler/ontology/validator — the execution authority |
| `synthesis/*`, `detector/*`, `train/*`, `evaluation/*` | KEEP | inherited Version-B implementations for C4–C12 |
| `data/*`, `cloud/*`, `cli/*` | KEEP | preprocessing, Modal wrappers, existing CLI |

**New modules required** (none exist today):
`recipes/selection.py` (prism_c3_selection_v1), `recipes/arms.py` (RND/DET §7.8.5),
`locks/` package, `pipeline/{orchestrator,state,registry,resume,budget}.py`,
`pipeline/stages/c0.py … c13.py`, `train.py`.

### Milestone scripts (`scripts/`)

| Path | Disposition | Note |
|---|---|---|
| `c0_fingerprint_version_b.py` | WRAP | becomes part of `stages/c0.py` preflight |
| `c1_build_reports.py` | WRAP | `stages/c1.py` |
| `c2_*.py` (7 files) | WRAP | `stages/c2.py`; pilot stays disposable |
| `c2b_*.py`, `c2c_*.py` | WRAP | fold into `stages/c2.py` substages |
| `c3_bank_lock.py` | REFACTOR_LATER | grows into the full C3 stage with selection |
| `v15_reconcile.py` | KEEP | reconciliation/audit utility |
| `m4_worker_smoke.py`, `m6…m9_*.py` | KEEP | inherited Version-B utilities |

The per-milestone scripts keep working during migration. `train.py` calls stage modules;
stage modules call the same functions the scripts call today. No behaviour changes in the
same commit as a move.

### Evidence

| Path | Disposition |
|---|---|
| `reports/c0…c3/**` | HISTORICAL_ONLY — never moved, never edited |
| `reports/v15_reconciliation/**` | KEEP |
| `docs/c0…c3/**`, `docs/M*.md`, `docs/spec/**` | HISTORICAL_ONLY |
| `configs/version_c/llm/*.yaml` | MOVE_LATER → `configs/llm/` (identity is over file *bytes*, not path — verify each identity after any move) |
| `configs/recipes/ontology_m7.yaml` | KEEP — path is referenced by frozen identities |
| `tests/c0…c3/**` | KEEP |
| `tests/test_m*.py`, `tests/unit/`, `tests/integration/` | KEEP — inherited suite, including its 7 documented failures |

> **Move hazard.** Several identities are computed over file bytes read from a specific
> path. Moving a config does not change its identity, but a move that also rewrites line
> endings or trailing whitespace *does*. Any move must be followed by re-deriving every
> affected identity and comparing against the lock. If an identity moves, stop.

## 3. Execution semantics to preserve (Appendix L)

```
python train.py --profile validate
python train.py --profile smoke  --resume
python train.py --profile full   --resume
```

Optional bounded controls, later: `--from Cx --to Cy`, `--phase`, `--backend`. These may
change placement or debugging scope; they may never change scientific data, topology,
loss semantics, search space, seed family, selection rule, target accessibility or
acceptance criteria.

Full-profile phase machine: `preflight → source-search → save every config output →
deterministic source selection → source config freeze → full scientific matrix → source
matrix freeze → P3 label-isolated prediction → scoring/statistics → final acceptance`.

One command must still preserve every milestone, method, config, seed and protocol
output, every failure and block, every checkpoint and every lock. Losing configs are
never deleted. A gate needing a choice outside a preregistered envelope stops with
`NEEDS_SCIENTIFIC_DECISION`.

## 4. Migration order

Each step is separately authorized; none is performed here.

1. **C3 selection contract** — `recipes/selection.py`, `recipes/arms.py`, the §7.8.2
   quota table as config, `C3_SELECTION_CONTRACT_IDENTITY`,
   `C3_BANK_CONTRACT_IDENTITY`, superseding BANK_LOCK for approval. *Blocks C3.*
2. **Execution skeleton** — `configs/execution/*.yaml`, `state/` files,
   `pipeline/{state,registry,resume}.py`, `train.py --profile validate` over existing
   stages. No scientific behaviour change.
3. **Dual status** — add `execution_profile` and `scientific_eligible` to newly written
   artifacts; historical acceptance files are annotated in the registry, never edited.
4. **Stage adapters** — `stages/c0.py … c3.py` wrapping today's scripts; then smoke.
5. **Layout move** — `docs/specs/`, `configs/llm/`, `locks/`, with identity re-derivation
   as an acceptance gate.
6. **C4–C13 stages** — built as their milestones are authorized.

## 5. What this plan deliberately does not do

No file is moved, renamed or deleted. No lock is superseded. No identity is recomputed
into an artifact. No scientific constant is introduced — every threshold, quota, schedule
and rule referenced here comes from the v1.5 spec.
