# C_LLM_RECIPE_CONTRACT — the Version-C LLM recipe planner

Milestone **C1**. Frozen unless a later milestone records a decision to change it.

This document defines how a real LLM may hand a structured recipe to PRISM-FAS-C
without free-form output, target leakage, untraceable generation, silent repair
or secret leakage. It is the contract; the implementation lives in
[`src/prism_fas/llm/`](../../src/prism_fas/llm/) and the evidence in
[`reports/c1/`](../../reports/c1/).

**C1 builds the contract. It does not build a recipe bank.** No scientific bank,
no 32-recipe pilot, no 384 candidate slots, no selection of 256 recipes.

---

## 1. Role of the LLM

The LLM is an **offline semantic recipe planner**. It receives a frozen generic
FAS ontology, the strict recipe schema, the compatibility constraints and batch
coverage quotas, and it returns structured JSON candidates.

It is **not**:

- the FAS classifier;
- an image generator — it emits no pixels;
- a target-image interpreter — it never receives an image of any kind;
- part of inference;
- part of the training loop.

It runs once, before training. After C3 freezes the bank, no downstream code
calls a provider again: training reads a frozen artifact. This is what makes the
LLM's contribution a **text-only prior** rather than visual target adaptation.

## 2. Allowed inputs

Only these may reach the provider:

- generic FAS task context (what an artifact recipe is, in physical terms);
- the frozen ontology: media, geometry shapes, semantic regions, artifacts,
  illumination values, generator routes, forbidden-shortcut policies;
- numeric ranges and the severity budget owned by the ontology;
- the medium→artifact and geometry→region compatibility tables;
- the response JSON Schema;
- per-batch counts and ontology-level coverage quotas;
- the standing physical-plausibility and diversity objectives.

## 3. Forbidden inputs

Refused **fail-closed** by [`firewall.py`](../../src/prism_fas/llm/firewall.py),
before a request leaves the process, in every provider including the test doubles:

| Category | Examples |
|---|---|
| dataset reference | SiW-Mv2, CASIA-FASD, MSU-MFSD, CelebA-Spoof, OULU-NPU, WMCA |
| target metric | ACER, APCER, BPCER, HTER, ROC-AUC, EER, attack-wise anything |
| target structure | `target_test`, labels, ground truth, attack family, subject id, prediction locks |
| target feedback | "beat the benchmark", "attacks Version B failed", "optimize target ACER", "the test set contains…" |
| binary payload | any `data:image/…;base64`, any `.png`/`.mp4`/`.parquet`/`.pt` reference |
| forbidden payload key | `image`, `inline_data`, `parts`, `file_data`, `labels`, `ground_truth`, `predictions`, … |

No face image is ever submitted. No CASIA/MSU/SiW private content is ever
submitted. No Version-B attack-wise target feedback is ever submitted.

The prompt builder runs the firewall over **its own output**, so a caller-supplied
coverage quota cannot smuggle target content in.

## 4. Provider and model contract

Verified at C1 against the official documentation **and** the installed SDK; the
evidence is in [`C1_PROVIDER_AUDIT.json`](../../reports/c1/C1_PROVIDER_AUDIT.json).

| Field | Frozen value |
|---|---|
| provider | Google Gemini Developer API |
| model | `gemini-3.6-flash` (documented **stable**) |
| SDK package | `google-genai` |
| SDK version (installed) | `2.17.0`; pinned `>=2.3.0,<3` |
| Python | 3.13.11 |
| API surface | `client.interactions.create(...)` |
| thinking | `generation_config={"thinking_level": "medium"}` |
| structured output | `response_format={"type": "text", "mime_type": "application/json", "schema": <JSON Schema>}` |

**Why the Interactions API.** The spec left the surface open
(`api_surface: generate_content_or_interactions_frozen_at_C1`). The Interactions
API is GA and is the surface Google recommends for current models, so C1 freezes
`interactions`.

**Sampling controls.** `temperature`, `top_p` and `top_k` are never sent. On this
surface they are not merely discouraged: `GenerationConfigParam` in the installed
SDK has no such fields at all.

> **Recorded discrepancy.** Spec Appendix E says current Gemini 3.x guidance
> *deprecates* these controls. The Gemini 3 developer guide instead says: *"For
> all Gemini 3 models, we strongly recommend keeping the temperature parameter at
> its default value of 1.0"*, and does not discuss `top_p`/`top_k`. The required
> **behaviour** is identical under either wording — do not send them — so no spec
> amendment is needed. It is recorded because the spec's stated rationale is
> narrower than what the documentation currently says.

**Disabled capabilities.** `tools` is never passed, which disables function
calling, Search grounding, URL context, code execution and file search together.
`store` is `False`, so no server-side conversation state is retained. `input` is a
plain `str`; there is no code path that attaches media.

**Provider abstraction.** Everything depends on
[`RecipeProvider`](../../src/prism_fas/llm/providers/base.py), never on a vendor
SDK. Three implementations ship: `MockRecipeProvider`, `ReplayRecipeProvider`,
`GeminiRecipeProvider`. The Google import is lazy, so the whole contract is
testable with no SDK and no credential installed.

## 5. Strict schema

The recipe schema is the **inherited Version-B schema v1.1**, reused unchanged for
comparability (spec §7.3). Any extension requires a version bump and a migration.

Rejection is total — nothing is ignored, nothing is stripped:

| Condition | Result |
|---|---|
| malformed JSON | REJECT |
| prose or a markdown fence around the JSON | REJECT — never scraped |
| missing required field | REJECT |
| unknown field | REJECT (`extra="forbid"`) |
| unknown enum | REJECT |
| out-of-range value | REJECT |
| wrong recipe count in the batch | REJECT |
| model-supplied system-owned field | REJECT (see §12) |

Two fields are deliberately **absent** from what the model is asked for:
`recipe_id` (assigned by the system) and per-artifact `parameters` (operator
internals belong to the compiler and the ontology).

## 6. Ontology boundary

**The ontology and the compiler are the execution authority. The LLM cannot
extend either.** The model may propose only values that already exist in the
frozen ontology; a value outside it is a rejected candidate, not a new ontology
entry. The request-side JSON Schema is generated *from* the ontology, so the
enums the model sees can never drift from the validator that judges it.

The schema is a request-side constraint, not the authority. A provider that
ignores it produces a rejected candidate. Validation is local and unconditional.

## 7. Validation stages

```
raw provider response
    -> json_parsing
    -> envelope_schema          exactly {"recipes": [...]} with the requested count
    -> typed_recipe_schema      pydantic, extra="forbid"
    -> ontology_membership      values enabled by this ontology version
    -> range_checks             medium/geometry/capture bands, operator-safe strengths
    -> compatibility_checks     medium->artifact, geometry->region, severity budget
    -> canonicalization
    -> recipe_identity          SHA-256 over the canonical text
    -> duplicate_detection
    -> ACCEPT / REJECT
```

The inherited validator contributes its own stages (`schema`, `canonical`,
`medium_artifact`, `geometry_region`, `strength_range`, `severity`, `duplicate`,
`route`, `shortcut`, `leakage`, `serialization`, `hash`) and reports **every**
issue rather than stopping at the first.

## 8. Canonicalization

- deterministic key order: sorted;
- deterministic float representation: rounded to 6 decimals;
- compact separators, `ensure_ascii=False`, LF only;
- regions carry a canonical order enforced by the schema, so list order is
  meaningful and is not re-sorted silently;
- identity: SHA-256 over the canonical UTF-8 text.

Response key order and equivalent float spellings collapse to one identity. A
real numeric difference does **not** — canonicalization never removes a
meaningful distinction.

## 9. Duplicate rule

Duplicates are detected on the **canonical scientific content identity**, with the
positional `recipe_id` excluded so renumbering cannot defeat detection. A
duplicate is REJECTED as a candidate and recorded in the audit trail. It applies
within a single response and across slots.

Fuzzy or embedding-based similarity is **not** used at C1: the spec does not
require it here, and coverage/diversity selection belongs to C3.

## 10. Compiler compatibility

Every accepted candidate must reach the inherited compiler
(`compile_recipe`, `m7-compiler-v1`) and produce:

- a validated operator graph,
- a region mask policy,
- a fixed **41-D** conditioning vector,
- a deterministic graph hash.

C1 proves this in tests rather than assuming it. The contract must not accept a
recipe that validates at the LLM layer but fails at the compiler layer, because
C3 would then freeze a bank that breaks during synthesis.

## 11. Prompt contract

The system instruction carries: role, the no-target scope rule, the schema, the
ontology, the compatibility tables, the hard output format, maximum artifacts and
regions, the forbidden-shortcut policy, and an explicit **JSON only** requirement.

The generation template carries **only** the batch size and ontology-level
coverage quotas. It must never carry corpus-derived deficits, target statistics,
target failures or any feedback from an evaluation.

Humans do not hand-pick outputs. Selection is machine-validated and deterministic.

> This is the **C1 template**, not the C3 prompt lock. C2 may still tune wording
> using source-only validity and coverage evidence. The exact UTF-8 bytes and
> their SHA-256 are frozen in the LLM bank lock **before** the 384-slot
> generation begins (spec §7.5).

## 12. No silent repair

**Mandatory.** If the provider returns invalid JSON, an unknown field, an invalid
enum, an incompatible medium/artifact pair, an out-of-range strength or a
duplicate, the code does **not** edit the output into something valid.

Every attempt is recorded. The only permitted recovery is asking the provider for
another candidate under the **same frozen contract**, bounded by the retry budget.
Canonical formatting is applied only *after* a candidate is already semantically
valid.

**`recipe_id` assignment is not a repair.** The spec requires the system to
compute the recipe id, the content hash, the provenance and the validation status
rather than trust the model (§7.3). The model is *forbidden* to supply them and is
**rejected** — not silently stripped — if it does. The id is then assigned
positionally and deterministically. That adds provenance the model was not allowed
to provide; it changes no semantic field.

Ontology **alias normalization** is available in the inherited validator and is
switched **OFF** in the C1 configuration — the strictest reading of §7.6's
"unknown enum → REJECT". The flag is identity-bearing, so enabling it would change
the recipe bank identity and may only happen as a source-only decision recorded
before C3.

## 13. Provenance

All 21 fields required by spec §7.7 are present in every record. A record missing
one raises rather than serializing a hole, and a record carrying a
credential-shaped key raises too.

Raw provider response text is preserved **verbatim** so it can be archived and
replayed. `raw_response_sha256`, `parsed_recipe_sha256`, the request identity, the
model id, the SDK version, the thinking configuration, the retry count and the
billing tier are all recorded.

**The API key is never a field and cannot become one.** Every string in a record
passes through the redaction pass first.

## 14. Secret policy

- The key is read from the environment variable named by `api_key_env`
  (default `GEMINI_API_KEY`), never from a config file, never from source.
- It is passed **explicitly** to the client, because the SDK also honours
  `GOOGLE_API_KEY` and lets it take precedence — explicit passing prevents a
  different project's credential being used silently.
- Never hard-coded, printed, logged, serialized to JSON, written to a report,
  included in an exception message, committed, or sent in a prompt.
- `.env` is git-ignored. `.env.example` carries the variable **name** only.
- Reports state `api_key_present: true|false` and nothing more.

To supply it locally, in PowerShell:

```powershell
$env:GEMINI_API_KEY="<your key>"
```

Do not commit it, and do not paste it into a chat transcript.

## 15. Retry classification

Bounded; no infinite loops. Semantic retries and transport attempts are counted
**separately**, so a flaky connection cannot consume the semantic budget.

| Retryable | Non-retryable |
|---|---|
| `transport` | `auth` |
| `server_error` | `model_unavailable` |
| `rate_limit` (429 rate_limit_exceeded) | `unsupported_config` |
| `invalid_candidate` | `quota_exhausted` (429 quota_exceeded) |
| | `forbidden_request` (firewall refusal) |
| | `contract_violation` |
| | `local_error` |

Semantic maximum: **2 retries per request** (§7.6). Transport attempts: 4, with
exponential backoff capped at 60 s. The two partitions are disjoint and exhaustive,
and a test asserts it. A retry may include only machine-generated validation
errors, never human aesthetic feedback.

## 16. Quota behaviour

```
429 rate_limit_exceeded   -> bounded exponential backoff -> retry the SAME frozen request
429 quota_exceeded        -> preserve every completed slot
                          -> write API_QUOTA_BLOCKED.json
                          -> STOP cleanly
                          -> tell the user
```

The code **never** enables billing, opens a billing page, attaches a billing
account, prepays, switches API project, switches provider, switches model, changes
the prompt, or regenerates a slot that already succeeded.

After a quota stop the user chooses: wait for the reset, or explicitly enable a
Paid Tier. On resume, only the **missing** slot ids are generated, under an
unchanged model, prompt, schema and selection algorithm. Billing tier is recorded
in provenance and is **not** a treatment factor.

## 17. Raw-response freeze

Once C3 begins, archived raw responses are **immutable provenance**. They are
never rewritten, re-requested or regenerated. The archive is the reproducible
artifact, because a hosted model is not reproducible: asking the same question
twice may return different text.

## 18. Replay semantics

`ReplayRecipeProvider` re-serves archived responses through the identical
parse → validate → canonicalize → identity path, with **no network call and no
client object**. Replaying the same archived response twice produces byte-identical
canonical recipes and identical identities.

Replay refuses to cross identities: if an archived response carries a
`request_sha256` that differs from the current request's, replay raises rather
than mixing two scientific identities. A slot that was never archived is a hard
error — replay never invents a response and never falls back to a live call.

## 19. C2 / C3 transition

| | C1 (this document) | C2 | C3 |
|---|---|---|---|
| purpose | contract + providers + tests | disposable pilot | frozen scientific banks |
| live calls | ≤ 2 optional disposable probes | 32 disposable candidates | exactly 384 raw slots (12 × 32) |
| output | no bank | pilot bank + audit, **never** enters the final bank | 3 × 256 frozen banks + locks |
| prompt | template, tunable | tunable on source-only validity/coverage only | **frozen bytes + SHA-256 in the bank lock** |
| raw responses | fixtures | archived | immutable provenance |

Before C3 may start: the user approves and freezes the model, the prompt and the
schema at the end of C2; the live Free-Tier quota snapshot is recorded; and the
minimum valid unique pool of **320** must be reachable, or C3 fails rather than the
validator being weakened.

## 20. Scientific non-goals

C1 claims nothing about recipe quality, diversity, detector performance,
cross-domain generalization or any LLM benefit. It establishes only that a
structured recipe can be received safely and traceably.

Version C will not claim "the LLM understands spoof physics" on the basis of valid
JSON, will not claim to be the first LLM-based FAS method without a literature
review, and will never describe P3 SiW-Mv2 as a never-before-seen blind target —
it is historically known from Version B and only procedurally isolated in Version C.
