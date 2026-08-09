# C0 — The LLM gap between Version B and Version C

This document establishes, **from repository evidence rather than from memory**, that the
frozen Version-B scientific recipe bank was produced by a deterministic offline generator
and that no external LLM was invoked. It then states exactly what Version C changes.

This is the single fact that justifies Version C existing at all. If Version B had already
used a real LLM, the Version-C treatment arm would not be a new factor.

## 1. What Version B actually did

### 1.1 Primary evidence — the frozen bank lock

`assets/recipe_banks/prism_recipe_bank_m7_v1/BANK_LOCK.json`, the lock of the bank that
entered the frozen Version-B science:

```json
"generator": {
  "provider": "deterministic_local",
  "model_id": "deterministic-source-only-recipe-generator",
  "revision": "m7-v1",
  "external_llm_invoked": false
}
```

with `"status": "frozen"`, `"recipe_count": 128`, and bank content identity
`fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb`.

That same identity is the `m7_recipe_bank_identity` recorded in
`reports/m10/SOURCE_MATRIX_LOCK.json → frozen_inputs`, and it is re-checked in the
M9↔M10 reference binding. So the bank named in the lock is the bank the science used —
not a sibling artifact.

### 1.2 The generator implementation

`src/prism_fas/recipes/generate.py`:

```python
# The M7 bank is produced by an offline deterministic structured generator.
# No external API, credential or hosted model is involved; `prompt.txt` is the
# constrained contract a future frozen LLM provider would be given instead.
GENERATOR_PROVIDER = "deterministic_local"
GENERATOR_MODEL_ID = "deterministic-source-only-recipe-generator"
GENERATOR_REVISION = "m7-v1"
GENERATOR_EXTERNAL_LLM_INVOKED = False
```

The module imports `numpy`, the local ontology, the local schema and the local compiler.
There is no HTTP client, no SDK and no credential path anywhere in it. The recipes are
produced by deterministic rotation over the ontology vocabularies and a seeded
`numpy` generator (`derive_seed` / `local_rng`), with severity allocated inside the
ontology's per-artifact safe bands under a total-severity budget.

### 1.3 The build configuration

`configs/recipes/bank_m7.yaml`:

```yaml
# The bank is generated offline by the deterministic structured generator in
# `prism_fas.recipes.generate`. No API key, hosted model or network access is
# involved; `prompt.txt` is frozen alongside the bank as the contract a future
# pinned LLM provider would receive.
generator:
  provider: "deterministic_local"
  model_id: "deterministic-source-only-recipe-generator"
  revision: "m7-v1"
  external_llm_invoked: false
  network_access: false

diversity:
  method: "offline_tfidf_cosine_v1"
  external_text_model: false
```

Note the second flag: even the *diversity* filter avoided an external text model, using
offline TF-IDF cosine instead. Version B had no external model in the recipe path at all.

### 1.4 The invariant was enforced, not merely recorded

`src/prism_fas/recipes/bank.py` refuses a lock that claims otherwise:

```python
if lock.get("generator", {}).get("external_llm_invoked") is not False:
    errors.append("generator claims an external LLM was invoked")
```

and `tests/test_m7_recipe_physics.py` asserts it:

```python
assert generator["model_id"] == "deterministic-source-only-recipe-generator"
assert generator["external_llm_invoked"] is False and GENERATOR_EXTERNAL_LLM_INVOKED is False
```

So `external_llm_invoked: false` is a validated structural property of the Version-B bank,
not a stale annotation someone forgot to update.

### 1.5 Corroborating project records

- `PROJECT_STATUS.md`: *"Generator | `deterministic_local` / `deterministic-source-only-recipe-generator` / `m7-v1`; **no external LLM, no network, no credential**"*.
- `README.md` (M7 section): the frozen bank *"was produced by an **offline deterministic generator** — no external LLM, network call or credential is involved, and `prompt.txt` is frozen alongside it as the contract a future pinned provider would receive."*
- `reports/paper/PRISM_FAS_B_FULL_PROJECT_NARRATIVE.html` records the same `external_llm_invoked=false` closure state, and the Version-C spec cites it (§Appendix D).
- `src/prism_fas/synthesis/random_operator_bank.py` — the A02 random-operator control — also carries `"external_llm_invoked": False`. Both Version-B recipe arms were non-LLM.

### 1.6 What `prompt.txt` is, and is not

`assets/recipe_banks/prism_recipe_bank_m7_v1/prompt.txt` exists and is hashed into the
bank lock (`prompt_sha256 = 6181410d…7af02`). It is **not evidence of an LLM call.** Both
the code comment and the config describe it as the constrained contract that a future
pinned provider *would* receive. It was frozen so that a later LLM milestone would have a
fixed, auditable prompt. Version C is that milestone.

### 1.7 Conclusion

> **Established.** The Version-B scientific recipe bank
> (`prism_recipe_bank_m7_v1`, 128 recipes, identity `fa989938…10cb`) was produced by
> `deterministic-source-only-recipe-generator`, offline, with no network access, no
> credential and no external LLM. The claim "Version B used an LLM" would be false. The
> original Version-B design *intended* an LLM (§2 below); the frozen implementation did
> not deliver one.

An important consequence for the Version-B result: hypothesis H4 there — *structured
recipe bank beats random augmentation* — compared a **deterministic** structured bank
against a random bank. It was not, and was never described as, an LLM test.

## 2. What the original Version-B design intended

The original design document
(`docs/spec/Spec_PRISM_FAS_Version_B_Detailed_2026.docx`, SHA256 `74c7557f…3292`)
specified an LLM recipe engine explicitly:

- §1.1 model table: *"Recipe LLM | Qwen2.5-7B-Instruct hoặc Qwen3-8B-Instruct | Offline / frozen | Sinh JSON recipe, rule-aware, tái lập bằng seed | API LLM với schema enforcement"*.
- §6.2: *"LLM nhận system prompt chứa schema, allowed values, constraint và ví dụ hợp lệ; không nhận target taxonomy. Sinh N candidate recipes dưới JSON mode hoặc constrained decoding."*
- §4.2 H3: *"LLM-compositional recipe bank tạo diversity có ích hơn random augmentation cùng số lượng sample."*
- Appendix B model registry pins `Qwen/Qwen2.5-7B-Instruct` as the recipe LLM.

The v1.1 implementation spec
(`docs/spec/PRISM_FAS_B_v1_1_..._Codex_Spec.docx`, SHA256 `44634d7c…cfeac`) kept the role
statement — *"LLM chỉ sinh structured attack recipe và prior text"* (§Trạng thái tài liệu),
*"Recipe engine | Ontology source-only + frozen LLM"* (§1.1) — but never specified a
provider, credential handling, prompt-freeze procedure or generation budget. The
implementation resolved that gap by shipping the deterministic generator and freezing the
prompt for later. That is the deviation Version C closes.

## 3. What Version C changes

### 3.1 Version B (frozen, historical)

```
deterministic offline generator            (deterministic-source-only-recipe-generator, m7-v1)
        |
        v
structured recipe (schema v1.1, 128 recipes)
        |
        v
recipe compiler (m7-compiler-v1)  -> operator graph + mask policy + 41-D conditioning
        |
        +--------------------+
        |                    |
        v                    v
     Physics              GPAT
```

### 3.2 Version C

```
Gemini structured recipe planner           (gemini-3.6-flash, thinking_level=medium,
        |                                   TEXT-ONLY, strict JSON schema, tools OFF)
        v
strict JSON / ontology validator           (schema + ontology + compatibility + severity;
        |                                   fail-closed, max 2 machine retries)
        v
structured executable recipe               (schema v1.1, 384 raw slots -> 256 final)
        |
        v
SAME compiler boundary                     (inherited compiler contract)
        |
        +--------------------+
        |                    |
        v                    v
     Physics              GPAT-C           (generator-neutral, trained once, frozen,
                                            shared by RND / DET / LLM)
```

The compiler boundary is deliberately unchanged. Everything downstream of the validated
recipe — compiler, conditioning vector, physics operators, GPAT residual mechanism,
quality gate, q semantics — is inherited. That is what makes the generator the only
changed factor in C-H1 and C-H2.

### 3.3 The controls that make it an experiment

Version C does not simply swap in an LLM. It runs three arms through the same pipeline:

| arm | generator | role |
|---|---|---|
| RND | rule-valid random operators from the same ontology | lower control |
| DET | deterministic structured generator (Version-B lineage) | structured non-LLM control |
| LLM | frozen Gemini planner | treatment |

each with exactly **256** recipes, the same compiler, the same validation, the same base
live-sample list, the same route budget, the same frozen generator-neutral GPAT-C
checkpoint, the same quality gate and the same **1024** accepted synthetic samples
(512 Physics + 512 GPAT). No arm may gain an advantage from extra training exposure.

## 4. What the LLM is not

The Gemini planner in Version C is an **offline semantic recipe planner**. It is NOT:

- the FAS classifier;
- an image generator (it emits no pixels);
- a target-image interpreter (it never receives an image of any kind);
- part of inference;
- part of the training loop.

It runs once, before training, and its output is frozen. After C3, downstream code never
calls a provider again; it reads a frozen bank. Concretely, the LLM MUST NOT receive raw
CASIA/MSU images, raw SiW images, target filenames, SiW taxonomy or family counts, target
failure analysis, target metrics, or any Version-B attack-wise target feedback. Its input
is generic ontology and schema text, which is what makes its contribution a text-only
prior rather than visual target adaptation.

## 5. What Version C may and may not claim about this

Allowed, if the evidence supports it: *LLM recipe planning improves cross-domain FAS*,
conditional on C-H1 and/or C-H2 being supported across 5 seeds on P3 with consistent
P1/P2 cross-source evidence, and stating that SiW was historically known from Version B.

Not allowed: *"first LLM-based FAS method"* without a literature review; *"the LLM
understands spoof physics"* on the basis of valid JSON output; any claim that a
target-time PromptHead measures a real target recipe.

Also not allowed: reporting the Version-B H4 result as an LLM finding. It was a
deterministic-vs-random comparison, and it did not favour the structured arm.
