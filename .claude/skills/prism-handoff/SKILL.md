---
name: prism-handoff
description: Produce or update a PRISM-FAS-C session handoff so the next session can resume without the previous transcript. Use when wrapping up a session, when context is running low, when the user asks for a handoff, status, or "where are we", or when recovering context at the start of a fresh session. Read-only with respect to science.
---

# PRISM-FAS-C session handoff

Purpose: make the project recoverable from the repository alone. A future session must
be able to continue correctly without reading this conversation.

This skill **observes and records**. It does not advance the science.

## Gather from evidence, not from memory

Collect each item from the repository, Git or artifacts — never from recollection:

**Repository**
- branch, HEAD, dirty/clean, files changed in this session
- untracked scientific artifacts, if any

**Position**
- milestone and substage
- execution profile (`validate` / `smoke` / `full`) and pipeline phase
- `engineering_status` and `scientific_status` per touched milestone

**Work performed**
- exact commands executed
- exact test counts: passed / failed / skipped / xfailed
- live provider calls made this session, by phase, with the archive path
- Modal / SSH / GPU jobs launched (expected: 0 unless explicitly authorized)

**Identities**
- contract identities: generation, selection, bank contract
- locks written or verified, with their paths and identities
- any identity that changed, with old and new values

**Source search**, when active
- search plan identity
- config ids completed, config ids remaining
- current deterministic winner, or `N/A`
- selected-config lock, or `N/A`

**Integrity**
- Version-B HEAD, tag, clean — verified read-only
- target firewall state: any SiW label access (expected: none)

**Open items**
- known failures, including the documented inherited set
- blockers, each with its exact cause
- unfinished runs and where they resume

## Produce

Update `docs/PROJECT_STATE.md` with proven facts only, and give the user a concise
summary containing:

- the exact resume command
- exactly **one** `next_authorized_action`

`next_authorized_action` is a single concrete action. Not a plan, not a list, not a
speculative sequence.

## Never

This skill must not:

- make a scientific decision, or recommend one as though it were settled
- call Gemini or any live provider
- run GPU, Modal or SSH work
- open SiW labels or any target metric
- change a config, a lock, a frozen constant or a scientific artifact
- mark incomplete science as complete, or promote smoke evidence to scientific evidence
- record an unverified value — if something could not be verified, write `UNKNOWN` and
  say why

If gathering the handoff reveals a contradiction between `PROJECT_STATE.md` and the
repository, report the contradiction and trust the repository.
