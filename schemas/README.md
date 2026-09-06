# Request trace format

The canonical workload format for this project. One UTF-8 JSONL file per trace:
the first nonblank record is a `trace_meta` header, every later record is a
request. `request-trace-v1.schema.json` is the normative definition;
`cache_delay_eval.trace` implements the same rules plus the ordering and
lineage checks that JSON Schema cannot express.

Validate any trace with:

```
cache-delay-validate-trace traces/example.jsonl
```

## Why arrival times are offsets

`arrival_time_ms` is an integer offset from the start of the trace, not a
wall-clock timestamp. Replay is then deterministic and there is no
floating-point ambiguity about the order of two nearby requests.

## The three prompt forms

Every request carries **exactly one** of `prompt`, `token_ids`, or
`block_hashes`. Each corresponds to a different kind of source workload.

| Form | Source | Notes |
| --- | --- | --- |
| `prompt` | ShareGPT, WildChat | Portable; needs tokenizing before block-level replay |
| `token_ids` | Pre-tokenized corpora | Exact block-level replay; the header's `model_id`/`tokenizer_id` identify what produced them |
| `block_hashes` | Mooncake | Per-block fingerprints, no text at all |

The third form exists because production traces cannot publish user prompts.
Mooncake instead releases one hash identifier per block, which is all a
prefix-cache study needs: requests sharing a prefix share their *leading* hash
identifiers. Such a record has no text and no token ids, so a schema that
demanded one of those two could not represent it.

A hash-only record must also carry `prompt_tokens`. The TTFT model takes prompt
length as an input, and with no text there is nothing to derive it from.
Multiplying block count by `block_size` is not a substitute, because the final
block of a prompt is usually partial.

`block_hashes` accepts integers or strings, so both a trace's own identifiers
and hashes recomputed locally from text are representable. The header's
optional `block_size` records the granularity they were computed at.

## Identity fields

Four optional fields describe how requests relate to each other. Keeping them
distinct is what makes the reuse breakdown possible.

- `session_id` — groups turns **within one conversation**. Consecutive turns
  share a long exact prefix, which is the dominant source of reuse in chat
  workloads.
- `user_id` — groups conversations belonging to the **same client** across
  sessions, e.g. from WildChat's hashed client identifier. Reuse across two
  sessions with one `user_id` is the inter-session case, the regime where a
  global cache index should beat routing by session affinity.
- `parent_request_id` — explicit lineage; must refer to an earlier request in
  the same session.
- `prefix_group` — ground-truth annotation for synthetic traces only. **A
  routing policy must never read it**; it exists to score decisions after the
  fact.

The reader rejects a `session_id` seen under two different `user_id` values,
which catches conversion bugs when deriving identity from a source trace.

## Version history

- **1.1** — added the `block_hashes` prompt form, `prompt_tokens`, `user_id`,
  and the header's `block_size`.
- **1.0** — `prompt` or `token_ids` only.

Readers accept both versions; writers emit 1.1.
