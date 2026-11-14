# Upstream and adopted work (`local` branch)

Primary upstream: `vllm-project/vllm-metal` (GitHub fork parent). Secondary upstreams tracked: none.

## Adopted from other repos

None. (Commits adopted from other repos would be listed here with `Source: <repo>@<sha>` and
`Co-authored-by` trailers on the commit.)

## Topic branches

| Topic | Upstream PR | Status |
|---|---|---|
| `topic/request-lifecycle` | none yet | not-submitted (found and soak-tested via macports-ports-local#1196; checked 2026-10-16 that upstream `main` @ `468b0c8` has no equivalent: the patch applies unchanged to `main` and to `v0.31.0`) |
| `topic/gloo-loopback` | none yet | not-submitted (upstream #633 "Use loopback for default UniProc rendezvous" sets `VLLM_HOST_IP`, not `GLOO_SOCKET_IFNAME`; `main` only sets it in `tools/*.py`) |
| `topic/gemma4-assistant-probe` | none yet | not-submitted (compat probe only catches ValueError; transformers 5.20 raises StrictDataclassError) |

## Origin

Both topics were developed as `files/` patches of the MacPorts port `python/py-vllm-metal-devel`
(`eejd/macports-ports-local`, patches `patch-runner-request-lifecycle.diff`,
`patch-gloo-loopback-interface.diff`, `patch-runner-lifecycle-tests.diff`) and moved here on
2026-10-16 (ADR-0008).

## Merge of upstream v0.33.0 (2026-11-14)

`local` now contains upstream `v0.33.0` (2026-11-04) by a merge commit (branch `merge/upstream-v0.33.0`; history kept, no force-push, no tag cut). Differences from pure `v0.33.0` are `topic/gloo-loopback` and these docs only. The other two topics are **not carried** on this base:

- `topic/gemma4-assistant-probe`: obsolete. Upstream deleted the whole Gemma4 assistant config shim (`compat.py`) that the probe patched; with vLLM 0.33.0+local01 and transformers 5.21, `vllm_metal.register()` returns `MetalPlatform` without it.
- `topic/request-lifecycle`: **not re-applied; reproduce first, then decide** (eejd/py-vllm-metal#5). Upstream reworked the same code (`prefill_token_ids`, `_awaiting_readmission` / `_readmit_request`, `_collect_prefill_continuation`). Our lifecycle test file was run unchanged except two harness adaptations to vLLM 0.33 (`EngineCore._weight_version`, `StructuredOutputsParams._backend` is now set by vLLM's processor) against bare `v0.33.0` with vLLM 0.33.0+local01: 4 of 11 pass, including both end-to-end structured-output scenarios (`test_failed_first_sample_is_not_masked_by_an_empty_cached_prefill`, `test_mask_constrains_first_token_and_decode_across_prefix_hit_and_chunks`), i.e. the headline failure of macports-ports-local#1196 does not reproduce on the new stack. The 7 failures are: 3 that assert our admission timing (`KeyError` on the request id before the first sample), 3 invariant probes that upstream does not enforce (a prompt-phase state with its token not materialized is routed to decode without an error; a cached entry with no scheduled tokens enqueues work; `execute_model` does not refuse while a forward is unconsumed) and 1 scenario that did not trigger a preemption under this vLLM. Whether those invariants are reachable from the real scheduler on vLLM 0.33 is not shown. The original failure was seen on vLLM 0.31 and has not been re-run there (no vLLM 0.31 is installed). Applying our version on top of upstream's makes upstream's own runner tests fail (`KeyError`), so nothing was carried. The topic branch and its tests are unchanged on `origin/topic/request-lifecycle`.

Packaging for MacPorts is not here (Python 3.14 metadata, system MLX build overrides): upstream `v0.33.0` still declares `requires-python >=3.12,<3.14` and `mlx==0.32.4`, so the port patchfiles need refreshing separately. Building the native extension from source against the installed MLX needed two workarounds on this host: `nanobind` in the build environment and a patched copy of MLX's `extension.cmake` (the installed one hardcodes a Metal toolchain path, `MetalToolchain-v27.1.266.1`, that no longer exists after the macOS update).
