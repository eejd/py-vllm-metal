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
- `topic/request-lifecycle`: **not re-applied, needs re-derivation.** Upstream reworked the same code (`prefill_token_ids`, `_awaiting_readmission` / `_readmit_request`, `_collect_prefill_continuation`) but still classifies cached requests by `state.generated_tokens == 0` and creates the state when the final prefill is sampled, so it does not have our computed-token classification, the desync error, the metadata-only skip, or the refusal to start a forward while one is unconsumed. Re-applying our version on top breaks upstream's own tests that drive `_start_paged_forward` without admitting first (`KeyError` on the request id in `test_v1_model_runner_generate.py`), so the two admission models have to be reconciled, ideally as one change proposed upstream. Until then a stack that needs the fix for a failed first structured-output sample stays on `v0.31.0-local02`. The topic branch and its tests (`tests/test_runner_request_lifecycle.py`) are unchanged on `origin/topic/request-lifecycle`.

Packaging for MacPorts is not here (Python 3.14 metadata, system MLX build overrides): upstream `v0.33.0` still declares `requires-python >=3.12,<3.14` and `mlx==0.32.4`, so the port patchfiles need refreshing separately. Building the native extension from source against the installed MLX needed two workarounds on this host: `nanobind` in the build environment and a patched copy of MLX's `extension.cmake` (the installed one hardcodes a Metal toolchain path, `MetalToolchain-v27.1.266.1`, that no longer exists after the macOS update).
