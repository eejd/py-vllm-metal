# Upstream and adopted work (`local` branch)

Primary upstream: `vllm-project/vllm-metal` (GitHub fork parent). Secondary upstreams tracked: none.

## Adopted from other repos

None. (Commits adopted from other repos would be listed here with `Source: <repo>@<sha>` and
`Co-authored-by` trailers on the commit.)

## Topic branches

| Topic | Upstream PR | Status |
|---|---|---|
| `topic/request-lifecycle` | none yet | **dropped 2026-11-14, not in `local` since v0.33.0** (superseded; see below). Historical: not-submitted (found and soak-tested via macports-ports-local#1196; checked 2026-10-16 that upstream `main` @ `468b0c8` has no equivalent: the patch applies unchanged to `main` and to `v0.31.0`) |
| `topic/gloo-loopback` | none yet | not-submitted (upstream #633 "Use loopback for default UniProc rendezvous" sets `VLLM_HOST_IP`, not `GLOO_SOCKET_IFNAME`; `main` only sets it in `tools/*.py`) |
| `topic/gemma4-assistant-probe` | none yet | **dropped 2026-11-14, not in `local` since v0.33.0** (obsolete; see below). Historical: not-submitted (compat probe only catches ValueError; transformers 5.20 raises StrictDataclassError) |

## Origin

The first two topics were developed as `files/` patches of the MacPorts port `python/py-vllm-metal-devel`
(`eejd/macports-ports-local`, patches `patch-runner-request-lifecycle.diff`,
`patch-gloo-loopback-interface.diff`, `patch-runner-lifecycle-tests.diff`) and moved here on
2026-10-16 (ADR-0008).

## Merge of upstream v0.33.0 (2026-11-14)

`local` now contains upstream `v0.33.0` (2026-11-04) by a merge commit (branch `merge/upstream-v0.33.0`; history kept, no force-push, no tag cut). Differences from pure `v0.33.0` are `topic/gloo-loopback` and these docs only. The other two topics are **not carried** on this base:

- `topic/gemma4-assistant-probe`: obsolete. Upstream deleted the Gemma4 assistant config shim in `compat.py` that the probe patched; with vLLM 0.33.0+local01 and transformers 5.21, `vllm_metal.register()` returns `MetalPlatform` without it.
- `topic/request-lifecycle`: **dropped, superseded on v0.33.0** (eejd/py-vllm-metal#5). The symptom it fixed (macports-ports-local#1196: a `json_schema` request killing EngineCore with "Scheduled cached request(s) have no RequestState", seen on vLLM 0.27.1) does not reproduce on bare v0.33.0 with a real model (Llama-3.2-1B-Instruct-4bit): sequential and 24-way concurrent `json_schema`/plain/aborted requests, and a chunked-prefill (256-token chunks) run on a 100-block KV pool that forced 37 preemptions with structured output, all HTTP 200, engine alive. Our lifecycle tests still fail 7 of 11 on bare v0.33.0 (admission-timing assertions and three invariant probes), which only shows upstream's design differs. Not covered: async scheduling, speculative decoding, other models. The branch tip is kept as tag `archive/topic-request-lifecycle-v0.31` (a92df6f); `v0.31.0-local02` also contains it.

Packaging for MacPorts is not here (Python 3.14 metadata, system MLX build overrides): upstream `v0.33.0` still declares `requires-python >=3.12,<3.14` and `mlx==0.32.4`, so the port patchfiles need refreshing separately. Building the native extension from source against the installed MLX needed two workarounds on this host: `nanobind` in the build environment and a patched copy of MLX's `extension.cmake` (the installed one hardcodes a Metal toolchain path, `MetalToolchain-v27.1.266.1`, that no longer exists after the macOS update).
