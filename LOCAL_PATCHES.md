# Local patches (`local` branch)

`local` = upstream release `v0.31.0` + the `topic/*` branches below (cherry-picked, `-x`).
It is the source of the MacPorts port `py-vllm-metal-local` (tag `v<base>-local<NN>`).
`main` is a fast-forward mirror of `vllm-project/vllm-metal` `main` and is never edited; the topics
are written against `main` and applied to the release base. Each topic is ONE upstreamable change
and is the only source of upstream PRs. Packaging-only patches (Python 3.14 metadata, system-MLX
build overrides, dependency bounds relaxed to MacPorts versions) are NOT here: they stay
`patchfiles` of the MacPorts port.

| Behaviour | Topic | Files | Tests | Upstream |
|---|---|---|---|---|
| Exactly one `RequestState` per paged request, admitted before any forward; the final-prefill sample advances it once; cached steps classified by the scheduler's computed-token boundary (prompt continuation / resume recompute -> prefill, unknown scheduled position refused, metadata-only entries enqueue no work); `execute_model` refuses to start while a forward is unconsumed by `sample_tokens`; PP stage mirror never replaces an admitted state. Fixes the engine dying on a failed first structured-output sample under async scheduling. | `topic/request-lifecycle` | `vllm_metal/v1/model_runner.py` | `tests/test_runner_request_lifecycle.py` (new, 11 tests; 7 fail on v0.31.0 without the fix), `tests/test_v1_model_runner_generate.py`, `tests/v1/mm/test_paged_forward_mm.py` | not submitted (macports-ports-local#1196) |
| `GLOO_SOCKET_IFNAME=lo0` for single-node, single-process serving (TP=PP=DP=1) when unset; explicit values (even empty) and every multi-process / multi-node topology untouched. Avoids gloo resolving the host name (tens of seconds, or a VPN tunnel). | `topic/gloo-loopback` | `vllm_metal/platform.py` | `tests/test_platform.py`, `tests/conftest.py` (env restore fixture), `tests/test_kv_offload_config.py` | not submitted (macports-ports-local#1196); related but different: upstream #633 sets `VLLM_HOST_IP` |

Rebasing: archive the old tip as `archive/local-pre-rebase-<sha>`, re-apply the topics on the new
release base, keep these rows, run `pytest tests` (some tests need network: compare with the base).
