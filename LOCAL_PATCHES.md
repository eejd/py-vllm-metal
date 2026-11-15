# Local patches (`local` branch)

`local` = upstream release `v0.33.0` (merged 2026-11-14) + the `topic/*` branches below. Two earlier topics are gone from this tree: `request-lifecycle` (superseded on v0.33.0; tip archived as tag `archive/topic-request-lifecycle-v0.31`) and `gemma4-assistant-probe` (obsolete; upstream deleted the shim). `v0.31.0-local02` and older tags still contain them. See `UPSTREAM.md`.
It is the source of the MacPorts port `py-vllm-metal-local` (tag `v<base>-local<NN>`).
`main` is a fast-forward mirror of `vllm-project/vllm-metal` `main` and is never edited; the topics
are written against `main` and applied to the release base. Each topic is ONE upstreamable change
and is the only source of upstream PRs. Packaging-only patches (Python 3.14 metadata, system-MLX
build overrides, dependency bounds relaxed to MacPorts versions) are NOT here: they stay
`patchfiles` of the MacPorts port.

| Behaviour | Topic | Files | Tests | Upstream |
|---|---|---|---|---|
| `GLOO_SOCKET_IFNAME=lo0` for single-node, single-process serving (TP=PP=DP=1) when unset; explicit values (even empty) and every multi-process / multi-node topology untouched. Avoids gloo resolving the host name (tens of seconds, or a VPN tunnel). | `topic/gloo-loopback` | `vllm_metal/platform.py` | `tests/test_platform.py`, `tests/conftest.py` (env restore fixture), `tests/test_kv_offload_config.py` | not submitted (macports-ports-local#1196); related but different: upstream #633 sets `VLLM_HOST_IP` |

Rebasing: archive the old tip as `archive/local-pre-rebase-<sha>`, re-apply the topics on the new
release base, keep these rows, run `pytest tests` (some tests need network: compare with the base).
