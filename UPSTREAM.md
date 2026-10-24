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
