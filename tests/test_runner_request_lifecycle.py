# SPDX-License-Identifier: Apache-2.0
"""Request-lifecycle regressions for the paged runner under real EngineCore ordering.

vLLM's real ``AsyncScheduler`` produces the ``SchedulerOutput`` stream and the
real ``EngineCore.step_with_batch_queue`` decides the order of ``execute_model``
and ``sample_tokens`` calls (including the deferred sampling used for
structured output). Only the model forward is replaced: a tiny LM whose greedy
next token is a pure function of its input token, so every forward input and
every emitted token can be checked exactly. Structured output uses a real
xgrammar matcher and bitmask over a tiny raw vocabulary.

The failure that motivated these tests (vllm-metal #1196): a final prefill that
fails while sampling left the request without any ``RequestState``, and the
engine's next step -- already scheduled as a cached decode -- died on
"no RequestState", hiding the sampling error.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import queue
import re
from collections import deque

import mlx.core as mx
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.request import FinishReason, Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

import vllm_metal.v1.model_runner as mr
from tests.stub_runner import make_stub_runner
from vllm_metal.attention.runtime.sdpa import SDPAPagedAttentionRuntime

xgr = pytest.importorskip("xgrammar")

VOCAB = 64
EOS = VOCAB - 1
BLOCK = 16


def successor(token: int) -> int:
    """The toy LM's greedy next token."""
    return (token * 7 + 3) % VOCAB


def chain(last_prompt_token: int, count: int) -> list[int]:
    """The unconstrained greedy continuation of ``count`` tokens."""
    out: list[int] = []
    token = last_prompt_token
    for _ in range(count):
        token = successor(token)
        out.append(token)
    return out


class ToyLM:
    """Forward stand-in recording every packed input; logits one-hot on successor."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        ids = input_ids[0].tolist()
        self.calls.append(ids)
        rows = [[10.0 if j == successor(t) else 0.0 for j in range(VOCAB)] for t in ids]
        return mx.array([rows])


def _raw_tokenizer() -> PreTrainedTokenizerFast:
    """Ten digit tokens, filler words, then EOS: matched by xgrammar as RAW vocab."""
    vocab = {str(i): i for i in range(10)}
    for i in range(10, EOS):
        vocab[f"w{i}"] = i
    vocab["<eos>"] = EOS
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="w10"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>")


@pytest.fixture(scope="module")
def model_config(tmp_path_factory: pytest.TempPathFactory) -> ModelConfig:
    model_dir = tmp_path_factory.mktemp("tiny-llama")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": VOCAB,
                "max_position_embeddings": 256,
                "num_key_value_heads": 2,
            }
        )
    )
    return ModelConfig(
        model=str(model_dir),
        tokenizer=str(model_dir),
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=256,
    )


@pytest.fixture(autouse=True)
def _isolate_platform_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Building a VllmConfig runs the Metal platform hook, which sets this.
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")


class Engine:
    """Real scheduler + real ``step_with_batch_queue`` + real runner on ToyLM."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        num_blocks: int = 64,
        max_batched_tokens: int = 24,
        max_num_seqs: int = 2,
    ) -> None:
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=CacheConfig(block_size=BLOCK, enable_prefix_caching=True),
            scheduler_config=SchedulerConfig(
                max_model_len=256,
                is_encoder_decoder=False,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_batched_tokens,
                async_scheduling=True,
                enable_chunked_prefill=True,
            ),
            parallel_config=ParallelConfig(),
        )
        vllm_config.cache_config.num_gpu_blocks = num_blocks
        spec = FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=2, head_size=8, dtype=torch.float16
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=spec.page_size_bytes * num_blocks,
                    layers=["l0"],
                    layer_stride=spec.page_size_bytes * num_blocks,
                    block_stride=spec.page_size_bytes,
                )
            ],
            kv_cache_groups=[KVCacheGroupSpec(["l0"], spec)],
        )
        self.structured_output = StructuredOutputManager(vllm_config)
        self.structured_output._use_async_grammar_compilation = False
        self.structured_output.backend = XgrammarBackend(
            vllm_config, tokenizer=_raw_tokenizer(), vocab_size=VOCAB
        )
        self.scheduler = AsyncScheduler(
            vllm_config, kv_cache_config, self.structured_output, block_size=BLOCK
        )
        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(BLOCK, sha256)

        self.lm = ToyLM()
        self.runner = make_stub_runner(
            model_args={"vocab_size": VOCAB},
            model=self.lm,
            _paged_attention_runtime=SDPAPagedAttentionRuntime(
                num_layers=1,
                num_kv_heads=1,
                head_dim=4,
                block_size=BLOCK,
                dtype=mx.float32,
            ),
            _paged_block_size=BLOCK,
            _rust_state_manager=None,
            num_layers=1,
            _intermediate_forward_supported=False,
        )

        # Recordings.
        self.scheduled: list[SchedulerOutput] = []
        # (req_id, start_pos, chunk tokens, full prompt handed to sampling)
        self.prefill_packs: list[
            tuple[str, int, tuple[int, ...], list[int] | None]
        ] = []
        self.emitted: dict[str, list[int]] = {}
        self.finish: dict[str, FinishReason | None] = {}
        self.admitted: dict[str, mr.RequestState] = {}
        self._wrap_runner()

        # The real EngineCore step, wired to the real executor/worker boundary.
        core = object.__new__(EngineCore)
        core.batch_queue = deque(maxlen=2)
        core.batch_queue_size = 2
        core.scheduler = self.scheduler
        core.is_ec_consumer = True  # no EC transfer configured
        core.is_pooling_model = False
        core.is_mm_encoder_only = False
        core.check_for_draft_tokens = False
        core.aborts_queue = queue.Queue()
        executor = object.__new__(UniProcExecutor)
        executor.driver_worker = self.runner
        core.model_executor = executor
        core._should_throttle_prefills = lambda: False
        core.log_error_detail = lambda scheduler_output: contextlib.nullcontext()
        core.capture_iteration_details = lambda so: contextlib.nullcontext(None)
        core._attach_iteration_details = lambda *args: None
        self.core = core

    def _wrap_runner(self) -> None:
        runner = self.runner
        execute_model = runner.execute_model
        build_prefill_pack = runner._build_prefill_pack

        def recording_execute(scheduler_output: SchedulerOutput):
            self.scheduled.append(scheduler_output)
            result = execute_model(scheduler_output)
            for new_req in scheduler_output.scheduled_new_reqs:
                state = runner._request_states.get(new_req.req_id)
                if state is not None:
                    self.admitted[new_req.req_id] = state
            return result

        def recording_pack(batch):
            pack = build_prefill_pack(batch)
            for prefill in pack:
                self.prefill_packs.append(
                    (
                        prefill.req_id,
                        prefill.start_pos,
                        tuple(prefill.token_ids),
                        prefill.full_prompt_token_ids,
                    )
                )
            return pack

        runner.execute_model = recording_execute
        runner._build_prefill_pack = recording_pack

    def add(
        self,
        req_id: str,
        prompt: list[int],
        max_tokens: int = 6,
        *,
        regex: str | None = None,
    ) -> Request:
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            structured_outputs=(
                StructuredOutputsParams(regex=regex) if regex is not None else None
            ),
        )
        if regex is not None:
            params.update_from_generation_config({}, EOS)
        request = Request(
            req_id,
            prompt_token_ids=prompt,
            sampling_params=params,
            pooling_params=None,
            arrival_time=0.0,
            block_hasher=self._block_hasher,
        )
        if request.use_structured_output:
            self.structured_output.grammar_init(request)
        self.scheduler.add_request(request)
        self.emitted.setdefault(req_id, [])
        return request

    def step(self) -> None:
        outputs, _ = self.core.step_with_batch_queue()
        for engine_outputs in (outputs or {}).values():
            for out in engine_outputs.outputs:
                self.emitted[out.request_id].extend(out.new_token_ids)
                if out.finish_reason is not None:
                    self.finish[out.request_id] = out.finish_reason

    def run(self, max_steps: int = 80) -> None:
        for _ in range(max_steps):
            if not (self.scheduler.has_requests() or self.core.batch_queue):
                return
            self.step()
        raise AssertionError("engine did not drain")

    def run_until(self, predicate, max_steps: int = 80) -> None:
        for _ in range(max_steps):
            if predicate():
                return
            self.step()
        raise AssertionError("condition not reached")

    @property
    def decode_inputs(self) -> list[int]:
        """Token fed to every single-token forward (decode rows), in call order."""
        return [call[0] for call in self.lm.calls if len(call) == 1]


class TestAdmittedRequestState:
    def test_final_prefill_advances_the_admitted_state_exactly_once(
        self, model_config: ModelConfig
    ) -> None:
        """No first-token loss, no duplicate advance, decode feeds the sampled token."""
        engine = Engine(model_config)
        prompt = list(range(1, 43))  # 42 tokens: chunks 24 + 18
        engine.add("r0", prompt, max_tokens=6)

        engine.run()

        admitted = engine.admitted["r0"]
        expected = chain(prompt[-1], 6)
        assert engine.emitted["r0"] == expected
        # The very object admitted before the first sample was advanced in place.
        assert admitted.token_ids == prompt + expected
        assert admitted.prompt_len == len(prompt)
        assert admitted.generated_tokens == 6
        # Every decode forward consumed the previously sampled token, never the
        # last prompt token and never an unsampled placeholder.
        assert engine.decode_inputs == expected[:-1]
        # Chunked prefill covered the prompt once, contiguously.
        covered = [t for call in engine.lm.calls if len(call) > 1 for t in call]
        assert covered == prompt

    def test_state_is_admitted_before_its_first_sample(
        self, model_config: ModelConfig
    ) -> None:
        engine = Engine(model_config)
        prompt = list(range(1, 11))
        engine.add("r0", prompt)
        engine.core.step_with_batch_queue()  # schedule + execute + sample step 1

        state = engine.admitted["r0"]
        assert engine.runner._request_states["r0"] is state
        assert state.prompt_len == len(prompt)
        assert state.token_ids[: len(prompt)] == prompt

    def test_second_advance_of_a_final_prefill_state_is_rejected(
        self, model_config: ModelConfig
    ) -> None:
        engine = Engine(model_config)
        prompt = list(range(1, 11))
        engine.add("r0", prompt)
        scheduler_output = engine.scheduler.schedule()
        assert engine.runner.execute_model(scheduler_output) is None
        # A second advance of the same state (e.g. a stale first-token write).
        engine.runner._request_states["r0"].token_ids.append(5)

        with pytest.raises(RuntimeError, match="desynchronized state"):
            engine.runner.sample_tokens(None)


class TestStructuredOutputOverRealOrdering:
    def test_mask_constrains_first_token_and_decode_across_prefix_hit_and_chunks(
        self, model_config: ModelConfig
    ) -> None:
        engine = Engine(model_config, max_num_seqs=2)
        shared = list(range(1, 33))  # two full blocks
        engine.add("warm", shared + [40, 41, 42], max_tokens=2)
        engine.run()
        engine.prefill_packs.clear()
        engine.lm.calls.clear()

        prompt = shared + list(range(50, 62))  # 44 tokens, 32 reusable from "warm"
        engine.add("so", prompt, max_tokens=8, regex="[0-9]{3}")
        engine.add("plain", list(range(20, 30)), max_tokens=4)
        engine.run()

        # The scheduler really reported a prefix-cache hit for the new request.
        new_so = next(
            n
            for out in engine.scheduled
            for n in out.scheduled_new_reqs
            if n.req_id == "so"
        )
        assert new_so.num_computed_tokens == 32

        # Chunked-prefix boundaries: contiguous from the hit to the prompt end,
        # each continuation carrying the whole prompt for penalties/metadata.
        packs = [p for p in engine.prefill_packs if p[0] == "so"]
        assert packs[0][1] == 32
        assert packs[0][2] == tuple(prompt[32 : 32 + len(packs[0][2])])
        position = 32
        for _, start, tokens, full_prompt in packs:
            assert start == position
            position += len(tokens)
            if start > 0:
                assert full_prompt == prompt
        assert position == len(prompt)

        # Grammar applied to the first sampled token and every decode token.
        sampled = engine.emitted["so"]
        assert [t for t in sampled if t != EOS] != []
        digits = "".join(str(t) for t in sampled if t != EOS)
        assert re.fullmatch("[0-9]{3}", digits)
        assert sampled[-1] == EOS
        assert engine.finish["so"] == FinishReason.STOP
        # The unconstrained request in the same batches is untouched by the mask.
        assert engine.emitted["plain"] == chain(29, 4)
        # Decode forwards consumed the sampled (masked) tokens, in order.
        remaining = iter(t for call in engine.lm.calls for t in call)
        assert all(token in remaining for token in sampled[:-1])

    def test_failed_first_sample_is_not_masked_by_an_empty_cached_prefill(
        self, model_config: ModelConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#1196: a failed first sample must not turn into an empty cached prefill."""
        monkeypatch.setattr("vllm_metal.v1.structured_output.xgr", None)
        engine = Engine(model_config)
        prompt = list(range(1, 43))
        engine.add("r0", prompt, max_tokens=6, regex="[0-9]{3}")

        with pytest.raises(
            RuntimeError, match="sample_tokens failed on the previous step.*xgrammar"
        ):
            engine.run()

        state = engine.runner._request_states["r0"]
        assert state.generated_tokens == 0
        assert state.token_ids == prompt
        assert engine.runner._execute_model_state is None
        # Only the two prompt chunks were forwarded: no empty prefill and no
        # decode of the last prompt token.
        assert [t for call in engine.lm.calls for t in call] == prompt


class TestCachedRequestClassification:
    def _admit_and_sample(self, engine: Engine, prompt: list[int]) -> mr.RequestState:
        engine.add("r0", prompt, max_tokens=8)
        engine.run_until(lambda: "r0" in engine.runner._request_states)
        return engine.runner._request_states["r0"]

    def test_unknown_cached_request_is_still_fatal(
        self, model_config: ModelConfig
    ) -> None:
        engine = Engine(model_config)
        scheduler_output = _cached_output("ghost", computed=42, scheduled=1)

        with pytest.raises(RuntimeError, match="no RequestState"):
            engine.runner.execute_model(scheduler_output)

        assert engine.lm.calls == []

    def test_unsampled_prompt_state_is_not_routed_to_decode(
        self, model_config: ModelConfig
    ) -> None:
        engine = Engine(model_config)
        prompt = list(range(1, 11))
        engine.runner._request_states["r0"] = mr.RequestState(
            token_ids=list(prompt),
            prompt_len=len(prompt),
            sampling_params=SamplingParams(temperature=0),
            block_ids=[[1]],
        )

        with pytest.raises(RuntimeError, match="did not materialize its token"):
            engine.runner.execute_model(
                _cached_output("r0", computed=len(prompt), scheduled=1)
            )

        assert engine.lm.calls == []
        assert engine.runner._execute_model_state is None

    def test_metadata_only_cached_entry_enqueues_no_work(
        self, model_config: ModelConfig
    ) -> None:
        engine = Engine(model_config)
        prompt = list(range(1, 11))
        engine.add("r0", prompt, max_tokens=8)
        engine.run_until(lambda: engine.lm.calls)  # first forward done
        engine.run_until(lambda: "r0" in engine.emitted and engine.emitted["r0"])
        calls_before = len(engine.lm.calls)
        batch = mr._ExecutionBatch()
        cached = CachedRequestData(
            req_ids=["r0"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[len(prompt)],
            num_output_tokens=[1],
        )

        scheduler_output = dataclasses.replace(
            _cached_output("r0", len(prompt), 1), num_scheduled_tokens={}
        )

        engine.runner._collect_cached_requests(batch, cached, scheduler_output)

        assert batch.paged_prefill_entries == []
        assert batch.paged_decode_reqs == []
        assert batch.req_ids == []
        assert len(engine.lm.calls) == calls_before

    def test_execute_model_never_replaces_an_unsampled_forward(
        self, model_config: ModelConfig
    ) -> None:
        engine = Engine(model_config)
        engine.add("r0", list(range(1, 11)))
        first = engine.scheduler.schedule()
        assert engine.runner.execute_model(first) is None
        pending = engine.runner._execute_model_state
        assert pending is not None

        with pytest.raises(RuntimeError, match="not been consumed by sample_tokens"):
            engine.runner.execute_model(_cached_output("r0", computed=10, scheduled=1))

        assert engine.runner._execute_model_state is pending


class TestFinishedPreemptedResumedLifecycle:
    def test_preempted_request_resumes_and_continues_its_exact_chain(
        self, model_config: ModelConfig
    ) -> None:
        # Six usable blocks: two 40-token prompts fit, their growth does not.
        engine = Engine(model_config, num_blocks=7)
        p0, p1 = list(range(1, 41)), list(range(101, 141))
        engine.add("r0", p0, max_tokens=14)
        engine.add("r1", p1, max_tokens=14)

        engine.run()

        resumed_steps = [
            out
            for out in engine.scheduled
            if out.scheduled_cached_reqs.resumed_req_ids
        ]
        assert resumed_steps, "scenario must exercise a real preemption/resume"
        assert any(out.preempted_req_ids for out in engine.scheduled)
        # Recompute after resume appends exactly one new token per step: the
        # emitted stream is the unbroken greedy chain for both requests.
        assert engine.emitted["r0"] == chain(p0[-1], 14)
        assert engine.emitted["r1"] == chain(p1[-1], 14)
        assert engine.finish == {
            "r0": FinishReason.LENGTH,
            "r1": FinishReason.LENGTH,
        }

    def test_finished_requests_release_runner_state_and_ids_can_be_reused(
        self, model_config: ModelConfig
    ) -> None:
        engine = Engine(model_config)
        prompt = list(range(1, 11))
        engine.add("r0", prompt, max_tokens=3)
        engine.run()
        first_state = engine.admitted["r0"]
        assert "r0" not in engine.runner._request_states
        assert "r0" not in engine.runner._paged_request_seq_lens

        engine.add("r0", list(range(20, 31)), max_tokens=3)
        engine.run()

        second_state = engine.admitted["r0"]
        assert second_state is not first_state
        assert engine.emitted["r0"] == chain(10, 3) + chain(30, 3)
        assert engine.runner._request_states == {}


def _cached_output(req_id: str, computed: int, scheduled: int) -> SchedulerOutput:
    """A hand-built step carrying one cached request (no new or finished work)."""
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[computed],
            num_output_tokens=[1],
        ),
        num_scheduled_tokens={req_id: scheduled},
        total_num_scheduled_tokens=scheduled,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=set(),
        has_structured_output_requests=False,
    )
