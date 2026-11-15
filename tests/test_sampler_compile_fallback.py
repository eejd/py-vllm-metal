# SPDX-License-Identifier: Apache-2.0
"""The Metal sampler survives a failing torch compiler without resampling."""

import mlx.core as mx
import pytest
import torch
import vllm.v1.sample.sampler as upstream_sampler
from vllm.sampling_params import SamplingParams

from vllm_metal.v1.sampling_batch import MetalSampler, SamplingBatch, sample_from_logits


@pytest.mark.parametrize("params", [{"logprobs": 1}, {"logprob_token_ids": [0, 3]}])
def test_failed_compile_falls_back_once_without_resampling(monkeypatch, caplog, params):
    attempts = []

    def failing_backend(graph, inputs):
        attempts.append(1)
        raise RuntimeError("compiler unavailable")

    compiled = torch.compile(lambda x, y: (x >= y).sum(dim=-1), backend=failing_backend)
    monkeypatch.setattr(upstream_sampler, "batched_count_greater_than", compiled)
    sampler = MetalSampler()
    calls = []
    sample = sampler.sample

    def sample_once(*args, **kwargs):
        calls.append(1)
        return sample(*args, **kwargs)

    monkeypatch.setattr(sampler, "sample", sample_once)
    batch = SamplingBatch(
        [SamplingParams(temperature=0, **params)], [[1]], [[]], vocab_size=4
    )
    for _ in range(2):
        result = sample_from_logits(mx.array([[0.0, 1.0, 4.0, 2.0]]), batch, sampler)
        assert result.token_ids == [2]
        assert result.logprobs.sampled_token_ranks.tolist() == [1]
        expected = torch.log_softmax(torch.tensor([0.0, 1.0, 4.0, 2.0]), 0)
        assert result.logprobs.logprobs[0, 0] == pytest.approx(expected[2].item())
    assert len(attempts) == 1
    assert len(calls) == 2
    assert caplog.text.count("falling back to eager") == 1


def test_non_compile_errors_propagate(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("real sampler error")

    monkeypatch.setattr(upstream_sampler.Sampler, "gather_logprobs", fail)
    sampler = MetalSampler()
    with pytest.raises(ValueError, match="real sampler error"):
        sampler.gather_logprobs(torch.zeros((1, 4)), 1, torch.tensor([0]))
