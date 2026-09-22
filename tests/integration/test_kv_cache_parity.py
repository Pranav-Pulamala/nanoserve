import torch

from tokserve.engine.inference import decode, prefill
from tokserve.engine.model import LlamaModel
from tokserve.generation.cached_generate import generate_with_cache
from tokserve.generation.generate import generate
from tokserve.generation.types import GenerationConfig
from tokserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a deterministic tiny model configuration."""

    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
    )


def test_prefill_logits_match_uncached_logits() -> None:
    torch.manual_seed(50)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt_ids = torch.tensor([[1, 4, 7, 2]], dtype=torch.int64)

    uncached_logits = model(prompt_ids)
    prefill_result = prefill(model, prompt_ids)

    torch.testing.assert_close(
        prefill_result.logits,
        uncached_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    assert prefill_result.cache.current_length == prompt_ids.shape[1]


def test_one_token_cached_decode_matches_uncached_final_logits() -> None:
    torch.manual_seed(51)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt_ids = torch.tensor([[1, 4, 7]], dtype=torch.int64)
    next_token = torch.tensor([[9]], dtype=torch.int64)

    prefill_result = prefill(model, prompt_ids)
    cached_logits = decode(
        model,
        next_token,
        prefill_result.cache,
    )

    complete_ids = torch.cat((prompt_ids, next_token), dim=1)
    uncached_logits = model(complete_ids)[:, -1:, :]

    torch.testing.assert_close(
        cached_logits,
        uncached_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    assert prefill_result.cache.current_length == 4


def test_multiple_cached_decode_steps_match_uncached_reference() -> None:
    torch.manual_seed(52)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    continuation = [
        torch.tensor([[4]], dtype=torch.int64),
        torch.tensor([[5]], dtype=torch.int64),
        torch.tensor([[6]], dtype=torch.int64),
    ]

    prefill_result = prefill(model, prompt_ids)
    complete_ids = prompt_ids.clone()

    for expected_length, token_ids in enumerate(continuation, start=4):
        cached_logits = decode(
            model,
            token_ids,
            prefill_result.cache,
        )
        complete_ids = torch.cat((complete_ids, token_ids), dim=1)
        uncached_logits = model(complete_ids)[:, -1:, :]

        torch.testing.assert_close(
            cached_logits,
            uncached_logits,
            rtol=1e-5,
            atol=1e-6,
        )
        assert prefill_result.cache.current_length == expected_length


def test_every_layer_retains_independent_hkv_cache() -> None:
    torch.manual_seed(53)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)

    prefill_result = prefill(model, prompt_ids)
    cache = prefill_result.cache

    assert len(cache) == model.config.num_hidden_layers
    assert all(
        layer.keys.shape
        == (
            1,
            model.config.num_key_value_heads,
            prompt_ids.shape[1],
            model.config.head_dim,
        )
        for layer in cache.layers
    )
    assert all(
        layer.values.shape
        == (
            1,
            model.config.num_key_value_heads,
            prompt_ids.shape[1],
            model.config.head_dim,
        )
        for layer in cache.layers
    )
    assert cache[0].keys.data_ptr() != cache[1].keys.data_ptr()
    assert cache[0].values.data_ptr() != cache[1].values.data_ptr()


def test_cached_greedy_generation_exactly_matches_uncached() -> None:
    torch.manual_seed(54)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt_ids = torch.tensor([[1, 4, 7]], dtype=torch.int64)
    config = GenerationConfig(max_new_tokens=5)

    uncached_result = generate(
        model,
        prompt_ids,
        config,
    )
    cached_result = generate_with_cache(
        model,
        prompt_ids,
        config,
    )

    assert torch.equal(
        cached_result.token_ids,
        uncached_result.token_ids,
    )
    assert torch.equal(
        cached_result.generated_token_ids,
        uncached_result.generated_token_ids,
    )
    assert cached_result.stop_reason == uncached_result.stop_reason


def test_cached_seeded_sampling_matches_uncached_for_uniform_logits() -> None:
    model = LlamaModel(tiny_config())

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    model.eval()
    prompt_ids = torch.tensor([[1, 2]], dtype=torch.int64)
    config = GenerationConfig(
        max_new_tokens=6,
        do_sample=True,
        temperature=0.8,
        top_k=12,
        top_p=0.9,
        seed=55,
    )

    uncached_result = generate(
        model,
        prompt_ids,
        config,
    )
    cached_result = generate_with_cache(
        model,
        prompt_ids,
        config,
    )

    assert torch.equal(
        cached_result.token_ids,
        uncached_result.token_ids,
    )


def test_decode_projects_only_the_new_token() -> None:
    torch.manual_seed(56)
    model = LlamaModel(tiny_config())
    model.eval()
    first_attention = model.layers[0].self_attention
    projected_lengths: dict[str, list[int]] = {
        "query": [],
        "key": [],
        "value": [],
    }

    def record_query_length(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        projected_lengths["query"].append(inputs[0].shape[1])

    def record_key_length(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        projected_lengths["key"].append(inputs[0].shape[1])

    def record_value_length(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        projected_lengths["value"].append(inputs[0].shape[1])

    handles = [
        first_attention.q_proj.register_forward_hook(record_query_length),
        first_attention.k_proj.register_forward_hook(record_key_length),
        first_attention.v_proj.register_forward_hook(record_value_length),
    ]

    try:
        prefill_result = prefill(
            model,
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
        )
        decode(
            model,
            torch.tensor([[5]], dtype=torch.int64),
            prefill_result.cache,
        )
        decode(
            model,
            torch.tensor([[6]], dtype=torch.int64),
            prefill_result.cache,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert projected_lengths == {
        "query": [4, 1, 1],
        "key": [4, 1, 1],
        "value": [4, 1, 1],
    }


def test_decode_preserves_previous_key_value_prefixes() -> None:
    torch.manual_seed(57)
    model = LlamaModel(tiny_config())
    model.eval()
    prefill_result = prefill(
        model,
        torch.tensor([[1, 2, 3]], dtype=torch.int64),
    )

    previous_keys = [layer.keys.clone() for layer in prefill_result.cache.layers]
    previous_values = [layer.values.clone() for layer in prefill_result.cache.layers]

    decode(
        model,
        torch.tensor([[4]], dtype=torch.int64),
        prefill_result.cache,
    )

    for layer_index, layer in enumerate(prefill_result.cache.layers):
        torch.testing.assert_close(
            layer.keys[:, :, :3, :],
            previous_keys[layer_index],
        )
        torch.testing.assert_close(
            layer.values[:, :, :3, :],
            previous_values[layer_index],
        )


def test_cached_generation_produces_valid_cpu_token_ids() -> None:
    torch.manual_seed(58)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt_ids = torch.tensor([[2, 5, 8]], dtype=torch.int64)

    result = generate_with_cache(
        model,
        prompt_ids,
        GenerationConfig(max_new_tokens=4),
    )

    assert result.token_ids.device.type == "cpu"
    assert result.token_ids.shape == (1, 7)
    assert torch.equal(result.token_ids[:, :3], prompt_ids)
    assert torch.all(result.generated_token_ids >= 0)
    assert torch.all(result.generated_token_ids < model.config.vocab_size)
