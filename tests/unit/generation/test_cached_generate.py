import torch

from tokserve.engine.model import LlamaModel
from tokserve.generation.cached_generate import generate_with_cache
from tokserve.generation.generate import generate
from tokserve.generation.types import GenerationConfig
from tokserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a small model configuration for generation tests."""

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


def zero_model() -> LlamaModel:
    """Return a model whose logits are equal for every token."""

    model = LlamaModel(tiny_config())

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    model.eval()
    return model


def test_cached_greedy_generation_matches_uncached_generation() -> None:
    torch.manual_seed(40)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt = torch.tensor([[1, 4, 7]], dtype=torch.int64)
    config = GenerationConfig(max_new_tokens=4)

    uncached = generate(model, prompt, config)
    cached = generate_with_cache(model, prompt, config)

    torch.testing.assert_close(cached.token_ids, uncached.token_ids)
    torch.testing.assert_close(
        cached.generated_token_ids,
        uncached.generated_token_ids,
    )
    assert cached.stop_reason == uncached.stop_reason


def test_cached_generation_prefills_prompt_once() -> None:
    torch.manual_seed(41)
    model = LlamaModel(tiny_config())
    model.eval()
    processed_lengths: list[int] = []

    def record_input_length(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        processed_lengths.append(inputs[0].shape[1])

    handle = model.embed_tokens.register_forward_hook(record_input_length)

    try:
        generate_with_cache(
            model,
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=4),
        )
    finally:
        handle.remove()

    assert processed_lengths == [4, 1, 1, 1]


def test_cached_decode_never_resends_the_prompt() -> None:
    torch.manual_seed(42)
    model = LlamaModel(tiny_config())
    model.eval()
    processed_token_counts: list[int] = []

    def record_processed_tokens(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        processed_token_counts.append(inputs[0].numel())

    handle = model.embed_tokens.register_forward_hook(record_processed_tokens)

    try:
        generate_with_cache(
            model,
            torch.tensor([[1, 2, 3]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=3),
        )
    finally:
        handle.remove()

    assert processed_token_counts == [3, 1, 1]


def test_cached_generation_preserves_prompt_and_token_limit() -> None:
    torch.manual_seed(43)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt = torch.tensor([[2, 5, 8]], dtype=torch.int64)

    result = generate_with_cache(
        model,
        prompt,
        GenerationConfig(max_new_tokens=5),
    )

    torch.testing.assert_close(result.token_ids[:, :3], prompt)
    assert result.token_ids.shape == (1, 8)
    assert result.generated_token_ids.shape == (1, 5)
    assert result.num_generated_tokens == 5
    assert result.stop_reason == "max_new_tokens"
    assert torch.all(result.generated_token_ids >= 0)
    assert torch.all(result.generated_token_ids < model.config.vocab_size)


def test_cached_generation_stops_on_eos() -> None:
    model = zero_model()
    prompt = torch.tensor([[1, 2]], dtype=torch.int64)

    result = generate_with_cache(
        model,
        prompt,
        GenerationConfig(
            max_new_tokens=5,
            eos_token_id=0,
        ),
    )

    torch.testing.assert_close(
        result.token_ids,
        torch.tensor([[1, 2, 0]]),
    )
    torch.testing.assert_close(
        result.generated_token_ids,
        torch.tensor([[0]]),
    )
    assert result.num_generated_tokens == 1
    assert result.stop_reason == "eos"


def test_zero_new_tokens_does_not_run_model() -> None:
    model = LlamaModel(tiny_config())
    forward_calls = 0

    def count_calls(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal forward_calls
        forward_calls += 1

    handle = model.embed_tokens.register_forward_hook(count_calls)

    try:
        prompt = torch.tensor([[1, 2]], dtype=torch.int64)
        result = generate_with_cache(
            model,
            prompt,
            GenerationConfig(max_new_tokens=0),
        )
    finally:
        handle.remove()

    assert forward_calls == 0
    torch.testing.assert_close(result.token_ids, prompt)
    assert result.generated_token_ids.shape == (1, 0)
    assert result.stop_reason == "max_new_tokens"


def test_cached_seeded_sampling_is_reproducible() -> None:
    model = zero_model()
    prompt = torch.tensor([[1, 2]], dtype=torch.int64)
    config = GenerationConfig(
        max_new_tokens=6,
        do_sample=True,
        temperature=0.8,
        top_k=12,
        top_p=0.9,
        seed=44,
    )

    first = generate_with_cache(model, prompt, config)
    second = generate_with_cache(model, prompt, config)

    torch.testing.assert_close(first.token_ids, second.token_ids)


def test_cached_seeded_sampling_matches_uncached_rng_consumption() -> None:
    model = zero_model()
    prompt = torch.tensor([[1, 2]], dtype=torch.int64)
    config = GenerationConfig(
        max_new_tokens=6,
        do_sample=True,
        temperature=0.9,
        top_k=10,
        top_p=0.8,
        seed=45,
    )

    uncached = generate(model, prompt, config)
    cached = generate_with_cache(model, prompt, config)

    torch.testing.assert_close(cached.token_ids, uncached.token_ids)


def test_cached_generation_disables_gradients() -> None:
    torch.manual_seed(46)
    model = LlamaModel(tiny_config())
    model.eval()
    gradient_states: list[bool] = []

    def record_gradient_state(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        gradient_states.append(torch.is_grad_enabled())

    handle = model.embed_tokens.register_forward_hook(record_gradient_state)

    try:
        generate_with_cache(
            model,
            torch.tensor([[1, 2]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=3),
        )
    finally:
        handle.remove()

    assert gradient_states == [False, False, False]


def test_cached_generation_rejects_sequence_beyond_model_limit() -> None:
    model = LlamaModel(tiny_config())
    prompt = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
        dtype=torch.int64,
    )

    try:
        generate_with_cache(
            model,
            prompt,
            GenerationConfig(max_new_tokens=3),
        )
    except ValueError as error:
        assert str(error) == (
            "prompt and generated tokens exceed max_position_embeddings"
        )
    else:
        raise AssertionError("generation should reject excessive sequence length")
