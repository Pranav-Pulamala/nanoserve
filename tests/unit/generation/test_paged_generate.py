from unittest.mock import call, patch

import pytest
import torch

from nanoserve.engine.model import LlamaModel
from nanoserve.engine.paged.allocator import BlockExhaustedError
from nanoserve.engine.paged.manager import PagedKVCacheManager
from nanoserve.generation.cached_generate import generate_with_cache
from nanoserve.generation.generate import generate
from nanoserve.generation.paged_generate import generate_with_paged_cache
from nanoserve.generation.types import GenerationConfig
from nanoserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a small deterministic Llama configuration."""

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


def create_manager(
    model: LlamaModel,
    *,
    num_blocks: int = 8,
    block_size: int = 2,
) -> PagedKVCacheManager:
    """Create a paged pool compatible with the model."""

    parameter = model.embed_tokens.weight

    return PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=num_blocks,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=block_size,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def zero_model() -> LlamaModel:
    """Return a model with equal logits for all vocabulary tokens."""

    model = LlamaModel(tiny_config())

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    model.eval()
    return model


def test_greedy_tokens_match_uncached_and_contiguous_generation() -> None:
    torch.manual_seed(80)
    model = LlamaModel(tiny_config())
    model.eval()
    manager = create_manager(model)
    prompt = torch.tensor([[1, 4, 7]], dtype=torch.int64)
    config = GenerationConfig(max_new_tokens=5)

    uncached = generate(model, prompt, config)
    contiguous = generate_with_cache(model, prompt, config)
    paged = generate_with_paged_cache(
        model,
        prompt,
        config,
        manager,
        sequence_id="greedy",
    )

    assert torch.equal(paged.token_ids, uncached.token_ids)
    assert torch.equal(paged.token_ids, contiguous.token_ids)
    assert paged.stop_reason == uncached.stop_reason
    assert paged.stop_reason == contiguous.stop_reason
    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0


def test_paged_generation_prefills_once_and_decodes_new_tokens_only() -> None:
    torch.manual_seed(81)
    model = LlamaModel(tiny_config())
    model.eval()
    manager = create_manager(model)
    processed_lengths: list[int] = []

    def record_embedding_input(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        processed_lengths.append(inputs[0].shape[1])

    handle = model.embed_tokens.register_forward_hook(record_embedding_input)

    try:
        generate_with_paged_cache(
            model,
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=4),
            manager,
            sequence_id="lengths",
        )
    finally:
        handle.remove()

    assert processed_lengths == [4, 1, 1, 1]
    assert manager.active_sequence_ids == ()


def test_blocks_are_allocated_only_when_needed() -> None:
    torch.manual_seed(82)
    model = LlamaModel(tiny_config())
    model.eval()
    manager = create_manager(model, block_size=2)

    with patch.object(
        manager.allocator,
        "allocate_many",
        wraps=manager.allocator.allocate_many,
    ) as allocate_many:
        generate_with_paged_cache(
            model,
            torch.tensor([[1, 2, 3]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=4),
            manager,
            sequence_id="allocation",
        )

    assert allocate_many.call_args_list == [call(2), call(1)]
    assert manager.allocator.free_count == manager.allocator.num_blocks


def test_paged_generation_preserves_prompt_and_token_limit() -> None:
    torch.manual_seed(83)
    model = LlamaModel(tiny_config())
    model.eval()
    manager = create_manager(model)
    prompt = torch.tensor([[2, 5, 8]], dtype=torch.int64)

    result = generate_with_paged_cache(
        model,
        prompt,
        GenerationConfig(max_new_tokens=5),
        manager,
        sequence_id="limit",
    )

    torch.testing.assert_close(result.token_ids[:, :3], prompt)
    assert result.token_ids.shape == (1, 8)
    assert result.generated_token_ids.shape == (1, 5)
    assert result.num_generated_tokens == 5
    assert result.stop_reason == "max_new_tokens"
    assert torch.all(result.generated_token_ids >= 0)
    assert torch.all(result.generated_token_ids < model.config.vocab_size)
    assert manager.allocator.allocated_count == 0


def test_eos_stops_and_releases_blocks() -> None:
    model = zero_model()
    manager = create_manager(model)
    prompt = torch.tensor([[1, 2]], dtype=torch.int64)

    result = generate_with_paged_cache(
        model,
        prompt,
        GenerationConfig(max_new_tokens=5, eos_token_id=0),
        manager,
        sequence_id="eos",
    )

    torch.testing.assert_close(
        result.token_ids,
        torch.tensor([[1, 2, 0]]),
    )
    assert result.num_generated_tokens == 1
    assert result.stop_reason == "eos"
    assert manager.active_sequence_ids == ()
    assert manager.allocator.free_count == manager.allocator.num_blocks


def test_zero_new_tokens_does_not_register_sequence() -> None:
    model = LlamaModel(tiny_config())
    manager = create_manager(model)
    prompt = torch.tensor([[1, 2]], dtype=torch.int64)

    result = generate_with_paged_cache(
        model,
        prompt,
        GenerationConfig(max_new_tokens=0),
        manager,
        sequence_id="zero",
    )

    torch.testing.assert_close(result.token_ids, prompt)
    assert result.generated_token_ids.shape == (1, 0)
    assert result.stop_reason == "max_new_tokens"
    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0


def test_seeded_sampling_matches_other_generation_paths() -> None:
    model = zero_model()
    manager = create_manager(model)
    prompt = torch.tensor([[1, 2]], dtype=torch.int64)
    config = GenerationConfig(
        max_new_tokens=6,
        do_sample=True,
        temperature=0.8,
        top_k=12,
        top_p=0.9,
        seed=84,
    )

    uncached = generate(model, prompt, config)
    contiguous = generate_with_cache(model, prompt, config)
    paged = generate_with_paged_cache(
        model,
        prompt,
        config,
        manager,
        sequence_id="sampling",
    )

    assert torch.equal(paged.token_ids, uncached.token_ids)
    assert torch.equal(paged.token_ids, contiguous.token_ids)
    assert manager.allocator.allocated_count == 0


def test_failure_releases_blocks_and_unregisters_sequence() -> None:
    torch.manual_seed(85)
    model = LlamaModel(tiny_config())
    model.eval()
    manager = create_manager(
        model,
        num_blocks=1,
        block_size=2,
    )

    with pytest.raises(BlockExhaustedError):
        generate_with_paged_cache(
            model,
            torch.tensor([[1, 2]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=3),
            manager,
            sequence_id="exhaustion",
        )

    assert manager.active_sequence_ids == ()
    assert manager.allocator.free_count == 1
    assert manager.allocator.allocated_count == 0


def test_generation_disables_gradients() -> None:
    torch.manual_seed(86)
    model = LlamaModel(tiny_config())
    model.eval()
    manager = create_manager(model)
    gradient_states: list[bool] = []

    def record_gradient_state(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        gradient_states.append(torch.is_grad_enabled())

    handle = model.embed_tokens.register_forward_hook(record_gradient_state)

    try:
        generate_with_paged_cache(
            model,
            torch.tensor([[1, 2]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=3),
            manager,
            sequence_id="gradients",
        )
    finally:
        handle.remove()

    assert gradient_states == [False, False, False]
    assert manager.active_sequence_ids == ()


def test_generation_rejects_result_beyond_model_limit() -> None:
    model = LlamaModel(tiny_config())
    manager = create_manager(model)
    prompt = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match="prompt and generated tokens exceed max_position_embeddings",
    ):
        generate_with_paged_cache(
            model,
            prompt,
            GenerationConfig(max_new_tokens=3),
            manager,
            sequence_id="too-long",
        )

    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0
