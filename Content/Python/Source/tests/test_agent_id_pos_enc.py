import os
import sys
import torch
import pytest

# Ensure modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Source.Networks import CrossAttentionFeatureExtractor


def _build_extractor():
    return CrossAttentionFeatureExtractor(
        agent_obs_size=4,
        num_agents=2,
        embed_dim=8,
        num_heads=2,
        transformer_layers=1,
        dropout_rate=0.0,
        use_agent_id=True,
        ff_hidden_factor=2,
        id_num_freqs=4,
        conv_init_scale=1.0,
        linear_init_scale=1.0,
        central_processing_configs=[],
        environment_shape_config={},
    )


def test_agent_id_pos_enc_follows_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = _build_extractor()
    device = torch.device("cuda")
    model = model.to(device)
    assert model.agent_id_pos_enc.linear.weight.device == device
    assert model.agent_id_pos_enc.freq_scales.device == device
