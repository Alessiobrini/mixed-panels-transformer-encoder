import torch

from src.models.mixed_frequency_transformer import MixedFrequencyTransformer


def _dummy_inputs(batch_size=2, seq_len=5):
    value = torch.randn(batch_size, seq_len)
    var_id = torch.zeros(batch_size, seq_len, dtype=torch.long)
    freq_id = torch.zeros(batch_size, seq_len, dtype=torch.long)
    return value, var_id, freq_id


def test_default_positional_encoding_enabled():
    model = MixedFrequencyTransformer(freq_vocab_size=2, var_vocab_size=3, max_len=10)
    value, var_id, freq_id = _dummy_inputs()

    out = model(value=value, var_id=var_id, freq_id=freq_id)

    assert out.shape == (value.size(0),)
    assert model.positional_encoding_enabled is True
    assert model.positional_encoding is not None


def test_positional_encoding_disabled():
    model = MixedFrequencyTransformer(
        freq_vocab_size=2,
        var_vocab_size=3,
        max_len=10,
        use_positional_encoding=False,
    )
    value, var_id, freq_id = _dummy_inputs()

    out = model(value=value, var_id=var_id, freq_id=freq_id)

    assert out.shape == (value.size(0),)
    assert model.positional_encoding_enabled is False
    assert model.positional_encoding is None
