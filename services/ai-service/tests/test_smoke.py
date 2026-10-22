from solia_ai import __version__
from solia_ai.config import load_config


def test_package_has_version():
    assert __version__


def test_packaged_config_has_expected_sections():
    cfg = load_config()
    for section in ("model", "training", "loss", "horizons", "labeling"):
        assert section in cfg
    assert cfg["horizons"] == ["30s", "1m", "5m", "15m", "1h"]
    assert cfg["model"]["seq_len"] > 0
    assert 0 <= cfg["loss"]["alpha"] <= 1
