import pytest
import stylegan2_torch


@pytest.fixture
def version():
    g = stylegan2_torch.Generator(128)
    return stylegan2_torch.__version__


def test_version(version: str):
    assert version is not None
