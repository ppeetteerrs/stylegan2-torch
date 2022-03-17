import stylegan2_torch
import pytest


@pytest.fixture
def version():
    return stylegan2_torch.__version__


def test_version(version: str):
    assert version is not None
