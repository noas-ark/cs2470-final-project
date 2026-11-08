import pytest

from s1.models.backends import vllm_shim
from s1.registry import get_bool


def args(**settings):
    from pathlib import Path  # noqa: PLC0415

    return vllm_shim.serve_args(settings, Path("/models/openjev"), 8000)


def test_prefix_caching_is_on_by_default():
    a = args()
    assert "--enable-prefix-caching" in a and "--no-enable-prefix-caching" not in a


def test_prefix_caching_can_be_turned_off():
    a = args(prefix_caching=False)
    assert "--no-enable-prefix-caching" in a and "--enable-prefix-caching" not in a


def test_other_serving_flags_follow_the_settings():
    a = args(tensor_parallel_size=4, quantization="none", max_model_len=4096)
    assert a[a.index("--tensor-parallel-size") + 1] == "4"
    assert a[a.index("--max-model-len") + 1] == "4096"
    assert "--quantization" not in a
    assert "--quantization" in args(quantization="fp8")


def test_get_bool():
    assert get_bool({"x": False}, "x", True) is False
    assert get_bool({}, "x", True) is True
    with pytest.raises(TypeError, match="true or false"):
        get_bool({"x": 1}, "x", True)
