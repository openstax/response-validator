from validator.utils import make_tristate


def test_make_tristate():
    assert make_tristate("foo") is True
    assert make_tristate("true") is True
    assert make_tristate("false") is False
    assert make_tristate("t") is True
    assert make_tristate("f") is False
    assert make_tristate("") is False
    assert make_tristate(1) is True
    assert make_tristate(42) is True

    assert make_tristate("foo", default=1) == 1
    assert make_tristate("true", default=1) == 1
    assert make_tristate("false", default=1) == 0
    assert make_tristate("t", default=1) == 1
    assert make_tristate("f", default=1) == 0
    assert make_tristate("", default=1) == 0
    assert make_tristate(1, default=1) == 1
    assert make_tristate(42, default=1) == 42

    assert make_tristate("auto") == "auto"
    assert make_tristate("auto", 1) == "auto"
