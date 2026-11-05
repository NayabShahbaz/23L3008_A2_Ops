import pytest

from calc import add, divide


def test_add():
    assert add(2, 3) == 5   # change to 6 to get a RED run, then revert for GREEN


def test_divide():
    assert divide(6, 3) == 2
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
