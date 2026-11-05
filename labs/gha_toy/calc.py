"""Tiny module for the GitHub Actions toy lab (Task 6.6)."""


def add(a, b):
    return a + b


def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("b must not be 0")
    return a / b
