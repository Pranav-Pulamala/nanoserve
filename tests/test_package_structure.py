import nanoserve.nn


def test_nn_package_is_importable() -> None:
    assert nanoserve.nn.__name__ == "nanoserve.nn"