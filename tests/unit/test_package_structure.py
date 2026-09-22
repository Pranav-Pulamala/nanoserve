import tokserve.nn


def test_nn_package_is_importable() -> None:
    assert tokserve.nn.__name__ == "tokserve.nn"
