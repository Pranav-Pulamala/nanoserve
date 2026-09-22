import tokserve.reference


def test_reference_package_is_importable() -> None:
    assert tokserve.reference.__name__ == "tokserve.reference"
