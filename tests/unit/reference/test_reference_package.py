import nanoserve.reference


def test_reference_package_is_importable() -> None:
    assert nanoserve.reference.__name__ == "nanoserve.reference"
