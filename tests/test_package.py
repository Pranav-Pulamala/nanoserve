import nanoserve


def test_package_version() -> None:
    assert nanoserve.__version__ == "0.1.0"