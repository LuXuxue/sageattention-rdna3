import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="native",
        choices=["native", "triton"],
        help="sageattention backend to test (native or triton)",
    )


@pytest.fixture(scope="session", autouse=True)
def _selected_backend(request):
    """Explicitly set the sageattention backend BEFORE the first sageattention import.

    core.py reads SAGEATTN_BACKEND at module import time; this fixture sets it
    before the sageattn fixture imports sageattention, eliminating ambiguity.
    """
    backend = request.config.getoption("--backend")
    os.environ["SAGEATTN_BACKEND"] = backend
    return backend


@pytest.fixture
def backend(_selected_backend):
    return _selected_backend