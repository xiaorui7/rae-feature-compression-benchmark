import pytest


@pytest.fixture(scope="session", autouse=True)
def limit_cpu_threads():
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield
