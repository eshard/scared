from .context import scared  # noqa: F401
from scared import traces
import numpy as np


def test_read_from_ram():
    ths = traces.formats.read_ths_from_ram(samples=np.random.randint(0, 255, (12, 100), dtype='uint8'))
    assert len(ths) == 12
