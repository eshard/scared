from ..context import scared
from scared import aes, traces
import numpy as np
import pytest


@pytest.mark.end_to_end
def test_nicv_on_dpa_v2():

    ths = traces.read_ths_from_ets_file('tests/end_to_end/dpa_v2_sub.ets')
    expected_key = aes.key_schedule(key=ths[0].key)[-1]
    sf = aes.selection_functions.encrypt.DeltaRLastRounds()
    container = scared.Container(ths[:15000])
    att = scared.NICVAttack(
        selection_function=sf,
        model=scared.HammingWeight(),
        discriminant=scared.maxabs,
    )
    att.run(container)
    last_key = np.argmax(att.scores, axis=0)
    assert np.array_equal(expected_key, last_key)
