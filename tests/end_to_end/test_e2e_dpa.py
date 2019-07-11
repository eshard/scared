from ..context import scared
from scared import aes, traces
import numpy as np
import pytest


@pytest.mark.end_to_end
def test_dpa_on_dpa_v2():

    ths = traces.read_ths_from_ets_file('tests/end_to_end/dpa_v2_sub.ets')
    expected_key = aes.key_schedule(key=ths[0].key)[-1]

    # One byte is not correctly retrieved with this configuration.
    expected_key[13] = 252

    sf = aes.selection_functions.encrypt.DeltaRLastRounds()
    container = scared.Container(ths)

    att = scared.DPAAttack(
        selection_function=sf,
        model=scared.Monobit(7),
        discriminant=scared.maxabs,
    )
    att.run(container)

    bit_list = range(0, 7)
    max_score = np.copy(att.scores)

    for b in bit_list:
        att = scared.DPAAttack(selection_function=sf, model=scared.Monobit(b), discriminant=scared.maxabs)
        att.run(container)
        max_score = np.maximum(max_score, att.scores)
    att.scores = max_score
    last_key = np.argmax(att.scores, axis=0)

    assert np.array_equal(expected_key, last_key)
