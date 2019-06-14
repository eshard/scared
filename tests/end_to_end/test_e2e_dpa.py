from ..context import scared
from scared import aes, traces
import numpy as np
import pytest


@pytest.mark.end_to_end
def test_dpa_on_dpa_v2():

    ths = traces.read_ths_from_ets_file('tests/end_to_end/dpa_v2_dpa_e2e.ets')
    expected_key = aes.key_schedule(key=ths[0].key)[-1]

    @scared.attack_selection_function
    def delta_last_two_rounds(ciphertext, guesses):
        res = np.empty((ciphertext.shape[0], len(guesses), ciphertext.shape[1]), dtype='uint8')
        for guess in guesses:
            s = aes.inv_sub_bytes(state=np.bitwise_xor(ciphertext, guess))
            res[:, guess, :] = np.bitwise_xor(aes.shift_rows(ciphertext), s)
        return res

    container = scared.Container(ths)

    att = scared.DPAAnalysis(
        selection_function=delta_last_two_rounds,
        model=scared.Monobit(7),
        discriminant=scared.maxabs,
    )
    att.run(container)

    bit_list = range(0, 7)
    max_score = np.copy(att.scores)

    for b in bit_list:
        att = scared.DPAAnalysis(selection_function=delta_last_two_rounds, model=scared.Monobit(b), discriminant=scared.maxabs)
        att.run(container)
        max_score = np.maximum(max_score, att.scores)
    att.scores = max_score
    last_key = np.argmax(att.scores, axis=0)

    assert np.array_equal(expected_key, last_key)
