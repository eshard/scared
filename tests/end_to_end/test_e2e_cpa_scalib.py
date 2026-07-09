from ..context import scared
from scared import aes, traces
from scared.scalib import CPAAttackSCALib
import numpy as np
import pytest


@pytest.mark.end_to_end
@pytest.mark.scalib
def test_cpa_scalib_matches_builtin_cpa_ranking():

    ths = traces.read_ths_from_ets_file('tests/end_to_end/dpa_v2_sub.ets')
    sf = aes.selection_functions.encrypt.LastSubBytes()

    builtin = scared.CPAAttack(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs)
    builtin.run(scared.Container(ths))

    scalib = CPAAttackSCALib(selection_function=sf, model=scared.HammingWeight(), discriminant=scared.maxabs)
    scalib.run(scared.Container(ths))

    assert np.array_equal(scalib.scores.argmax(axis=0), builtin.scores.argmax(axis=0))
