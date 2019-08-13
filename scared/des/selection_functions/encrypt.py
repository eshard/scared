from scared.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped
from scared.des import base as des
import numpy as _np


def _des_function(data, guesses, at_round, after_step):
    result = _np.empty((data.shape[0], len(guesses), data.shape[1]), dtype='uint8')
    data = data.astype('uint8')
    for guess in guesses:
        # expanded key with every byte to current key guess on 6-bit word
        current_expanded_key_guess = _np.bitwise_xor(_np.zeros((128), dtype=_np.uint8), guess)
        result[:, guess, :] = des.encrypt(data, current_expanded_key_guess, at_round=at_round, after_step=after_step)
    return result


def _add_round_key(data, guesses):
    return _des_function(data, guesses, 0, des.Steps.ADD_ROUND_KEY)


def _sboxes(data, guesses):
    return _des_function(data, guesses, 0, des.Steps.SBOXES)


def _first_round(data, guesses):
    return _des_function(data, guesses, 0, des.Steps.INV_PERMUTATION_P_RIGHT)


def _delta_last_rounds(data, guesses):
    return _des_function(data, guesses, 0, des.Steps.INV_PERMUTATION_P_DELTA_RIGHT)


def _first_key(key):
    return des.key_schedule(key)[0]


def _last_key(key):
    return des.key_schedule(key)[-1]


class FirstAddRoundKey:
    """Build an attack selection function which computes intermediate values after DES encrypt round key operation at first round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(64)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.

    """

    def __new__(cls, guesses=_np.arange(64, dtype='uint8'), words=None, plaintext_tag='plaintext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _add_round_key,
            expected_key_function=_first_key,
            words=words,
            guesses=guesses,
            target_tag=plaintext_tag,
            key_tag=key_tag)


class LastAddRoundKey:
    """Build an attack selection function which computes intermediate values after DES encrypt round key operation at last round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(64)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.

    """

    def __new__(cls, guesses=_np.arange(64, dtype='uint8'), words=None, ciphertext_tag='ciphertext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _add_round_key,
            expected_key_function=_last_key,
            words=words, guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag
        )


class FirstSboxes:
    """Build an attack selection function which computes intermediate values after DES encrypt S-box operation at first round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(64)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.

    """

    def __new__(cls, guesses=_np.arange(64, dtype='uint8'), words=None, plaintext_tag='plaintext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _sboxes,
            expected_key_function=_first_key,
            words=words,
            guesses=guesses,
            target_tag=plaintext_tag,
            key_tag=key_tag
        )


class LastSboxes:
    """Build an attack selection function which computes intermediate values after DES encrypt S-box operation at last round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(64)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.

    """

    def __new__(cls, guesses=_np.arange(64, dtype='uint8'), words=None, ciphertext_tag='ciphertext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _sboxes,
            expected_key_function=_last_key,
            words=words,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag
        )


class FeistelRFirstRounds:
    """Build an attack selection function which computes intermediate values after DES Feistel R, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(64)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.
    """

    def __new__(cls, guesses=_np.arange(64, dtype='uint8'), words=None, plaintext_tag='plaintext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _first_round,
            expected_key_function=_first_key,
            words=words,
            guesses=guesses,
            target_tag=plaintext_tag,
            key_tag=key_tag
        )


class FeistelRLastRounds:
    """Build an attack selection function which computes intermediate values after DES Feistel R, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(64)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.
    """

    def __new__(cls, guesses=_np.arange(64, dtype='uint8'), words=None, ciphertext_tag='ciphertext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _first_round,
            expected_key_function=_last_key,
            words=words,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag
        )


class DeltaRFirstRounds:
    """Build an attack selection function which computes delta intermediate values between DES encrypt first two rounds, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(64)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.
    """

    def __new__(cls, guesses=_np.arange(64, dtype='uint8'), words=None, plaintext_tag='plaintext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _delta_last_rounds,
            expected_key_function=_first_key,
            words=words,
            guesses=guesses,
            target_tag=plaintext_tag,
            key_tag=key_tag
        )


class DeltaRLastRounds:
    """Build an attack selection function which computes delta intermediate values between DES encrypt last two rounds, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(64)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.
    """

    def __new__(cls, guesses=_np.arange(64, dtype='uint8'), words=None, ciphertext_tag='ciphertext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _delta_last_rounds,
            expected_key_function=_last_key,
            words=words,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag
        )
