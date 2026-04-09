from scared.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped

import numpy as _np
from .utils import add_round_key as _add_round_key, sub_bytes as _sub_bytes
from .utils import inv_sub_bytes as _inv_sub_bytes, delta_last_rounds as _delta_last_rounds
from .utils import first_key as _first_key, last_key as _last_key


class FirstAddRoundKey:
    """Build an attack selection function which computes intermediate values after AES encrypt round key operation at first round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.
    """

    def __new__(cls, guesses=_np.arange(256, dtype='uint8'), words=None, plaintext_tag='plaintext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _add_round_key,
            expected_key_function=_first_key,
            words=words,
            guesses=guesses,
            target_tag=plaintext_tag,
            key_tag=key_tag)


class LastAddRoundKey:
    """Build an attack selection function which computes intermediate values after AES encrypt round key operation at last round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.

    """

    def __new__(cls, guesses=_np.arange(256, dtype='uint8'), words=None, ciphertext_tag='ciphertext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _add_round_key,
            expected_key_function=_last_key,
            words=words, guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag
        )


class FirstSubBytes:
    """Build an attack selection function which computes intermediate values after AES encrypt sub bytes (S-box) operation at first round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.

    """

    def __new__(cls, guesses=_np.arange(256, dtype='uint8'), words=None, plaintext_tag='plaintext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _sub_bytes,
            expected_key_function=_first_key,
            words=words,
            guesses=guesses,
            target_tag=plaintext_tag,
            key_tag=key_tag
        )


class LastSubBytes:
    """Build an attack selection function which computes intermediate values after AES encrypt sub bytes (S-box) operation at last round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.

    """

    def __new__(cls, guesses=_np.arange(256, dtype='uint8'), words=None, ciphertext_tag='ciphertext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _inv_sub_bytes,
            expected_key_function=_last_key,
            words=words,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag
        )


class DeltaRLastRounds:
    """Build an attack selection function which computes delta intermediate values between AES encrypt last two rounds, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.

    """

    def __new__(cls, guesses=_np.arange(256, dtype='uint8'), words=None, ciphertext_tag='ciphertext', key_tag='key'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _delta_last_rounds,
            expected_key_function=_last_key,
            words=words,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag
        )
