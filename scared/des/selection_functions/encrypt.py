from scared.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped
import numpy as _np
from .utils import add_round_key as _add_round_key, sboxes as _sboxes
from .utils import first_round as _first_round, delta_last_rounds as _delta_last_rounds
from .utils import first_key as _first_key, last_key as _last_key


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
    """Build an attack selection function which computes intermediate values after DES encrypt Feistel R at first round, for guesses values.

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
    """Build an attack selection function which computes intermediate values after DES encrypt Feistel R at last round, for guesses values.

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
