from . import encrypt as _encrypt


class FirstAddRoundKey(_encrypt.LastAddRoundKey):
    """Build an attack selection function which computes intermediate values after DES decrypt round key operation at first round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.
    """


class LastAddRoundKey(_encrypt.FirstAddRoundKey):
    """Build an attack selection function which computes intermediate values after DES decrypt round key operation at last round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.
    """


class FirstSboxes(_encrypt.LastSboxes):
    """Build an attack selection function which computes intermediate values after DES decrypt sub bytes (S-box) operation at first round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.
    """


class LastSboxes(_encrypt.FirstSboxes):
    """Build an attack selection function which computes intermediate values after DES decrypt sub bytes (S-box) operation at last round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.
    """


class FeistelRFirstRounds(_encrypt.FeistelRLastRounds):
    """Build an attack selection function which computes intermediate values after DES decrypt Feistel R at first round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.
    """


class FeistelRLastRounds(_encrypt.FeistelRFirstRounds):
    """Build an attack selection function which computes intermediate values after DES decrypt Feistel R at last round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.
    """


class DeltaRFirstRounds(_encrypt.DeltaRLastRounds):
    """Build an attack selection function which computes delta intermediate values between DES decrypt first two rounds, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.
    """


class DeltaRLastRounds(_encrypt.DeltaRFirstRounds):
    """Build an attack selection function which computes delta intermediate values between DES decrypt last two rounds, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.
    """
