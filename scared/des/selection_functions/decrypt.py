from . import encrypt

FirstAddRoundKey = encrypt.LastAddRoundKey
FirstAddRoundKey.__doc__ = """Build an attack selection function which computes intermediate values after DES decrypt round key operation at first round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.

    """

LastAddRoundKey = encrypt.FirstAddRoundKey
LastAddRoundKey.__doc__ = """Build an attack selection function which computes intermediate values after DES decrypt round key operation at last round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.

    """

FirstSboxes = encrypt.LastSboxes
FirstSboxes.__doc__ = """Build an attack selection function which computes intermediate values after DES decrypt sub bytes (S-box) operation at first round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.

    """

LastSboxes = encrypt.FirstSboxes
LastSboxes.__doc__ = """Build an attack selection function which computes intermediate values after DES decrypt sub bytes (S-box) operation at last round, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.

    """

FeistelRFirstRounds = encrypt.FeistelRLastRounds
FeistelRFirstRounds.__doc__ = """Build an attack selection function which computes intermediate values after DES Feistel R, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.
    """

FeistelRLastRounds = encrypt.FeistelRFirstRounds
FeistelRLastRounds.__doc__ = """Build an attack selection function which computes intermediate values after DES Feistel R, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.
    """

DeltaRFirstRounds = encrypt.DeltaRLastRounds
DeltaRFirstRounds.__doc__ = """Build an attack selection function which computes delta intermediate values between DES decrypt first two rounds, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        ciphertext_tag (str, default='ciphertext'): tag (key value) of the ciphertext metadata to use to retrieve ciphertext
            values from the metadata dict when selection function is called.
    """

DeltaRLastRounds = encrypt.DeltaRFirstRounds
DeltaRLastRounds.__doc__ = """Build an attack selection function which computes delta intermediate values between DES decrypt first two rounds, for guesses values.

    Args:
        guesses (numpy.array, default=numpy.arange(256)): default guesses value used for key hypothesis.
        words (:class:`scared.traces.Samples.SUPPORTED_INDICES_TYPES`, default=None): selection of key words computed.
        plaintext_tag (str, default='plaintext'): tag (key value) of the plaintext metadata to use to retrieve plaintext
            values from the metadata dict when selection function is called.
    """
