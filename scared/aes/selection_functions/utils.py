from scared.aes import base as aes
import numpy as _np


def add_round_key(data, guesses):
    """Compute XOR between input data and key guesses.

    This operation expands dimensions.

    Args:
        data (numpy.ndarray): input data array of shape (n_data, n_bytes).
            Will be cast to uint8.
        guesses (numpy.ndarray): vector of key guesses.

    Returns:
        (numpy.ndarray): result array of shape (n_data, n_guesses, n_bytes).

    """
    res = _np.empty((len(guesses), ) + data.shape, dtype='uint8')
    data = data.astype('uint8')
    for i, g in enumerate(guesses):
        res[i] = _np.bitwise_xor(data, g)
    return res.swapaxes(0, 1)


def sub_bytes(data, guesses):
    """Return AES state after SubBytes operation for each data and each guess.

    This operation expands dimensions.

    Args:
        data (numpy.ndarray): input data array of shape (n_data, n_bytes).
        guesses (numpy.ndarray): vector of key guesses.

    Returns:
        (numpy.ndarray): S-box output array of shape (n_data, n_guesses, n_bytes).

    """
    return aes.sub_bytes(add_round_key(data=data, guesses=guesses))


def inv_sub_bytes(data, guesses):
    """Return AES state after inverse SubBytes operation for each data and each guess.

    This operation expands dimensions.

    Args:
        data (numpy.ndarray): input data array of shape (n_data, n_bytes).
        guesses (numpy.ndarray): vector of key guesses.

    Returns:
        (numpy.ndarray): inverse S-box output array of shape (n_data, n_guesses, n_bytes).

    """
    return aes.inv_sub_bytes(add_round_key(data=data, guesses=guesses))


def delta_last_rounds(data, guesses):
    """Compute XOR between given data and inverse SubBytes results, for each data and each guess.

    This operation expands dimensions.

    Args:
        data (numpy.ndarray): input data array of shape (n_data, n_bytes).
        guesses (numpy.ndarray): vector of key guesses.

    Returns:
        (numpy.ndarray): delta values of shape (n_data, n_guesses, n_bytes).

    """
    data = data.astype('uint8')
    return _np.bitwise_xor(
        aes.shift_rows(data),
        aes.inv_sub_bytes(
            add_round_key(data=data, guesses=guesses)
        ).swapaxes(0, 1)
    ).swapaxes(0, 1)


def first_key(key):
    """Compute and return the first round key.

    Args:
        key (numpy.ndarray): AES master key array (16, 24, or 32 bytes for AES-128, 192, 256).

    Returns:
        (numpy.ndarray): first round key.

    """
    return aes.key_schedule(key)[0]


def last_key(key):
    """Compute and return the last round key.

    Args:
        key (numpy.ndarray): AES master key array (16, 24, or 32 bytes for AES-128, 192, 256).

    Returns:
        (numpy.ndarray): last round key.

    """
    return aes.key_schedule(key)[-1]
