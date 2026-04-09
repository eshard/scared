from scared.des import base as des
import numpy as _np


def _des_function(data, guesses, at_round, after_step):
    result = _np.empty((len(guesses), ) + data.shape, dtype='uint8')
    data = data.astype('uint8')
    for i, guess in enumerate(guesses):
        # expanded key with every byte to current key guess on 6-bit word
        current_expanded_key_guess = _np.bitwise_xor(_np.zeros((128), dtype=_np.uint8), guess)
        result[i] = des.encrypt(data, current_expanded_key_guess, at_round=at_round, after_step=after_step)
    return result.swapaxes(0, 1)


def add_round_key(data, guesses):
    """Return DES state after xor operation for each data and each guess.

    This operation expands dimensions.

    Args:
        data (numpy.ndarray): input data array of shape (n_data, n_bytes).
            Will be cast to uint8.
        guesses (numpy.ndarray): vector of key guesses.

    Returns:
        (numpy.ndarray): result array of shape (n_data, n_guesses, n_bytes).

    """
    return _des_function(data, guesses, 0, des.Steps.ADD_ROUND_KEY)


def sboxes(data, guesses):
    """Return DES state after Sbox operation for each data and each guess.

    This operation expands dimensions.

    Args:
        data (numpy.ndarray): input data array of shape (n_data, n_bytes).
        guesses (numpy.ndarray): vector of key guesses.

    Returns:
        (numpy.ndarray): S-box output array of shape (n_data, n_guesses, n_bytes).

    """
    return _des_function(data, guesses, 0, des.Steps.SBOXES)


def first_round(data, guesses):
    """Return DES state after first round for each data and each guess.

    This operation expands dimensions.

    Args:
        data (numpy.ndarray): input data array of shape (n_data, n_bytes).
        guesses (numpy.ndarray): vector of key guesses.

    Returns:
        (numpy.ndarray): S-box output array of shape (n_data, n_guesses, n_bytes).

    """
    return _des_function(data, guesses, 0, des.Steps.INV_PERMUTATION_P_RIGHT)


def delta_last_rounds(data, guesses):
    """Return delta of DES states R_i and R_(i+1) for each data and each guess.

    This operation expands dimensions.

    Args:
        data (numpy.ndarray): input data array of shape (n_data, n_bytes).
        guesses (numpy.ndarray): vector of key guesses.

    Returns:
        (numpy.ndarray): S-box output array of shape (n_data, n_guesses, n_bytes).

    """
    return _des_function(data, guesses, 0, des.Steps.INV_PERMUTATION_P_DELTA_RIGHT)


def first_key(key):
    """Compute and return the first round key.

    Args:
        key (numpy.ndarray): DES master key array.

    Returns:
        (numpy.ndarray): first round key.

    """
    return des.key_schedule(key)[0]


def last_key(key):
    """Compute and return the last round key.

    Args:
        key (numpy.ndarray): DES master key array.

    Returns:
        (numpy.ndarray): last round key.

    """
    return des.key_schedule(key)[-1]
