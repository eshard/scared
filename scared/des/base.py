"""DES module provides DES cipher primitives to encrypt, decrypt and compute any intermediate states.

Ready-to-use selection functions compatible with side-channel analysis framework are provided.
"""
import enum
import typing as _typ

import numpy as _np

from .._utils import _is_bytes_array


SBOXES = _np.array([
    [0xE, 0x0, 0x4, 0xF, 0xD, 0x7, 0x1, 0x4, 0x2, 0xE, 0xF, 0x2, 0xB, 0xD, 0x8, 0x1, 0x3, 0xA, 0xA, 0x6, 0x6,
        0xC, 0xC, 0xB, 0x5, 0x9, 0x9, 0x5, 0x0, 0x3, 0x7, 0x8, 0x4, 0xF, 0x1, 0xC, 0xE, 0x8, 0x8, 0x2, 0xD, 0x4,
        0x6, 0x9, 0x2, 0x1, 0xB, 0x7, 0xF, 0x5, 0xC, 0xB, 0x9, 0x3, 0x7, 0xE, 0x3, 0xA, 0xA, 0x0, 0x5, 0x6, 0x0,
        0xD],
    [0xF, 0x3, 0x1, 0xD, 0x8, 0x4, 0xE, 0x7, 0x6, 0xF, 0xB, 0x2, 0x3, 0x8, 0x4, 0xE, 0x9, 0xC, 0x7, 0x0, 0x2,
        0x1, 0xD, 0xA, 0xC, 0x6, 0x0, 0x9, 0x5, 0xB, 0xA, 0x5, 0x0, 0xD, 0xE, 0x8, 0x7, 0xA, 0xB, 0x1, 0xA, 0x3,
        0x4, 0xF, 0xD, 0x4, 0x1, 0x2, 0x5, 0xB, 0x8, 0x6, 0xC, 0x7, 0x6, 0xC, 0x9, 0x0, 0x3, 0x5, 0x2, 0xE, 0xF,
        0x9],
    [0xA, 0xD, 0x0, 0x7, 0x9, 0x0, 0xE, 0x9, 0x6, 0x3, 0x3, 0x4, 0xF, 0x6, 0x5, 0xA, 0x1, 0x2, 0xD, 0x8, 0xC,
        0x5, 0x7, 0xE, 0xB, 0xC, 0x4, 0xB, 0x2, 0xF, 0x8, 0x1, 0xD, 0x1, 0x6, 0xA, 0x4, 0xD, 0x9, 0x0, 0x8, 0x6,
        0xF, 0x9, 0x3, 0x8, 0x0, 0x7, 0xB, 0x4, 0x1, 0xF, 0x2, 0xE, 0xC, 0x3, 0x5, 0xB, 0xA, 0x5, 0xE, 0x2, 0x7,
        0xC],
    [0x7, 0xD, 0xD, 0x8, 0xE, 0xB, 0x3, 0x5, 0x0, 0x6, 0x6, 0xF, 0x9, 0x0, 0xA, 0x3, 0x1, 0x4, 0x2, 0x7, 0x8,
        0x2, 0x5, 0xC, 0xB, 0x1, 0xC, 0xA, 0x4, 0xE, 0xF, 0x9, 0xA, 0x3, 0x6, 0xF, 0x9, 0x0, 0x0, 0x6, 0xC, 0xA,
        0xB, 0x1, 0x7, 0xD, 0xD, 0x8, 0xF, 0x9, 0x1, 0x4, 0x3, 0x5, 0xE, 0xB, 0x5, 0xC, 0x2, 0x7, 0x8, 0x2, 0x4,
        0xE],
    [0x2, 0xE, 0xC, 0xB, 0x4, 0x2, 0x1, 0xC, 0x7, 0x4, 0xA, 0x7, 0xB, 0xD, 0x6, 0x1, 0x8, 0x5, 0x5, 0x0, 0x3,
        0xF, 0xF, 0xA, 0xD, 0x3, 0x0, 0x9, 0xE, 0x8, 0x9, 0x6, 0x4, 0xB, 0x2, 0x8, 0x1, 0xC, 0xB, 0x7, 0xA, 0x1,
        0xD, 0xE, 0x7, 0x2, 0x8, 0xD, 0xF, 0x6, 0x9, 0xF, 0xC, 0x0, 0x5, 0x9, 0x6, 0xA, 0x3, 0x4, 0x0, 0x5, 0xE,
        0x3],
    [0xC, 0xA, 0x1, 0xF, 0xA, 0x4, 0xF, 0x2, 0x9, 0x7, 0x2, 0xC, 0x6, 0x9, 0x8, 0x5, 0x0, 0x6, 0xD, 0x1, 0x3,
        0xD, 0x4, 0xE, 0xE, 0x0, 0x7, 0xB, 0x5, 0x3, 0xB, 0x8, 0x9, 0x4, 0xE, 0x3, 0xF, 0x2, 0x5, 0xC, 0x2, 0x9,
        0x8, 0x5, 0xC, 0xF, 0x3, 0xA, 0x7, 0xB, 0x0, 0xE, 0x4, 0x1, 0xA, 0x7, 0x1, 0x6, 0xD, 0x0, 0xB, 0x8, 0x6,
        0xD],
    [0x4, 0xD, 0xB, 0x0, 0x2, 0xB, 0xE, 0x7, 0xF, 0x4, 0x0, 0x9, 0x8, 0x1, 0xD, 0xA, 0x3, 0xE, 0xC, 0x3, 0x9,
        0x5, 0x7, 0xC, 0x5, 0x2, 0xA, 0xF, 0x6, 0x8, 0x1, 0x6, 0x1, 0x6, 0x4, 0xB, 0xB, 0xD, 0xD, 0x8, 0xC, 0x1,
        0x3, 0x4, 0x7, 0xA, 0xE, 0x7, 0xA, 0x9, 0xF, 0x5, 0x6, 0x0, 0x8, 0xF, 0x0, 0xE, 0x5, 0x2, 0x9, 0x3, 0x2,
        0xC],
    [0xD, 0x1, 0x2, 0xF, 0x8, 0xD, 0x4, 0x8, 0x6, 0xA, 0xF, 0x3, 0xB, 0x7, 0x1, 0x4, 0xA, 0xC, 0x9, 0x5, 0x3,
        0x6, 0xE, 0xB, 0x5, 0x0, 0x0, 0xE, 0xC, 0x9, 0x7, 0x2, 0x7, 0x2, 0xB, 0x1, 0x4, 0xE, 0x1, 0x7, 0x9, 0x4,
        0xC, 0xA, 0xE, 0x8, 0x2, 0xD, 0x0, 0xF, 0x6, 0xC, 0xA, 0x9, 0xD, 0x0, 0xF, 0x3, 0x3, 0x5, 0x5, 0x6, 0x8,
        0xB]
], dtype=_np.uint8)

ROUND_KEY_BITS_INDEXES = _np.array([
    # Round key 0
    [
        [9, 50, 33, 59, 48, 16],
        [32, 56, 1, 8, 18, 41],
        [2, 34, 25, 24, 43, 57],
        [58, 0, 35, 26, 17, 40],
        [21, 27, 38, 53, 36, 3],
        [46, 29, 4, 52, 22, 28],
        [60, 20, 37, 62, 14, 19],
        [44, 13, 12, 61, 54, 30]

    ],
    # Round key 1
    [
        [1, 42, 25, 51, 40, 8],
        [24, 48, 58, 0, 10, 33],
        [59, 26, 17, 16, 35, 49],
        [50, 57, 56, 18, 9, 32],
        [13, 19, 30, 45, 28, 62],
        [38, 21, 27, 44, 14, 20],
        [52, 12, 29, 54, 6, 11],
        [36, 5, 4, 53, 46, 22]

    ],
    # Round key 2
    [
        [50, 26, 9, 35, 24, 57],
        [8, 32, 42, 49, 59, 17],
        [43, 10, 1, 0, 48, 33],
        [34, 41, 40, 2, 58, 16],
        [60, 3, 14, 29, 12, 46],
        [22, 5, 11, 28, 61, 4],
        [36, 27, 13, 38, 53, 62],
        [20, 52, 19, 37, 30, 6]

    ],
    # Round key 3
    [
        [34, 10, 58, 48, 8, 41],
        [57, 16, 26, 33, 43, 1],
        [56, 59, 50, 49, 32, 17],
        [18, 25, 24, 51, 42, 0],
        [44, 54, 61, 13, 27, 30],
        [6, 52, 62, 12, 45, 19],
        [20, 11, 60, 22, 37, 46],
        [4, 36, 3, 21, 14, 53]

    ],
    # Round key 4
    [
        [18, 59, 42, 32, 57, 25],
        [41, 0, 10, 17, 56, 50],
        [40, 43, 34, 33, 16, 1],
        [2, 9, 8, 35, 26, 49],
        [28, 38, 45, 60, 11, 14],
        [53, 36, 46, 27, 29, 3],
        [4, 62, 44, 6, 21, 30],
        [19, 20, 54, 5, 61, 37]

    ],
    # Round key 5
    [
        [2, 43, 26, 16, 41, 9],
        [25, 49, 59, 1, 40, 34],
        [24, 56, 18, 17, 0, 50],
        [51, 58, 57, 48, 10, 33],
        [12, 22, 29, 44, 62, 61],
        [37, 20, 30, 11, 13, 54],
        [19, 46, 28, 53, 5, 14],
        [3, 4, 38, 52, 45, 21]
    ],
    # Round key 6
    [
        [51, 56, 10, 0, 25, 58],
        [9, 33, 43, 50, 24, 18],
        [8, 40, 2, 1, 49, 34],
        [35, 42, 41, 32, 59, 17],
        [27, 6, 13, 28, 46, 45],
        [21, 4, 14, 62, 60, 38],
        [3, 30, 12, 37, 52, 61],
        [54, 19, 22, 36, 29, 5]
    ],
    # Round key 7
    [
        [35, 40, 59, 49, 9, 42],
        [58, 17, 56, 34, 8, 2],
        [57, 24, 51, 50, 33, 18],
        [48, 26, 25, 16, 43, 1],
        [11, 53, 60, 12, 30, 29],
        [5, 19, 61, 46, 44, 22],
        [54, 14, 27, 21, 36, 45],
        [38, 3, 6, 20, 13, 52]
    ],
    # Round key 8
    [
        [56, 32, 51, 41, 1, 34],
        [50, 9, 48, 26, 0, 59],
        [49, 16, 43, 42, 25, 10],
        [40, 18, 17, 8, 35, 58],
        [3, 45, 52, 4, 22, 21],
        [60, 11, 53, 38, 36, 14],
        [46, 6, 19, 13, 28, 37],
        [30, 62, 61, 12, 5, 44]
    ],
    # Round key 9
    [
        [40, 16, 35, 25, 50, 18],
        [34, 58, 32, 10, 49, 43],
        [33, 0, 56, 26, 9, 59],
        [24, 2, 1, 57, 48, 42],
        [54, 29, 36, 19, 6, 5],
        [44, 62, 37, 22, 20, 61],
        [30, 53, 3, 60, 12, 21],
        [14, 46, 45, 27, 52, 28]
    ],
    # Round key 10
    [
        [24, 0, 48, 9, 34, 2],
        [18, 42, 16, 59, 33, 56],
        [17, 49, 40, 10, 58, 43],
        [8, 51, 50, 41, 32, 26],
        [38, 13, 20, 3, 53, 52],
        [28, 46, 21, 6, 4, 45],
        [14, 37, 54, 44, 27, 5],
        [61, 30, 29, 11, 36, 12]
    ],
    # Round key 11
    [
        [8, 49, 32, 58, 18, 51],
        [2, 26, 0, 43, 17, 40],
        [1, 33, 24, 59, 42, 56],
        [57, 35, 34, 25, 16, 10],
        [22, 60, 4, 54, 37, 36],
        [12, 30, 5, 53, 19, 29],
        [61, 21, 38, 28, 11, 52],
        [45, 14, 13, 62, 20, 27]
    ],
    # Round key 12
    [
        [57, 33, 16, 42, 2, 35],
        [51, 10, 49, 56, 1, 24],
        [50, 17, 8, 43, 26, 40],
        [41, 48, 18, 9, 0, 59],
        [6, 44, 19, 38, 21, 20],
        [27, 14, 52, 37, 3, 13],
        [45, 5, 22, 12, 62, 36],
        [29, 61, 60, 46, 4, 11]
    ],
    # Round key 13
    [
        [41, 17, 0, 26, 51, 48],
        [35, 59, 33, 40, 50, 8],
        [34, 1, 57, 56, 10, 24],
        [25, 32, 2, 58, 49, 43],
        [53, 28, 3, 22, 5, 4],
        [11, 61, 36, 21, 54, 60],
        [29, 52, 6, 27, 46, 20],
        [13, 45, 44, 30, 19, 62]
    ],
    # Round key 14
    [
        [25, 1, 49, 10, 35, 32],
        [48, 43, 17, 24, 34, 57],
        [18, 50, 41, 40, 59, 8],
        [9, 16, 51, 42, 33, 56],
        [37, 12, 54, 6, 52, 19],
        [62, 45, 20, 5, 38, 44],
        [13, 36, 53, 11, 30, 4],
        [60, 29, 28, 14, 3, 46]
    ],
    # Round key 15
    [
        [17, 58, 41, 2, 56, 24],
        [40, 35, 9, 16, 26, 49],
        [10, 42, 33, 32, 51, 0],
        [1, 8, 43, 34, 25, 48],
        [29, 4, 46, 61, 44, 11],
        [54, 37, 12, 60, 30, 36],
        [5, 28, 45, 3, 22, 27],
        [52, 21, 20, 6, 62, 38]
    ]
], dtype=_np.uint8)

ROUND_KEY_MISSING_BITS_INDEXES = _np.array([
    [5, 6, 10, 11, 42, 45, 49, 51],
    [2, 3, 34, 37, 41, 43, 60, 61],
    [18, 21, 25, 44, 45, 51, 54, 56],
    [2, 5, 9, 28, 29, 35, 38, 40],
    [12, 13, 22, 24, 48, 51, 52, 58],
    [6, 8, 27, 32, 35, 36, 42, 60],
    [11, 16, 20, 26, 44, 48, 53, 57],
    [0, 4, 10, 28, 32, 37, 41, 62],
    [2, 20, 24, 27, 29, 33, 54, 57],
    [4, 8, 11, 13, 17, 38, 41, 51],
    [1, 19, 22, 25, 35, 57, 60, 62],
    [3, 6, 9, 41, 44, 46, 48, 50],
    [25, 28, 30, 32, 34, 53, 54, 58],
    [9, 12, 14, 16, 18, 37, 38, 42],
    [0, 2, 21, 22, 26, 27, 58, 61],
    [13, 14, 18, 19, 50, 53, 57, 59]
], dtype=_np.uint8)


PC1 = [57, 49, 41, 33, 25, 17, 9,
       1, 58, 50, 42, 34, 26, 18,
       10, 2, 59, 51, 43, 35, 27,
       19, 11, 3, 60, 52, 44, 36,
       63, 55, 47, 39, 31, 23, 15,
       7, 62, 54, 46, 38, 30, 22,
       14, 6, 61, 53, 45, 37, 29,
       21, 13, 5, 28, 20, 12, 4]


PC2 = [14, 17, 11, 24, 1, 5,
       3, 28, 15, 6, 21, 10,
       23, 19, 12, 4, 26, 8,
       16, 7, 27, 20, 13, 2,
       41, 52, 31, 37, 47, 55,
       30, 40, 51, 45, 33, 48,
       44, 49, 39, 56, 34, 53,
       46, 42, 50, 36, 29, 32]


DES_ROUNDS = 16


class Steps(enum.IntEnum):
    """Enumeration for the DES round steps."""

    INITIAL_PERMUTATION = 0  # Resulting in LR (Left-Right)
    EXPANSIVE_PERMUTATION = 1  # Expansion on R part
    ADD_ROUND_KEY = 2
    SBOXES = 3
    PERMUTATION_P = 4
    XOR_WITH_SAVED_LEFT_RIGHT = 5  # Thus we obtain new Right-Left
    PERMUTE_RIGHT_LEFT = 6  # Thus we obtain Left-Right for next round
    INV_PERMUTATION_P_RIGHT = 7  # Inverted permutation P of Ri
    INV_PERMUTATION_P_DELTA_RIGHT = 8  # Inverted permutation P of (Ri xor Ri-1)
    FINAL_PERMUTATION = 9  # The final cipher


def _is_bytes_of_len(state, length=[8]):
    _is_bytes_array(state)
    if state.shape[-1] not in length:
        raise ValueError(f'state last dimension should be in {length}, not {state.shape[-1]}.')
    return True


def key_schedule(key, interrupt_after_round=15):
    """Compute DES key schedule.

    Args:
        key (numpy.ndarray): numpy byte array (dtype uint8), with last dimension 8 bytes long. The key to use as input.

    Returns:
        (numpy.ndarray): numpy byte array containing all round keys except those after interrupt_after_round, with shape (number of keys, number of rounds, 8),
            or (number of rounds, 8) if only one key has been provided. All the generated round keys are 6bits words.

    Examples:
        import numpy as np
        key = np.array([0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6, 0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C], dtype=np.uint8)
        schedule = key_schedule(key)

    Raises:
        TypeError: For key not being a numpy array, or for interrupt_after_round not being an int.
        ValueError: For key not being uint8 type, or containing keys longer than 8 bytes. Or for interrupt_after_round not being between 0 and 15.

    """
    _is_bytes_of_len(key)
    if not isinstance(interrupt_after_round, int):
        raise TypeError(f"Wrong argument type for interrupt_after_round, got {type(interrupt_after_round)} instead of int.")
    if interrupt_after_round < 0 or interrupt_after_round > 15:
        raise ValueError(f"Wrong DES key round number, must be between 0 and 15 and not {interrupt_after_round}.")

    dimensions = key.shape[:-1]
    key = key.reshape((-1, key.shape[-1]))
    # we allocate the table that will contain splitted key bits
    key_bits = _np.empty((key.shape[0], 64), dtype=_np.uint8)
    # split the key into bits
    for current_key_byte in _np.arange(8):
        key_bits[:, current_key_byte * 8 + 0] = ((key[:, current_key_byte] & 0x80) != 0x00)
        key_bits[:, current_key_byte * 8 + 1] = ((key[:, current_key_byte] & 0x40) != 0x00)
        key_bits[:, current_key_byte * 8 + 2] = ((key[:, current_key_byte] & 0x20) != 0x00)
        key_bits[:, current_key_byte * 8 + 3] = ((key[:, current_key_byte] & 0x10) != 0x00)
        key_bits[:, current_key_byte * 8 + 4] = ((key[:, current_key_byte] & 0x08) != 0x00)
        key_bits[:, current_key_byte * 8 + 5] = ((key[:, current_key_byte] & 0x04) != 0x00)
        key_bits[:, current_key_byte * 8 + 6] = ((key[:, current_key_byte] & 0x02) != 0x00)
        key_bits[:, current_key_byte * 8 + 7] = ((key[:, current_key_byte] & 0x01) != 0x00)
    # we allocate the output table at the expected number of bytes
    output_key = _np.zeros((key.shape[0], 8 * (interrupt_after_round + 1)), dtype=_np.uint8)
    for current_round in _np.arange(16):
        for current_word in _np.arange(8):
            output_key[:, current_round * 8 + current_word] = \
                0x20 * key_bits[:, ROUND_KEY_BITS_INDEXES[current_round][current_word][0]] + \
                0x10 * key_bits[:, ROUND_KEY_BITS_INDEXES[current_round][current_word][1]] + \
                0x08 * key_bits[:, ROUND_KEY_BITS_INDEXES[current_round][current_word][2]] + \
                0x04 * key_bits[:, ROUND_KEY_BITS_INDEXES[current_round][current_word][3]] + \
                0x02 * key_bits[:, ROUND_KEY_BITS_INDEXES[current_round][current_word][4]] + \
                0x01 * key_bits[:, ROUND_KEY_BITS_INDEXES[current_round][current_word][5]]
        if current_round == interrupt_after_round:
            break
    if dimensions:
        final_shape = (-1, (interrupt_after_round + 1), 8)
    else:
        final_shape = ((interrupt_after_round + 1), 8)
    return output_key.reshape(final_shape)


def get_master_key(round_key, nb_round, plaintext, expected_ciphertext):
    """Retrieve the DES master key from a key schedule round key.

    Compute the 256 possible master keys from a round key. Then, from the 256 possible masters keys, compute a DES with the provided plaintext,
    and check if the expected_ciphertext is found. If so, the correct key is discovered and will be returned.

    Args:
        round_key (numpy.ndarray): the round key to analyze. 8x6bit word.
        nb_round (int): the number of the round corresponding to the round key.
        plaintext (numpy.ndarray): a 8 byte input plaintext.
        expected_ciphertext (numpy.ndarray): a 8 byte output message computed from the key to discover.

    Returns:
        (numpy.ndarray) the found master key, or None if not found.

    """
    _is_bytes_of_len(round_key, length=[8])
    _is_bytes_of_len(plaintext, length=[8])
    _is_bytes_of_len(expected_ciphertext, length=[8])
    if not (round_key < 64).all():
        raise ValueError(f'round_key should be a 8x6bit array, but it contains at least one value coded on more than 6 bits (> 63).')
    if not isinstance(nb_round, int):
        raise TypeError(f'nb_round must be an integer value, not a value of type {type(nb_round)}.')
    if nb_round < 0 or nb_round > 15:
        raise ValueError(f'nb_round must be between 0 and 15, not {nb_round}.')

    guess_keys = _find_possible_keys(round_key, nb_round)
    for guess in guess_keys:
        computed_ciphertext = encrypt(plaintext, guess)
        if _np.array_equal(expected_ciphertext, computed_ciphertext):
            return guess
    return None


def _find_possible_keys(round_key, nb_round):
    # This is ci_di before performing PC-2
    # 255 value means the bit value is unknown
    ci_di = _np.array([0, 0, 0, 0, 0, 0,
                       0, 0, 255, 0, 0, 0,
                       0, 0, 0, 0, 0, 255,
                       0, 0, 0, 255, 0, 0, 255,
                       0, 0, 0, 0, 0, 0, 0,
                       0, 0, 255, 0, 0, 255,
                       0, 0, 0, 0, 255, 0,
                       0, 0, 0, 0, 0, 0,
                       0, 0, 0, 255, 0, 0], dtype='uint8')
    # We remove PC-2
    for index in range(48):
        word = int(index / 6)
        bit = 1 << (5 - index % 6)
        if round_key[word] & bit != 0:
            ci_di[PC2[index] - 1] = 1
    # We split in Ci | Di
    ci_di = ci_di.reshape(2, 28)
    # We remove the left shits according to the round number
    nb_shift = [1, 2, 4, 6, 8, 10, 12, 14, 15, 17, 19, 21, 23, 25, 27, 28]
    ci_di = _np.roll(ci_di, + nb_shift[nb_round], 1)
    # We come back as a 56 bit array before PC-1
    ci_di = ci_di.reshape(56)
    # We remove PC-1
    master_key = [0] * 64
    for index in range(len(PC1)):
        master_key[PC1[index] - 1] = ci_di[index]
    # We can now compute the 256 possible keys, and return them in an array
    guessed_keys = _convert_hypothesis_bits_into_keys(master_key)
    return _np.array([[(guess >> (8 * hit)) & 0xFF for hit in range(7, -1, -1)] for guess in guessed_keys], dtype='uint8')


def _convert_hypothesis_bits_into_keys(array):
    len_array = len(array)
    if len_array == 1:
        return [array[0]]
    else:
        bit = array[0]
        keys_for_value_0 = _convert_hypothesis_bits_into_keys(array[1:])
        if not bit:
            return keys_for_value_0
        keys_for_value_1 = [hit + (1 << (len_array - 1)) for hit in keys_for_value_0]
        if bit == 255:
            # If the bit is unknown, we have to return both the values for bit 0 and 1
            return keys_for_value_1 + keys_for_value_0
        else:
            return keys_for_value_1


def initial_permutation(state):
    """Compute DES initial permutation (IP) operation resulting on a 8 bytes words state.

    This operation outputs the L0R0 value.

    Args:
        state (numpy.ndarray): a uint8 array of 8 bytes words as last dimension.

    Returns:
        (numpy.ndarray) DES initial permutation result with same dimensions as input state.

    """
    _is_bytes_of_len(state)
    dimensions = state.shape
    data = state.reshape((-1, 8))
    out = _np.zeros((data.shape[0], 8), dtype=_np.uint8)
    for current_byte in _np.arange(8):
        out[:, 0] <<= 1
        out[:, 0] += data[:, 7 - current_byte] >> 6 & 0x01
        out[:, 1] <<= 1
        out[:, 1] += data[:, 7 - current_byte] >> 4 & 0x01
        out[:, 2] <<= 1
        out[:, 2] += data[:, 7 - current_byte] >> 2 & 0x01
        out[:, 3] <<= 1
        out[:, 3] += data[:, 7 - current_byte] & 0x01
        out[:, 4] <<= 1
        out[:, 4] += data[:, 7 - current_byte] >> 7 & 0x01
        out[:, 5] <<= 1
        out[:, 5] += data[:, 7 - current_byte] >> 5 & 0x01
        out[:, 6] <<= 1
        out[:, 6] += data[:, 7 - current_byte] >> 3 & 0x01
        out[:, 7] <<= 1
        out[:, 7] += data[:, 7 - current_byte] >> 1 & 0x01
    return out.reshape(dimensions)


def expansive_permutation(state):
    """Compute DES expansive permutation (EP) operation resulting on a 8x6bit words state.

    Args:
        state (numpy.ndarray): a uint8 array of 4 bytes words as last dimension.

    Returns:
        (numpy.ndarray) DES expansive permutation result with 8x6bit dimension words.

    """
    _is_bytes_of_len(state, length=[4])
    dimensions = state.shape
    data = state.reshape((-1, 4))
    out = _np.zeros((data.shape[0], 8), dtype=_np.uint8)
    # fill output data with specifically picked bits
    for current_2byte_block in _np.arange(4):
        out[:, 2 * current_2byte_block] = (data[:, (current_2byte_block + 3) % 4] & 0x01) << 5
        out[:, 2 * current_2byte_block] += (data[:, current_2byte_block] & 0xF8) >> 3
        out[:, 2 * current_2byte_block + 1] = (data[:, current_2byte_block] & 0x1F) << 1
        out[:, 2 * current_2byte_block + 1] += (data[:, (current_2byte_block + 1) % 4] & 0x80) >> 7
    return out.reshape(dimensions[:-1] + (8,))


def add_round_key(state, keys):
    """Compute DES xor operation between a words state and a round keys array, both 8x6bit.

    Depending on the shapes of state and keys, the result can be:

        - one state added to one key if state is (8,) and keys is (8,)
        - one state added to n keys if state is (8,) and keys is (n, 8)
        - n states added to 1 key if state is (n, 8) and keys is (8)
        - states added to keys, combined by pairs if state is (n, 8) and keys is (n, 8).

    In every other case, a ValueError will be raised.

    Args:
        state (numpy.ndarray): a uint8 array of 8x6bit words as last dimension.
        keys (numpy.ndarray): a uint8 array of 8x6bit round keys as last dimension.

    Returns:
        (numpy.ndarray) xor result between state and keys.

    """
    _is_bytes_of_len(state)
    _is_bytes_of_len(keys)
    return _np.bitwise_xor(state, keys)


def sboxes(state):
    """Compute DES SBOXes operation resulting on a 8x4bits words state.

    Args:
        state (numpy.ndarray): a uint8 array of 8x6bits words as last dimension.

    Returns:
        (numpy.ndarray) DES SBOXes operation result with 8x4bits words.

    """
    _is_bytes_of_len(state)
    dimensions = state.shape
    data = state.reshape((-1, 8))
    out = _np.empty((data.shape[0], 8), dtype=_np.uint8)
    # each 6-bit word is passed through the corresponding SBOX
    for current_word in _np.arange(8):
        out[:, current_word] = SBOXES[current_word][data[:, current_word]]
    return out.reshape(dimensions)


def permutation_p(state):
    """Compute DES permutation P (PP) operation resulting on a 4 bytes words state.

    Args:
        state (numpy.ndarray): a uint8 array of 8x4bits words as last dimension.

    Returns:
        (numpy.ndarray) DES permutation P result with 4 bytes dimension words.

    """
    _is_bytes_of_len(state)
    dimensions = state.shape
    data = state.reshape((-1, 8))
    out = _np.empty((data.shape[0], 4), dtype=_np.uint8)

    # fill output data with specifically picked bits
    out[:, 0] = (data[:, 3] & 0x01) << 7
    out[:, 0] += (data[:, 1] & 0x02) << 5
    out[:, 0] += (data[:, 4] & 0x01) << 5
    out[:, 0] += (data[:, 5] & 0x08) << 1
    out[:, 0] += (data[:, 7] & 0x08)
    out[:, 0] += (data[:, 2] & 0x01) << 2
    out[:, 0] += (data[:, 6] & 0x01) << 1
    out[:, 0] += (data[:, 4] & 0x08) >> 3

    out[:, 1] = (data[:, 0] & 0x08) << 4
    out[:, 1] += (data[:, 3] & 0x02) << 5
    out[:, 1] += (data[:, 5] & 0x02) << 4
    out[:, 1] += (data[:, 6] & 0x04) << 2
    out[:, 1] += (data[:, 1] & 0x08)
    out[:, 1] += (data[:, 4] & 0x04)
    out[:, 1] += (data[:, 7] & 0x02)
    out[:, 1] += (data[:, 2] & 0x04) >> 2

    out[:, 2] = (data[:, 0] & 0x04) << 5
    out[:, 2] += (data[:, 1] & 0x01) << 6
    out[:, 2] += (data[:, 5] & 0x01) << 5
    out[:, 2] += (data[:, 3] & 0x04) << 2
    out[:, 2] += (data[:, 7] & 0x01) << 3
    out[:, 2] += (data[:, 6] & 0x02) << 1
    out[:, 2] += (data[:, 0] & 0x02)
    out[:, 2] += (data[:, 2] & 0x08) >> 3

    out[:, 3] = (data[:, 4] & 0x02) << 6
    out[:, 3] += (data[:, 3] & 0x08) << 3
    out[:, 3] += (data[:, 7] & 0x04) << 3
    out[:, 3] += (data[:, 1] & 0x04) << 2
    out[:, 3] += (data[:, 5] & 0x04) << 1
    out[:, 3] += (data[:, 2] & 0x02) << 1
    out[:, 3] += (data[:, 0] & 0x01) << 1
    out[:, 3] += (data[:, 6] & 0x08) >> 3

    return out.reshape(dimensions[:-1] + (4,))


def inv_permutation_p(state):
    """Compute inverse of DES permutation P (PP) operation, resulting on a 8x4bits words state.

    Args:
        state (numpy.ndarray): a uint8 array of 4 bytes words as last dimension.

    Returns:
        (numpy.ndarray) Inverse of DES permutation P result with 8x4bits dimension words.

    """
    _is_bytes_of_len(state, length=[4])
    dimensions = state.shape
    data = state.reshape((-1, 4))
    out = _np.zeros((data.shape[0], 8), dtype=_np.int64)

    out[:, 0] += ((data[:, 1] & 0x80) != 0) << 3
    out[:, 0] += ((data[:, 2] & 0x80) != 0) << 2
    out[:, 0] += ((data[:, 2] & 0x02) != 0) << 1
    out[:, 0] += ((data[:, 3] & 0x02) != 0)

    out[:, 1] += ((data[:, 1] & 0x08) != 0) << 3
    out[:, 1] += ((data[:, 3] & 0x10) != 0) << 2
    out[:, 1] += ((data[:, 0] & 0x40) != 0) << 1
    out[:, 1] += ((data[:, 2] & 0x40) != 0)

    out[:, 2] += ((data[:, 2] & 0x01) != 0) << 3
    out[:, 2] += ((data[:, 1] & 0x01) != 0) << 2
    out[:, 2] += ((data[:, 3] & 0x04) != 0) << 1
    out[:, 2] += ((data[:, 0] & 0x04) != 0)

    out[:, 3] += ((data[:, 3] & 0x40) != 0) << 3
    out[:, 3] += ((data[:, 2] & 0x10) != 0) << 2
    out[:, 3] += ((data[:, 1] & 0x40) != 0) << 1
    out[:, 3] += ((data[:, 0] & 0x80) != 0)

    out[:, 4] += ((data[:, 0] & 0x01) != 0) << 3
    out[:, 4] += ((data[:, 1] & 0x04) != 0) << 2
    out[:, 4] += ((data[:, 3] & 0x80) != 0) << 1
    out[:, 4] += ((data[:, 0] & 0x20) != 0)

    out[:, 5] += ((data[:, 0] & 0x10) != 0) << 3
    out[:, 5] += ((data[:, 3] & 0x08) != 0) << 2
    out[:, 5] += ((data[:, 1] & 0x20) != 0) << 1
    out[:, 5] += ((data[:, 2] & 0x20) != 0)

    out[:, 6] += ((data[:, 3] & 0x01) != 0) << 3
    out[:, 6] += ((data[:, 1] & 0x10) != 0) << 2
    out[:, 6] += ((data[:, 2] & 0x04) != 0) << 1
    out[:, 6] += ((data[:, 0] & 0x02) != 0)

    out[:, 7] += ((data[:, 0] & 0x08) != 0) << 3
    out[:, 7] += ((data[:, 3] & 0x20) != 0) << 2
    out[:, 7] += ((data[:, 1] & 0x02) != 0) << 1
    out[:, 7] += ((data[:, 2] & 0x08) != 0)

    return out.reshape(dimensions[:-1] + (8,)).astype(_np.uint8)


def final_permutation(state):
    """Compute DES final permutation (FP) operation resulting on a 8 bytes words state.

    This operation outputs the R16L16 value, ie: the final ciphertext.

    Args:
        state (numpy.ndarray): a uint8 array of 8 bytes words as last dimension.

    Returns:
        (numpy.ndarray) DES final permutation result with same dimensions as input state.

    """
    _is_bytes_of_len(state)
    dimensions = state.shape
    data = state.reshape((-1, 8))
    out = _np.zeros((data.shape[0], 8), dtype=_np.uint8)
    for current_byte in _np.arange(8):
        real_current_byte_indice = int(current_byte / 2) + 4 * (1 - (current_byte % 2))
        out[:, 0] <<= 1
        out[:, 0] += (data[:, real_current_byte_indice]) & 0x01
        out[:, 1] <<= 1
        out[:, 1] += (data[:, real_current_byte_indice] >> 1) & 0x01
        out[:, 2] <<= 1
        out[:, 2] += (data[:, real_current_byte_indice] >> 2) & 0x01
        out[:, 3] <<= 1
        out[:, 3] += (data[:, real_current_byte_indice] >> 3) & 0x01
        out[:, 4] <<= 1
        out[:, 4] += (data[:, real_current_byte_indice] >> 4) & 0x01
        out[:, 5] <<= 1
        out[:, 5] += (data[:, real_current_byte_indice] >> 5) & 0x01
        out[:, 6] <<= 1
        out[:, 6] += (data[:, real_current_byte_indice] >> 6) & 0x01
        out[:, 7] <<= 1
        out[:, 7] += (data[:, real_current_byte_indice] >> 7) & 0x01
    return out.reshape(dimensions)


def encrypt(plaintext, key, at_round=None, after_step=Steps.FINAL_PERMUTATION, at_des=None):
    """Encrypt plaintext using DES with provided key.

    DES, TDES2 and TDES3 modes are supported, depending on the length of key (respectively 8 bytes, 16 bytes and 24 bytes).
    Multiple parallel encryption is supported, with several modes:

        - multiple plaintexts with one key
        - one plaintext with multiple keys
        - multiple plaintexts and keys associated by pairs

    Encryption can be stopped at any desired round, any step in the targeted round, and any desired DES execution for TDES2 and TDES3.

    Args:
        plaintext (numpy.ndarray): a uint8 array of 8 bytes words as last dimension.
            Multiple plaintexts can be provided as an array of shape (N, 8).
        key (numpy.ndarray): a uint8 array of 8, 16 or 24 bytes words as last dimension.
            Multiple keys can be provided as an array of shape (N, 8), (N, 16), (N, 24).
        at_round (int, default: None): stop encryption at the end of the targeted round. Must be between 0 and 15.
        after_step (int): stop encryption after targeted operation of the round. Each round have 8 operations (for rounds where some operations are missing,
            the identity function is ised instead).
            Must be between 0 and 9. Use Steps enumeration to benefit from an explicit steps naming.
        at_des (int, default: None): stop encryption at the end of the targeted DES encryption. Not usable with classic DES because there is only one DES
            operation, it allows to stop at first or second DES operation instead of the third for TDES2 and TDES3.
            Must be between 0 and 2, depending on key size.

    Returns:
        (numpy.ndarray) resulting ciphertext.

        If multiple keys and/or plaintexts are provided, ciphertext can be:
            - if one key and one plaintext are provided, ciphertext has a shape (8)
            - if N keys (resp. N plaintexts) are provided with one plaintext (resp. one key), ciphertext has a shape (N, 8)
                with N resulting encryptions of plaintext (resp. encryptions of plaintexts with key)
            - if N keys and N plaintexts are provided, result has a shape (N, 8), with N resulting encryptions of each plaintext with each key
                of same index. Keys and plaintexts must have the same first dimension.

    Raises:
        TypeError and ValueError if plaintext and key are not of the proper types or have incompatible shapes.

    """
    return _ParametricCipher().parametric_cipher(state=plaintext, key=key, at_round=at_round, after_step=after_step,
                                                 at_des=at_des, mode='encrypt')


def decrypt(ciphertext, key, at_round=None, after_step=Steps.FINAL_PERMUTATION, at_des=None):
    """Decrypt ciphertext using DES with provided key.

    DES, TDES2 and TDES3 modes are supported, depending on the length of key (respectively 8 bytes, 16 bytes and 24 bytes).
    Multiple parallel decryption is supported, with several modes:

        - multiple ciphertexts with one key
        - one ciphertext with multiple keys
        - multiple ciphertexts and keys associated by pairs

    Decryption can be stopped at any desired round and any step in the targeted round, and any desired DES execution for TDES2 and TDES3.

    Args:
        ciphertext (numpy.ndarray): a uint8 array of 8 bytes words as last dimension.
            Multiple ciphertext can be provided as an array of shape (N, 8).
        key (numpy.ndarray): a uint8 array of 8, 16 or 24 bytes words as last dimension.
            Multiple keys can be provided as an array of shape (N, 8).
        at_round (int, default: None): stop decryption at the end of the targeted round. Must be between 0 and 15.
        after_step (int): stop decryption after targeted operation of the round. Each round have 8 operations (for rounds where some operations are missing,
            the identity function is ised instead).
            Must be between 0 and 9. Use Steps enumeration to benefit from an explicit steps naming.
        at_des (int, default: None): stop decryption at the end of the targeted DES decryption. Not usable with classic DES because there is only one DES
            operation, it allows to stop at first or second DES operation instead of the third for TDES2 and TDES3.
            Must be between 0 and 2, depending on key size.

    Returns:
        (numpy.ndarray) resulting plaintext.

        If multiple keys and/or ciphertexts are provided, plaintext can be:
            - if one key and one ciphertext are provided, plaintext has a shape (8)
            - if N keys (resp. N ciphertexts) are provided with one ciphertext (resp. one key), plaintext has a shape (N, 8)
                with N resulting decryption of ciphertext (resp. decryptions of ciphertexts with key)
            - if N keys and N ciphertexts are provided, result has a shape (N, 8), with N resulting decryptions of each ciphertext with each key
                of same index. Keys and ciphertexts must have the same first dimension.

    Raises:
        TypeError and ValueError if ciphertext and key are not of the proper types or have incompatible shapes.

    """
    return _ParametricCipher().parametric_cipher(state=ciphertext, key=key, at_round=at_round, after_step=after_step,
                                                 at_des=at_des, mode='decrypt')


class _ParametricCipher():

    def parametric_cipher(self, state, key, after_step, at_round=None, at_des=None, mode='encrypt'):
        _is_bytes_of_len(state)
        _is_bytes_of_len(key, length=[8, 16, 24, 128, 256, 384])
        if key.ndim > 2 or state.ndim > 2:
            raise ValueError(f'{mode} support only list of inputs and keys. Inputs and keys should be limited to arrays of shape (N, 8).')
        elif key.ndim == 2 and state.ndim == 2 and key.shape[0] != state.shape[0]:
            raise ValueError(f'{mode} support using 2 dimensions array for both keys and inputs only if first dimension of both are equal.')

        self.at_round = self._set_at_round(at_round)
        self.after_step = self._set_after_step(after_step)
        self.at_des = self._set_at_des(at_des, key)
        self.mode = self._set_mode(mode)

        des_keys = self._prepare_keys(key=key, state=state)
        des_iterations = self._prepare_des_iterations()

        if state.ndim == 1:
            out_state = _np.array([_np.copy(state) for i in range(des_keys.shape[1])], dtype='uint8').reshape(-1, 8)
        else:
            out_state = _np.array([_np.copy(s) for s in state], dtype='uint8')

        self.total_words = _np.max([out_state.shape[0], des_keys.shape[1]])
        for des_number, des_rounds in enumerate(des_iterations):
            for round_number, round_operations in enumerate(des_rounds):
                for step, operation in enumerate(round_operations):
                    out_state = self._parametric_cipher_step(out_state=out_state, operation=operation, des_keys=des_keys,
                                                             des_number=des_number, round_number=round_number)
        return out_state.squeeze()

    def _parametric_cipher_step(self, out_state, operation, des_keys, des_number, round_number):  # noqa: C901
        if operation is Steps.INITIAL_PERMUTATION:
            out_state = initial_permutation(out_state)
        elif operation is Steps.EXPANSIVE_PERMUTATION:
            self.saved_left_right = _np.copy(out_state)
            out_state = expansive_permutation(out_state[:, 4:8])
        elif operation is Steps.ADD_ROUND_KEY:
            out_state = add_round_key(state=out_state, keys=des_keys[des_number, :, round_number, :])
        elif operation is Steps.SBOXES:
            out_state = sboxes(out_state)
        elif operation is Steps.PERMUTATION_P:
            out_state[:, 0:4] = permutation_p(out_state)
            out_state[:, 4:8] = _np.zeros((self.total_words, 4), dtype=_np.uint8)
        elif operation is Steps.XOR_WITH_SAVED_LEFT_RIGHT:
            out_state = _np.bitwise_xor(out_state, self.saved_left_right)
        elif operation is Steps.PERMUTE_RIGHT_LEFT:
            out_state = _np.roll(out_state, shift=4, axis=-1)
        elif operation is Steps.INV_PERMUTATION_P_RIGHT:
            out_state = inv_permutation_p(out_state[:, 4:8])
        elif operation is Steps.INV_PERMUTATION_P_DELTA_RIGHT:
            out_state = inv_permutation_p(_np.bitwise_xor(out_state[:, 0:4], out_state[:, 4:8]))
        elif operation is Steps.FINAL_PERMUTATION:
            out_state = final_permutation(out_state)
        return out_state

    def _prepare_keys(self, key, state):
        if key.ndim == 1:
            # we make the same shape dimensions (number of keys, bytes) whether there are one or multiple keys
            keys_to_use = key.reshape((1, ) + key.shape)
        else:
            keys_to_use = key

        expanded_keys = []
        for current_des in range(self.at_des + 1):
            if current_des == 2 and (key.shape[-1] == 16 or key.shape[-1] == 256):
                # in case we are in TDES2 mode, for the third key we duplicate key 1
                if key.shape[-1] == 16:
                    expanded_keys.append(key_schedule(keys_to_use[:, 0:8]))
                else:
                    expanded_keys.append(keys_to_use[:, 0:128].reshape(-1, 16, 8))
            elif self.mode == 'decrypt' and (key.shape[-1] == 24 or key.shape[-1] == 384):
                # for TDES3 key 1 and key 3 are switched at decrypt
                if key.shape[-1] == 24:
                    expanded_keys.append(key_schedule(keys_to_use[:, (2 - current_des) * 8:(2 - current_des + 1) * 8]))
                else:
                    expanded_keys.append(keys_to_use[:, (2 - current_des) * 128:(2 - current_des + 1) * 128].reshape(-1, 16, 8))
            else:
                if key.shape[-1] == 8 or key.shape[-1] == 16 or key.shape[-1] == 24:
                    expanded_keys.append(key_schedule(keys_to_use[:, current_des * 8:(current_des + 1) * 8]))
                else:
                    expanded_keys.append(keys_to_use[:, current_des * 128:(current_des + 1) * 128].reshape(-1, 16, 8))

            if (self.mode == 'encrypt' and current_des == 1) or (self.mode == 'decrypt' and current_des != 1):
                expanded_keys[-1] = _np.flip(expanded_keys[-1], axis=1)

        # we finally return a 4 dimension numpy array (des number, number of keys, rounds, bytes)
        return _np.array(expanded_keys, dtype=_np.uint8)

    MANDATORY_ROUND_ELEMENTS = [Steps.EXPANSIVE_PERMUTATION,
                                Steps.ADD_ROUND_KEY,
                                Steps.SBOXES,
                                Steps.PERMUTATION_P,
                                Steps.XOR_WITH_SAVED_LEFT_RIGHT]
    FIRST_ROUND = [Steps.INITIAL_PERMUTATION] + MANDATORY_ROUND_ELEMENTS + [Steps.PERMUTE_RIGHT_LEFT, None, None, None]
    ROUND = [None] + MANDATORY_ROUND_ELEMENTS + [Steps.PERMUTE_RIGHT_LEFT, None, None, None]
    LAST_ROUND = [None] + MANDATORY_ROUND_ELEMENTS + [None, None, None, None]
    FINAL_ROUND = [None] + MANDATORY_ROUND_ELEMENTS + [None, None, None, Steps.FINAL_PERMUTATION]

    def _prepare_des_iterations(self):
        des_iterations = []
        for current_des in range(self.at_des + 1):
            if current_des == 0:
                if current_des == self.at_des:
                    operations = [self.FIRST_ROUND, self.ROUND, self.FINAL_ROUND]
                    des_iterations.append(self._prepare_rounds(at_round=self.at_round, after_step=self.after_step, operations=operations))
                else:
                    operations = [self.FIRST_ROUND, self.ROUND, self.LAST_ROUND]
                    des_iterations.append(self._prepare_rounds(at_round=(DES_ROUNDS - 1), after_step=(len(Steps) - 1), operations=operations))
            else:
                if current_des == self.at_des:
                    operations = [self.ROUND, self.ROUND, self.FINAL_ROUND]
                    des_iterations.append(self._prepare_rounds(at_round=self.at_round, after_step=self.after_step, operations=operations))
                else:
                    operations = [self.ROUND, self.ROUND, self.LAST_ROUND]
                    des_iterations.append(self._prepare_rounds(at_round=(DES_ROUNDS - 1), after_step=(len(Steps) - 1), operations=operations))
        return des_iterations

    def _prepare_rounds(self, at_round, after_step, operations) -> _typ.List[_typ.Union[Steps, None]]:
        rounds = [operations[0]]
        for i in range(1, at_round + 1):
            if i == DES_ROUNDS - 1:
                rounds.append(operations[-1])
            else:
                rounds.append(operations[1])
        rounds[-1] = rounds[-1][:after_step + 1]

        # These steps are always performed if asked as after_step
        if after_step == Steps.PERMUTE_RIGHT_LEFT:
            rounds[-1][after_step] = Steps.PERMUTE_RIGHT_LEFT
        elif after_step == Steps.INV_PERMUTATION_P_RIGHT:
            rounds[-1][after_step] = Steps.INV_PERMUTATION_P_RIGHT
            rounds[-1][after_step - 1] = Steps.PERMUTE_RIGHT_LEFT
        elif after_step == Steps.INV_PERMUTATION_P_DELTA_RIGHT:
            rounds[-1][after_step] = Steps.INV_PERMUTATION_P_DELTA_RIGHT

        return rounds

    # Param checks

    def _set_at_round(self, at_round):
        if at_round is not None:
            if not isinstance(at_round, int):
                raise TypeError(f'at_round must be an integer value, not {at_round}.')
            if at_round >= DES_ROUNDS or at_round < 0:
                raise ValueError(f'at_round must be > 0 and < {DES_ROUNDS}, not {at_round}.')
            return at_round
        else:
            return DES_ROUNDS - 1

    def _set_after_step(self, after_step):
        if not isinstance(after_step, int):
            raise TypeError(f'after_step must be an integer or a Steps enum value, not {after_step}.')
        if after_step < 0 or after_step > (len(Steps) - 1):
            raise ValueError(f'after_step can\'t be < 0 or > {len(Steps) - 1}, not {after_step}. Use Step enumeration.')
        return after_step

    def _set_at_des(self, at_des, key):
        if at_des is not None:
            if not isinstance(at_des, int):
                raise TypeError(f'at_des must be an integer value, not {at_des}.')
            elif (key.shape[-1] == 8 or key.shape[-1] == 128) and at_des != 0:
                raise ValueError(f'at_des can only be 0 if DES is used.')
            elif (key.shape[-1] == 16 or key.shape[-1] == 256 or key.shape[-1] == 24 or key.shape[-1] == 384) and at_des != 0 and at_des != 1 and at_des != 2:
                raise ValueError(f'at_des can only be 0, 1 or 2 if TDES2 or TDES3 are used.')
            return at_des
        else:
            if key.shape[-1] == 8 or key.shape[-1] == 128:
                return 0
            else:
                return 2

    def _set_mode(self, mode):
        if not isinstance(mode, str):
            raise TypeError(f'mode must be an str, not a {type(mode)}.')
        if mode != 'encrypt' and mode != 'decrypt':
            raise ValueError(f'mode must be "enctypt" or "decrypt", not "{mode}"')
        return mode
