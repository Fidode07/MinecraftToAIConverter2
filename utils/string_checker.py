from typing import *


def is_empty(s: Any) -> bool:
    """
    Checks if a string has a real value
    :param s: The given param
    :return: True if there is something in
    """
    return not all([s, isinstance(s, str), len(str(s).strip())])
