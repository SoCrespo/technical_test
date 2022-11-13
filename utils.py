# encoding: utf-8

import re
from typing import List

def split_fedas_code(x: str) -> List[int]:
    """
    Returns a list of 4 digits groups from a fedas code of 6 digits :
    ""123456"" -> [1, 23, 45, 6], or [-1, -1, -1, -1] if the fedas code is empty.
    """
    if x == "":
        return [-1, -1, -1, -1]
    if not isinstance(x, str):
        raise TypeError(f"Expected a string, got {type(x)}")
    if len(x) != 6:
        raise ValueError(f"Expected a string of 6 digits, got {len(x)}")
    return [int(x[:1]), int(x[1:3]), int(x[3:5]), int(x[5:])]


def normalize(x, keep_digits=False):
    """
    Return x in lowercase and with only letters 
    (punctuation or special characters).
    If keep digits is True, digits are kept.
    """
    if not isinstance(x, str):
        return ""
    x = x.lower()
    if keep_digits:
        result = re.sub(r"[^a-z0-9 _]", " ", x)
    else:
        result = re.sub(r"[^a-z _]", " ", x)
    return " ".join(result.split())
