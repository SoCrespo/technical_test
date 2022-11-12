from utils import *

import pytest

def test_split_fedas_code():
    assert split_fedas_code("123456") == [1, 23, 45, 6]
    assert split_fedas_code("") == [-1, -1, -1, -1]
    with pytest.raises(TypeError):
        split_fedas_code(123456)
        split_fedas_code(123456.0)
        split_fedas_code("1234")
        