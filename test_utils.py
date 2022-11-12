from utils import *

from numpy import nan
import pytest

def test_split_fedas_code():
    assert split_fedas_code("123456") == [1, 23, 45, 6]
    assert split_fedas_code("") == [-1, -1, -1, -1]
    with pytest.raises(TypeError):
        split_fedas_code(123456)
        split_fedas_code(123456.0)
        split_fedas_code("1234")

        
def test_normalize():
    assert normalize("Hello World!") == "hello world"
    assert normalize("12(TEXT22-X ") == "text x"
    assert normalize("09-SHOES (LOW)") == "shoes low"
    assert normalize(2) ==""
    assert normalize(nan) ==""
    assert normalize("12(TEXT22-X ", keep_digits=True) == "12 text22 x"