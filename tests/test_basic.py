from pybasic import BaSiC


# DEBUG fails because Settings is now a pydantic dataclass
def test_basic_verify_init():

    basic = BaSiC()

    assert all([d == 128 for d in basic.darkfield.shape])
    assert all([d == 128 for d in basic.flatfield.shape])

    return


def test_basic():
    ...
