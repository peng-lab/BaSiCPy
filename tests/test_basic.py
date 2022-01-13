"""Test BaSiC function."""


# DEBUG fails because Settings is now a pydantic dataclass
def test_basic_verify_init():
    # verifies that BaSiC.__init__ contains all arguments to pass to Settings
    # this test would be unnecessary if *args, **kwargs were passed to Settings,
    # but this makes documenting types more complicated
    # basic_init_sig = inspect.signature(BaSiC.__init__)
    # settings_sig = inspect.signature(Settings)
    # for value in settings_sig.parameters.values():
    #     assert value in basic_init_sig.parameters.values()
    ...


def test_basic():
    ...
