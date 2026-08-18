import numpy as np
import pytest

from macro_data.util.rates import compound_rate, fisher_real_rate


def test_compound_rate_uses_exact_geometric_conversion():
    assert compound_rate(0.01, 4) == pytest.approx((1.01**4) - 1.0)


def test_fisher_real_rate_supports_scalars_and_arrays():
    assert fisher_real_rate(0.05, 0.02) == pytest.approx(1.05 / 1.02 - 1.0)
    np.testing.assert_allclose(
        fisher_real_rate(np.array([0.05, 0.03]), np.array([0.02, 0.01])),
        np.array([1.05 / 1.02 - 1.0, 1.03 / 1.01 - 1.0]),
    )


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (compound_rate, (-1.0, 4)),
        (fisher_real_rate, (-1.0, 0.02)),
        (fisher_real_rate, (0.05, -1.0)),
    ],
)
def test_rate_transforms_reject_invalid_gross_rates(function, args):
    with pytest.raises(ValueError):
        function(*args)


@pytest.mark.parametrize("invalid_rate", [np.nan, np.inf, -np.inf])
def test_compound_rate_rejects_nonfinite_inputs(invalid_rate):
    with pytest.raises(ValueError, match="rate must be finite"):
        compound_rate(invalid_rate, 4)


@pytest.mark.parametrize("invalid_rate", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("argument", ["nominal", "inflation"])
def test_fisher_real_rate_rejects_nonfinite_inputs(invalid_rate, argument):
    nominal_rate = invalid_rate if argument == "nominal" else 0.05
    inflation_rate = invalid_rate if argument == "inflation" else 0.02

    with pytest.raises(ValueError, match=rf"{argument}_rate must be finite"):
        fisher_real_rate(nominal_rate, inflation_rate)
