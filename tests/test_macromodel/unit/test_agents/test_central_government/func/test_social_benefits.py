import numpy as np

from macromodel.agents.central_government.func.social_benefits import (
    DefaultSocialBenefitsSetter,
)


class RecordingBenefitModel:
    def __init__(self, growth_ratio):
        self.growth_ratio = growth_ratio
        self.seen_features = None

    def predict(self, features):
        self.seen_features = features
        return np.array([self.growth_ratio])


class TestSocialBenefitsSetter:
    def test__compute_unemployment_benefits(self):
        assert (
            DefaultSocialBenefitsSetter().compute_unemployment_benefits(
                prev_unemployment_benefits=100.0,
                benefit_indexation_inflation=np.array([0.01, 0.02]),
                current_unemployment_rate=0.1,
                current_estimated_growth=0.0,
                model=None,
            )
            == 100.0
        )

    def test__model_features_use_benefit_indexation_inflation_as_data_cpi_inflation(self):
        model = RecordingBenefitModel(growth_ratio=1.1)

        benefit = DefaultSocialBenefitsSetter().compute_unemployment_benefits(
            prev_unemployment_benefits=100.0,
            benefit_indexation_inflation=np.array([0.01, 0.03]),
            current_unemployment_rate=0.2,
            current_estimated_growth=0.0,
            model=model,
        )

        assert np.isclose(benefit, 110.0)
        assert list(model.seen_features.columns) == ["Data CPI Inflation", "Unemployment Rate"]
        assert model.seen_features["Data CPI Inflation"].iloc[0] == 0.03
        assert model.seen_features["Unemployment Rate"].iloc[0] == 0.2

    def test__compute_regular_transfer_to_households(self):
        assert (
            DefaultSocialBenefitsSetter().compute_regular_transfer_to_households(
                prev_regular_transfer_to_households=100.0,
                benefit_indexation_inflation=np.array([0.01, 0.02]),
                current_unemployment_rate=0.1,
                current_estimated_growth=0.0,
                model=None,
            )
            == 100.0
        )
