from typing import Literal

from pydantic import BaseModel, Field


class SocialBenefits(BaseModel):
    name: Literal[
        "ConstantSocialBenefitsSetter",
        "DefaultSocialBenefitsSetter",
        "GrowthSocialBenefitsSetter",
    ] = "GrowthSocialBenefitsSetter"
    path_name: str = "social_benefits"
    parameters: dict = Field(default_factory=dict)


class SocialHousing(BaseModel):
    name: Literal["DefaultSocialHousing"] = "DefaultSocialHousing"
    path_name: str = "social_housing"
    parameters: dict = Field(default_factory=lambda: {"rent_as_fraction_of_unemployment_rate": 0.25})


class DebtInterest(BaseModel):
    name: Literal["CurrentPolicyRateDebtInterest", "SmoothedPolicyRateDebtInterest"] = "CurrentPolicyRateDebtInterest"
    path_name: str = "debt_interest"
    parameters: dict = Field(default_factory=dict)


class CentralGovernmentFunctions(BaseModel):
    social_benefits: SocialBenefits = SocialBenefits()
    social_housing: SocialHousing = SocialHousing()
    debt_interest: DebtInterest = DebtInterest()


class CentralGovernmentTaxOverrides(BaseModel):
    """Optional country-level overrides for fiscal rates and tax vectors.

    The defaults keep the historical reader values and the generic model
    accounting unchanged.  Country configuration files can opt into a
    separate household/firm capital-formation tax base or scale the shared
    production-tax vector.
    """

    employer_social_insurance_rate: float | None = None
    value_added_tax_rate: float | None = None
    household_investment_vat_rate: float | None = None
    household_capital_formation_rate: float | None = None
    firm_capital_formation_rate: float | None = None
    production_tax_vector_scale: float | None = Field(default=None, gt=0.0)


class CentralGovernmentConfiguration(BaseModel):
    functions: CentralGovernmentFunctions = CentralGovernmentFunctions()
    tax_overrides: CentralGovernmentTaxOverrides = Field(default_factory=CentralGovernmentTaxOverrides)
