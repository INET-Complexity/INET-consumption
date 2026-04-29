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


class CentralGovernmentConfiguration(BaseModel):
    functions: CentralGovernmentFunctions = CentralGovernmentFunctions()
