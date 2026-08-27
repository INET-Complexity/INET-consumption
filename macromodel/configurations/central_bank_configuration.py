from typing import Literal

from pydantic import BaseModel, Field


class CentralBankTaylorRuleOverrides(BaseModel):
    """Optional overrides for the ARDL-estimated SmoothTaylorRule coefficients.

    `rho`, `r_star`, `phi_pi`, and `phi_q` are estimated from historical policy-rate,
    CPI, and output-gap data in `DefaultSyntheticCentralBank._estimate_smooth_taylor_rule`
    and are otherwise not configurable per country. The defaults here (`None`) keep
    those estimated values unchanged; setting any field overrides just that
    coefficient in `SmoothTaylorRule.compute_rate`, regardless of which policy rule
    is selected in `functions.policy_rate.name`.
    """

    targeted_inflation_rate: float | None = None
    rho: float | None = None
    r_star: float | None = None
    phi_pi: float | None = None
    phi_q: float | None = None


class CentralBankPolicy(BaseModel):
    """Central bank policy rate determination configuration.

    Defines the mechanism for setting monetary policy through:
    - Interest rate determination
    - Policy rule implementation
    - Rate adjustment mechanisms

    The configuration supports:
    - Constant policy rates: Fixed interest rates
    - Poledna policy rates: Dynamic rate adjustment based on economic conditions

    Attributes:
        path_name (str): Module path for policy rate functions
        name (Literal): Selected policy mechanism ("ConstantPolicyRate",
            "PolednaPolicyRate", or "SmoothTaylorRule")
        parameters (dict): Configuration parameters for policy implementation
    """

    path_name: str = "policy_rate"
    name: Literal["ConstantPolicyRate", "PolednaPolicyRate", "SmoothTaylorRule"] = "ConstantPolicyRate"
    parameters: dict = {}


class CentralBankFunctions(BaseModel):
    """Collection of central bank function configurations.

    Aggregates the various functional components that define
    central bank operations through:
    - Monetary policy implementation
    - Interest rate management
    - Policy rule execution

    Attributes:
        policy_rate (CentralBankPolicy): Policy rate determination configuration
    """

    policy_rate: CentralBankPolicy = CentralBankPolicy()


class CentralBankConfiguration(BaseModel):
    """Complete central bank behavior configuration.

    Defines the overall configuration for central bank operations through:
    - Policy frameworks
    - Operational procedures
    - Monetary tools
    - Implementation mechanisms

    The configuration determines how the central bank:
    - Sets interest rates
    - Implements monetary policy
    - Responds to economic conditions
    - Manages policy transmission

    Attributes:
        functions (CentralBankFunctions): Collection of function configurations
            that define central bank behavior
        taylor_rule_overrides (CentralBankTaylorRuleOverrides): Optional per-country
            overrides for the estimated SmoothTaylorRule coefficients
    """

    functions: CentralBankFunctions = CentralBankFunctions()
    taylor_rule_overrides: CentralBankTaylorRuleOverrides = Field(default_factory=CentralBankTaylorRuleOverrides)
