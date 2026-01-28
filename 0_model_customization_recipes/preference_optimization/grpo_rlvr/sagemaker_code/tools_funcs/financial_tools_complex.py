"""
Financial Operations Mock APIs for Agent Training
Complex argument structures with 8 functions
Returns simple string answers for training data comparison
"""

import json
from typing import List, Dict, Union, Optional
from datetime import datetime

# =============================================================================
# DETERMINISTIC DATA TABLES
# =============================================================================

# Asset classes and their risk profiles
ASSET_CLASSES = {
    "equity_us_large": {"risk": 0.18, "return": 0.095, "liquidity": "high"},
    "equity_us_small": {"risk": 0.22, "return": 0.11, "liquidity": "high"},
    "equity_intl_dev": {"risk": 0.20, "return": 0.085, "liquidity": "high"},
    "equity_emerging": {"risk": 0.28, "return": 0.12, "liquidity": "medium"},
    "fixed_income_govt": {"risk": 0.05, "return": 0.035, "liquidity": "high"},
    "fixed_income_corp": {"risk": 0.08, "return": 0.048, "liquidity": "high"},
    "fixed_income_high_yield": {"risk": 0.14, "return": 0.065, "liquidity": "medium"},
    "real_estate": {"risk": 0.16, "return": 0.078, "liquidity": "low"},
    "commodities": {"risk": 0.24, "return": 0.055, "liquidity": "medium"},
    "alternatives": {"risk": 0.19, "return": 0.088, "liquidity": "low"},
    "cash": {"risk": 0.01, "return": 0.025, "liquidity": "high"},
}

# Tax brackets (simplified)
TAX_BRACKETS = {
    "single": [(11000, 0.10), (44725, 0.12), (95375, 0.22), (182100, 0.24), (231250, 0.32), (578125, 0.35), (float('inf'), 0.37)],
    "married": [(22000, 0.10), (89050, 0.12), (190750, 0.22), (364200, 0.24), (462500, 0.32), (693750, 0.35), (float('inf'), 0.37)],
    "head": [(15700, 0.10), (59850, 0.12), (95350, 0.22), (182100, 0.24), (231250, 0.32), (578100, 0.35), (float('inf'), 0.37)],
}

# Currency exchange rates (base USD)
EXCHANGE_RATES = {
    "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5, "CHF": 0.88,
    "CAD": 1.36, "AUD": 1.52, "CNY": 7.24, "INR": 83.2, "BRL": 4.98,
}

# Credit score ranges
CREDIT_TIERS = {
    "excellent": (750, 850, 0.035),
    "good": (700, 749, 0.045),
    "fair": (650, 699, 0.062),
    "poor": (600, 649, 0.085),
    "bad": (300, 599, 0.125),
}

# Loan types and their characteristics
LOAN_TYPES = {
    "conventional": {"min_down": 0.03, "pmi_threshold": 0.20, "max_dti": 0.43},
    "fha": {"min_down": 0.035, "pmi_threshold": 0.10, "max_dti": 0.50},
    "va": {"min_down": 0.0, "pmi_threshold": 0.0, "max_dti": 0.41},
    "jumbo": {"min_down": 0.10, "pmi_threshold": 0.20, "max_dti": 0.38},
}

# Investment account types
ACCOUNT_TYPES = {
    "taxable_brokerage": {"tax_deferred": False, "contribution_limit": None, "withdrawal_penalty": False},
    "traditional_401k": {"tax_deferred": True, "contribution_limit": 23000, "withdrawal_penalty": True},
    "roth_401k": {"tax_deferred": False, "contribution_limit": 23000, "withdrawal_penalty": True},
    "traditional_ira": {"tax_deferred": True, "contribution_limit": 7000, "withdrawal_penalty": True},
    "roth_ira": {"tax_deferred": False, "contribution_limit": 7000, "withdrawal_penalty": True},
    "hsa": {"tax_deferred": True, "contribution_limit": 4150, "withdrawal_penalty": False},
}

# Insurance premium factors
INSURANCE_FACTORS = {
    "life": {"age_mult": 0.015, "health_mult": {"excellent": 0.8, "good": 1.0, "fair": 1.3, "poor": 1.8}},
    "disability": {"age_mult": 0.012, "occupation_mult": {"low": 0.9, "medium": 1.0, "high": 1.4}},
    "ltc": {"age_mult": 0.025, "health_mult": {"excellent": 0.85, "good": 1.0, "fair": 1.4, "poor": 2.1}},
}


# =============================================================================
# COMPLEX API FUNCTIONS - 8-12 parameters each
# =============================================================================

def calculate_portfolio_allocation(
    total_investment: float,
    risk_tolerance: str,
    time_horizon_years: int,
    current_age: int,
    retirement_age: int,
    income_need_percentage: float,
    existing_allocations: Dict[str, float],
    esg_preference: bool,
    tax_loss_harvesting: bool,
    rebalancing_frequency: str,
    inflation_protection: bool,
    liquidity_requirement: str
) -> str:
    """
    Calculate optimal portfolio allocation across multiple asset classes based on comprehensive
    investor profile including risk tolerance, time horizon, age, income needs, existing holdings,
    ESG preferences, tax strategies, rebalancing approach, inflation protection, and liquidity needs.

    Args:
        total_investment: Total amount to invest in USD. Must be positive. Example: 500000
        risk_tolerance: Investor risk profile. Must be one of:
            - "conservative": Low risk, capital preservation focus
            - "moderate_conservative": Below-average risk tolerance
            - "moderate": Balanced risk/return approach
            - "moderate_aggressive": Above-average risk tolerance
            - "aggressive": High risk, growth focus
        time_horizon_years: Investment time horizon in years. Must be 1-50. Example: 25
        current_age: Investor's current age. Must be 18-100. Example: 45
        retirement_age: Target retirement age. Must be greater than current_age. Example: 65
        income_need_percentage: Annual income needed from portfolio as percentage. 0.0-0.10. Example: 0.04
        existing_allocations: Dictionary of current holdings by asset class with percentages.
            Keys must be from ASSET_CLASSES. Values sum should be <= 1.0.
            Example: {"equity_us_large": 0.30, "fixed_income_govt": 0.20}
        esg_preference: Whether to apply ESG (Environmental, Social, Governance) filters. Boolean.
        tax_loss_harvesting: Whether to enable tax-loss harvesting strategy. Boolean.
        rebalancing_frequency: How often to rebalance. Must be one of:
            - "monthly", "quarterly", "semi_annual", "annual", "threshold_based"
        inflation_protection: Whether to include inflation-protected securities. Boolean.
        liquidity_requirement: Liquidity needs. Must be one of:
            - "immediate": Need access within days
            - "short_term": Need access within months
            - "medium_term": Need access within 1-3 years
            - "long_term": No near-term liquidity needs

    Returns:
        String with recommended allocation percentages and expected return/risk metrics.
        Format: "Allocation: equity_us_large 35%, fixed_income_govt 25%, ..., Expected Return: 6.8%, Risk: 12.3%"
    """
    # Validation
    if total_investment <= 0:
        return "Error: Investment amount must be positive"
    if risk_tolerance not in ["conservative", "moderate_conservative", "moderate", "moderate_aggressive", "aggressive"]:
        return "Error: Invalid risk tolerance"
    if time_horizon_years < 1 or time_horizon_years > 50:
        return "Error: Time horizon must be 1-50 years"
    if current_age < 18 or current_age > 100:
        return "Error: Current age must be 18-100"
    if retirement_age <= current_age:
        return "Error: Retirement age must be greater than current age"
    if income_need_percentage < 0 or income_need_percentage > 0.10:
        return "Error: Income need must be 0-10%"
    if rebalancing_frequency not in ["monthly", "quarterly", "semi_annual", "annual", "threshold_based"]:
        return "Error: Invalid rebalancing frequency"
    if liquidity_requirement not in ["immediate", "short_term", "medium_term", "long_term"]:
        return "Error: Invalid liquidity requirement"
    
    # Calculate base allocation based on risk tolerance
    risk_map = {
        "conservative": {"equity": 0.30, "fixed": 0.55, "other": 0.15},
        "moderate_conservative": {"equity": 0.45, "fixed": 0.40, "other": 0.15},
        "moderate": {"equity": 0.60, "fixed": 0.30, "other": 0.10},
        "moderate_aggressive": {"equity": 0.75, "fixed": 0.18, "other": 0.07},
        "aggressive": {"equity": 0.85, "fixed": 0.10, "other": 0.05},
    }
    
    base = risk_map[risk_tolerance]
    
    # Adjust for age (glide path)
    years_to_retirement = retirement_age - current_age
    age_adjustment = min(0.15, (100 - current_age) * 0.002)
    
    # Adjust for liquidity
    liquidity_adj = {"immediate": -0.10, "short_term": -0.05, "medium_term": 0.0, "long_term": 0.05}
    
    # Calculate final allocation
    equity_pct = base["equity"] + age_adjustment + liquidity_adj[liquidity_requirement]
    fixed_pct = base["fixed"] - age_adjustment * 0.5
    other_pct = 1.0 - equity_pct - fixed_pct
    
    # Clamp values
    equity_pct = max(0.20, min(0.90, equity_pct))
    fixed_pct = max(0.05, min(0.60, fixed_pct))
    other_pct = max(0.05, 1.0 - equity_pct - fixed_pct)
    
    # Calculate expected return and risk
    expected_return = equity_pct * 0.095 + fixed_pct * 0.042 + other_pct * 0.068
    expected_risk = (equity_pct * 0.18 + fixed_pct * 0.06 + other_pct * 0.15) * 0.85
    
    # Apply adjustments
    if esg_preference:
        expected_return *= 0.98
    if inflation_protection:
        expected_return -= 0.005
        expected_risk -= 0.01
    
    return f"Allocation: equity {equity_pct*100:.1f}%, fixed_income {fixed_pct*100:.1f}%, alternatives {other_pct*100:.1f}%, Expected Return: {expected_return*100:.1f}%, Risk: {expected_risk*100:.1f}%"



def calculate_mortgage_affordability(
    annual_income: float,
    monthly_debts: float,
    down_payment: float,
    credit_score: int,
    loan_type: str,
    property_state: str,
    property_tax_rate: float,
    hoa_fees: float,
    homeowners_insurance: float,
    pmi_required: bool,
    interest_rate_override: Optional[float],
    loan_term_years: int
) -> str:
    """
    Calculate maximum affordable home price and monthly payment based on comprehensive
    financial profile including income, debts, down payment, credit score, loan type,
    location-specific costs, insurance, PMI requirements, and loan terms.

    Args:
        annual_income: Gross annual income in USD. Must be positive. Example: 120000
        monthly_debts: Total monthly debt obligations (car, student loans, credit cards).
            Must be non-negative. Example: 850
        down_payment: Available down payment amount in USD. Must be non-negative. Example: 60000
        credit_score: FICO credit score. Must be 300-850. Example: 740
        loan_type: Type of mortgage loan. Must be one of:
            - "conventional": Standard mortgage, 3% down minimum
            - "fha": FHA loan, 3.5% down minimum, more lenient credit
            - "va": VA loan for veterans, 0% down
            - "jumbo": For high-value properties, 10% down minimum
        property_state: Two-letter state code for property location. Affects taxes and costs.
            Example: "CA", "TX", "NY", "FL"
        property_tax_rate: Annual property tax rate as decimal. 0.003-0.025. Example: 0.012
        hoa_fees: Monthly HOA/condo fees in USD. Must be non-negative. Example: 250
        homeowners_insurance: Monthly homeowners insurance cost in USD. Must be positive. Example: 150
        pmi_required: Whether PMI (Private Mortgage Insurance) is required. Boolean.
            Typically required when down payment < 20% for conventional loans.
        interest_rate_override: Optional custom interest rate as decimal. If None, calculated from credit score.
            Must be 0.02-0.12 if provided. Example: 0.065
        loan_term_years: Mortgage term in years. Must be one of: 15, 20, 30. Example: 30

    Returns:
        String with maximum home price, monthly payment breakdown, and DTI ratio.
        Format: "Max Price: $485,000, Monthly Payment: $3,245 (P&I: $2,180, Tax: $485, Ins: $150, HOA: $250, PMI: $180), DTI: 38%"
    """
    # Validation
    if annual_income <= 0:
        return "Error: Annual income must be positive"
    if monthly_debts < 0:
        return "Error: Monthly debts cannot be negative"
    if down_payment < 0:
        return "Error: Down payment cannot be negative"
    if credit_score < 300 or credit_score > 850:
        return "Error: Credit score must be 300-850"
    if loan_type not in LOAN_TYPES:
        return f"Error: Invalid loan type. Use: {', '.join(LOAN_TYPES.keys())}"
    if property_tax_rate < 0.003 or property_tax_rate > 0.025:
        return "Error: Property tax rate must be 0.3%-2.5%"
    if hoa_fees < 0:
        return "Error: HOA fees cannot be negative"
    if homeowners_insurance <= 0:
        return "Error: Homeowners insurance must be positive"
    if interest_rate_override is not None and (interest_rate_override < 0.02 or interest_rate_override > 0.12):
        return "Error: Interest rate must be 2%-12%"
    if loan_term_years not in [15, 20, 30]:
        return "Error: Loan term must be 15, 20, or 30 years"
    
    # Determine interest rate
    if interest_rate_override:
        interest_rate = interest_rate_override
    else:
        # Find credit tier
        for tier, (min_score, max_score, rate) in CREDIT_TIERS.items():
            if min_score <= credit_score <= max_score:
                interest_rate = rate
                break
    
    # Adjust rate for loan type
    loan_adjustments = {"conventional": 0.0, "fha": 0.005, "va": -0.0025, "jumbo": 0.0075}
    interest_rate += loan_adjustments[loan_type]
    
    # Calculate max DTI
    monthly_income = annual_income / 12
    max_dti = LOAN_TYPES[loan_type]["max_dti"]
    max_housing_payment = monthly_income * max_dti - monthly_debts
    
    if max_housing_payment <= 0:
        return "Error: Debt-to-income ratio too high, no affordable home price"
    
    # Calculate max loan amount (iterative approach simplified)
    monthly_rate = interest_rate / 12
    num_payments = loan_term_years * 12
    
    # Estimate PMI if required
    pmi_monthly = 0
    if pmi_required and loan_type == "conventional":
        pmi_monthly = 100  # Placeholder, will adjust
    
    # Available for P&I
    available_pi = max_housing_payment - hoa_fees - homeowners_insurance - pmi_monthly
    
    # Calculate max loan using mortgage formula
    if monthly_rate > 0:
        max_loan = available_pi * ((1 - (1 + monthly_rate) ** -num_payments) / monthly_rate)
    else:
        max_loan = available_pi * num_payments
    
    # Calculate max price
    max_price = max_loan + down_payment
    
    # Recalculate actual payments
    monthly_tax = (max_price * property_tax_rate) / 12
    monthly_pi = max_loan * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
    
    # Recalculate PMI based on actual loan
    if pmi_required:
        ltv = max_loan / max_price
        if ltv > 0.80:
            pmi_monthly = max_loan * 0.005 / 12
        else:
            pmi_monthly = 0
    
    total_monthly = monthly_pi + monthly_tax + homeowners_insurance + hoa_fees + pmi_monthly
    dti_ratio = (total_monthly + monthly_debts) / monthly_income
    
    return f"Max Price: ${max_price:,.0f}, Monthly Payment: ${total_monthly:,.0f} (P&I: ${monthly_pi:,.0f}, Tax: ${monthly_tax:,.0f}, Ins: ${homeowners_insurance:,.0f}, HOA: ${hoa_fees:,.0f}, PMI: ${pmi_monthly:,.0f}), DTI: {dti_ratio*100:.1f}%"



def optimize_tax_strategy(
    gross_income: float,
    filing_status: str,
    state: str,
    retirement_contributions: Dict[str, float],
    capital_gains_short: float,
    capital_gains_long: float,
    dividend_income_qualified: float,
    dividend_income_ordinary: float,
    itemized_deductions: float,
    dependents: int,
    self_employment_income: float,
    rental_income: float
) -> str:
    """
    Calculate comprehensive tax liability and provide optimization recommendations based on
    income sources, filing status, state, retirement contributions, capital gains, dividends,
    deductions, dependents, self-employment income, and rental income.

    Args:
        gross_income: W-2 wages and salary income in USD. Must be non-negative. Example: 150000
        filing_status: Tax filing status. Must be one of:
            - "single": Single filer
            - "married": Married filing jointly
            - "head": Head of household
        state: Two-letter state code. Affects state tax calculations. Example: "CA", "TX", "FL"
        retirement_contributions: Dictionary of retirement account contributions by type.
            Keys must be from ACCOUNT_TYPES. Values are contribution amounts.
            Example: {"traditional_401k": 23000, "hsa": 4150}
        capital_gains_short: Short-term capital gains (taxed as ordinary income). Non-negative. Example: 5000
        capital_gains_long: Long-term capital gains (preferential rates). Non-negative. Example: 15000
        dividend_income_qualified: Qualified dividends (preferential rates). Non-negative. Example: 3000
        dividend_income_ordinary: Ordinary dividends (ordinary rates). Non-negative. Example: 1200
        itemized_deductions: Total itemized deductions (mortgage interest, SALT, charity, etc.).
            Non-negative. Example: 18000
        dependents: Number of dependent children/relatives. Must be 0-10. Example: 2
        self_employment_income: Net self-employment income. Can be negative. Example: 25000
        rental_income: Net rental income from properties. Can be negative. Example: 8000

    Returns:
        String with total tax liability, effective rate, and optimization suggestions.
        Format: "Total Tax: $32,450, Effective Rate: 18.2%, Marginal Rate: 24%, Suggestions: Max 401k, Consider Roth conversion"
    """
    # Validation
    if gross_income < 0:
        return "Error: Gross income cannot be negative"
    if filing_status not in TAX_BRACKETS:
        return "Error: Invalid filing status. Use: single, married, head"
    if capital_gains_short < 0 or capital_gains_long < 0:
        return "Error: Capital gains cannot be negative"
    if dividend_income_qualified < 0 or dividend_income_ordinary < 0:
        return "Error: Dividend income cannot be negative"
    if itemized_deductions < 0:
        return "Error: Itemized deductions cannot be negative"
    if dependents < 0 or dependents > 10:
        return "Error: Dependents must be 0-10"
    
    # Calculate AGI
    total_retirement = sum(retirement_contributions.values())
    se_deduction = self_employment_income * 0.0765 if self_employment_income > 0 else 0
    
    agi = (gross_income + capital_gains_short + dividend_income_ordinary + 
           self_employment_income + rental_income - total_retirement - se_deduction)
    
    # Standard deduction
    standard_deductions = {"single": 14600, "married": 29200, "head": 21900}
    standard_deduction = standard_deductions[filing_status]
    
    # Use greater of standard or itemized
    deduction = max(standard_deduction, itemized_deductions)
    
    # Taxable income
    taxable_income = max(0, agi - deduction)
    
    # Calculate ordinary income tax
    brackets = TAX_BRACKETS[filing_status]
    tax = 0
    prev_bracket = 0
    marginal_rate = 0
    
    for bracket_limit, rate in brackets:
        if taxable_income > prev_bracket:
            taxable_in_bracket = min(taxable_income, bracket_limit) - prev_bracket
            tax += taxable_in_bracket * rate
            marginal_rate = rate
            prev_bracket = bracket_limit
        else:
            break
    
    # Add long-term capital gains tax (simplified)
    if capital_gains_long > 0:
        if taxable_income < 44625:  # 0% bracket
            ltcg_tax = 0
        elif taxable_income < 492300:  # 15% bracket
            ltcg_tax = capital_gains_long * 0.15
        else:  # 20% bracket
            ltcg_tax = capital_gains_long * 0.20
        tax += ltcg_tax
    
    # Add qualified dividend tax (same as LTCG)
    if dividend_income_qualified > 0:
        if taxable_income < 44625:
            div_tax = 0
        elif taxable_income < 492300:
            div_tax = dividend_income_qualified * 0.15
        else:
            div_tax = dividend_income_qualified * 0.20
        tax += div_tax
    
    # Self-employment tax
    if self_employment_income > 0:
        se_tax = self_employment_income * 0.153
        tax += se_tax
    
    # Child tax credit
    child_credit = min(dependents * 2000, tax)
    tax -= child_credit
    
    # Calculate effective rate
    total_income = gross_income + capital_gains_short + capital_gains_long + dividend_income_qualified + dividend_income_ordinary + self_employment_income + rental_income
    effective_rate = tax / total_income if total_income > 0 else 0
    
    # Generate suggestions
    suggestions = []
    if total_retirement < 23000:
        suggestions.append("Max 401k")
    if marginal_rate >= 0.24 and "roth_ira" not in retirement_contributions:
        suggestions.append("Consider Roth conversion")
    if itemized_deductions < standard_deduction:
        suggestions.append("Bunch deductions")
    
    suggestion_str = ", ".join(suggestions) if suggestions else "Optimized"
    
    return f"Total Tax: ${tax:,.0f}, Effective Rate: {effective_rate*100:.1f}%, Marginal Rate: {marginal_rate*100:.0f}%, Suggestions: {suggestion_str}"



def calculate_retirement_readiness(
    current_age: int,
    retirement_age: int,
    current_savings: float,
    annual_contribution: float,
    employer_match_percent: float,
    expected_return: float,
    inflation_rate: float,
    desired_retirement_income: float,
    social_security_estimate: float,
    pension_income: float,
    healthcare_cost_annual: float,
    life_expectancy: int
) -> str:
    """
    Assess retirement readiness by projecting savings growth, comparing to income needs,
    factoring in Social Security, pensions, healthcare costs, and longevity.

    Args:
        current_age: Current age in years. Must be 18-80. Example: 45
        retirement_age: Target retirement age. Must be greater than current_age, max 75. Example: 67
        current_savings: Total current retirement savings. Non-negative. Example: 350000
        annual_contribution: Annual retirement contributions. Non-negative. Example: 25000
        employer_match_percent: Employer match as percentage of contribution. 0.0-1.0. Example: 0.50
        expected_return: Expected annual investment return as decimal. 0.03-0.12. Example: 0.07
        inflation_rate: Expected annual inflation rate as decimal. 0.01-0.05. Example: 0.03
        desired_retirement_income: Desired annual retirement income in today's dollars. Positive. Example: 80000
        social_security_estimate: Estimated annual Social Security benefit in today's dollars. Non-negative. Example: 28000
        pension_income: Annual pension income if applicable. Non-negative. Example: 15000
        healthcare_cost_annual: Estimated annual healthcare costs in retirement. Positive. Example: 12000
        life_expectancy: Expected age at death for planning. Must be > retirement_age. Example: 90

    Returns:
        String with projected savings, income gap/surplus, and readiness assessment.
        Format: "Projected Savings: $1,850,000, Annual Gap: $15,000, Readiness: 82%, Status: On Track"
    """
    if current_age < 18 or current_age > 80:
        return "Error: Current age must be 18-80"
    if retirement_age <= current_age or retirement_age > 75:
        return "Error: Retirement age must be > current age and <= 75"
    if current_savings < 0 or annual_contribution < 0:
        return "Error: Savings and contributions must be non-negative"
    if employer_match_percent < 0 or employer_match_percent > 1.0:
        return "Error: Employer match must be 0-100%"
    if expected_return < 0.03 or expected_return > 0.12:
        return "Error: Expected return must be 3%-12%"
    if inflation_rate < 0.01 or inflation_rate > 0.05:
        return "Error: Inflation rate must be 1%-5%"
    if desired_retirement_income <= 0 or healthcare_cost_annual <= 0:
        return "Error: Income and healthcare costs must be positive"
    if social_security_estimate < 0 or pension_income < 0:
        return "Error: SS and pension cannot be negative"
    if life_expectancy <= retirement_age:
        return "Error: Life expectancy must be > retirement age"
    
    # Project savings at retirement
    years_to_retirement = retirement_age - current_age
    total_contribution = annual_contribution * (1 + employer_match_percent)
    
    # Future value calculation
    fv_current = current_savings * ((1 + expected_return) ** years_to_retirement)
    fv_contributions = total_contribution * (((1 + expected_return) ** years_to_retirement - 1) / expected_return)
    projected_savings = fv_current + fv_contributions
    
    # Calculate retirement needs
    years_in_retirement = life_expectancy - retirement_age
    inflation_adjusted_income = desired_retirement_income * ((1 + inflation_rate) ** years_to_retirement)
    inflation_adjusted_healthcare = healthcare_cost_annual * ((1 + inflation_rate) ** years_to_retirement)
    
    annual_need = inflation_adjusted_income + inflation_adjusted_healthcare - social_security_estimate - pension_income
    total_need = annual_need * years_in_retirement
    
    # Calculate gap/surplus
    gap = total_need - projected_savings
    readiness_pct = (projected_savings / total_need * 100) if total_need > 0 else 100
    
    if readiness_pct >= 100:
        status = "On Track"
    elif readiness_pct >= 80:
        status = "Nearly There"
    elif readiness_pct >= 60:
        status = "Needs Improvement"
    else:
        status = "Behind"
    
    return f"Projected Savings: ${projected_savings:,.0f}, Annual Gap: ${annual_need:,.0f}, Readiness: {readiness_pct:.0f}%, Status: {status}"


def analyze_debt_payoff_strategy(
    debts: List[Dict[str, Union[float, str]]],
    monthly_payment_budget: float,
    strategy: str,
    extra_payment_allocation: str,
    interest_rate_threshold: float,
    consolidation_available: bool,
    consolidation_rate: float,
    balance_transfer_fee: float,
    credit_score_impact_weight: float
) -> str:
    """
    Analyze optimal debt payoff strategy across multiple debts considering various methods,
    consolidation options, balance transfers, and credit score impacts.

    Args:
        debts: List of debt dictionaries, each containing:
            - "balance": Outstanding balance (float, positive)
            - "rate": Annual interest rate (float, 0.01-0.30)
            - "minimum": Minimum monthly payment (float, positive)
            - "type": Debt type (str: "credit_card", "student_loan", "auto_loan", "personal_loan")
            Example: [{"balance": 5000, "rate": 0.18, "minimum": 150, "type": "credit_card"}]
        monthly_payment_budget: Total monthly amount available for debt payments. Positive. Example: 1200
        strategy: Payoff strategy. Must be one of:
            - "avalanche": Pay highest interest rate first
            - "snowball": Pay lowest balance first
            - "hybrid": Balance of both approaches
        extra_payment_allocation: How to allocate extra payments. Must be one of:
            - "single_focus": All extra to one debt
            - "proportional": Spread across all debts
            - "high_interest_only": Extra only to debts above threshold
        interest_rate_threshold: Interest rate threshold for high_interest_only strategy. 0.05-0.25. Example: 0.12
        consolidation_available: Whether debt consolidation loan is available. Boolean.
        consolidation_rate: Interest rate for consolidation loan if available. 0.04-0.15. Example: 0.08
        balance_transfer_fee: Fee for balance transfers as percentage. 0.0-0.05. Example: 0.03
        credit_score_impact_weight: Weight given to credit score impact (0.0-1.0). Higher = prioritize score. Example: 0.3

    Returns:
        String with payoff timeline, total interest paid, and recommended approach.
        Format: "Payoff Time: 38 months, Total Interest: $3,245, Strategy: Avalanche, Recommendation: Consolidate high-interest debts"
    """
    if not debts or len(debts) == 0:
        return "Error: Must provide at least one debt"
    if monthly_payment_budget <= 0:
        return "Error: Payment budget must be positive"
    if strategy not in ["avalanche", "snowball", "hybrid"]:
        return "Error: Invalid strategy"
    if extra_payment_allocation not in ["single_focus", "proportional", "high_interest_only"]:
        return "Error: Invalid extra payment allocation"
    if interest_rate_threshold < 0.05 or interest_rate_threshold > 0.25:
        return "Error: Interest rate threshold must be 5%-25%"
    if consolidation_rate < 0.04 or consolidation_rate > 0.15:
        return "Error: Consolidation rate must be 4%-15%"
    if balance_transfer_fee < 0 or balance_transfer_fee > 0.05:
        return "Error: Balance transfer fee must be 0%-5%"
    if credit_score_impact_weight < 0 or credit_score_impact_weight > 1.0:
        return "Error: Credit score weight must be 0.0-1.0"
    
    # Calculate total debt and minimum payments
    total_balance = sum(d["balance"] for d in debts)
    total_minimum = sum(d["minimum"] for d in debts)
    
    if monthly_payment_budget < total_minimum:
        return f"Error: Payment budget ${monthly_payment_budget:.0f} less than minimum ${total_minimum:.0f}"
    
    extra_payment = monthly_payment_budget - total_minimum
    
    # Sort debts based on strategy
    if strategy == "avalanche":
        sorted_debts = sorted(debts, key=lambda x: x["rate"], reverse=True)
    elif strategy == "snowball":
        sorted_debts = sorted(debts, key=lambda x: x["balance"])
    else:  # hybrid
        sorted_debts = sorted(debts, key=lambda x: (x["rate"] * 0.5 + (x["balance"]/total_balance) * 0.5), reverse=True)
    
    # Simulate payoff
    months = 0
    total_interest = 0
    remaining_debts = [d.copy() for d in sorted_debts]
    
    while remaining_debts and months < 600:  # Max 50 years
        months += 1
        extra_this_month = extra_payment
        
        for debt in remaining_debts[:]:
            monthly_rate = debt["rate"] / 12
            interest = debt["balance"] * monthly_rate
            total_interest += interest
            
            payment = debt["minimum"]
            if debt == remaining_debts[0] and extra_payment_allocation == "single_focus":
                payment += extra_this_month
                extra_this_month = 0
            
            principal = payment - interest
            debt["balance"] -= principal
            
            if debt["balance"] <= 0:
                remaining_debts.remove(debt)
    
    # Check if consolidation is better
    recommendation = f"Use {strategy}"
    if consolidation_available:
        consolidation_interest = total_balance * consolidation_rate * (months / 12)
        if consolidation_interest < total_interest * 0.8:
            recommendation = "Consolidate all debts"
    
    # Check for high-interest balance transfers
    high_interest_debts = [d for d in debts if d["rate"] > interest_rate_threshold]
    if high_interest_debts and balance_transfer_fee < 0.04:
        recommendation += ", Transfer high-interest balances"
    
    return f"Payoff Time: {months} months, Total Interest: ${total_interest:,.0f}, Strategy: {strategy.title()}, Recommendation: {recommendation}"



def calculate_insurance_needs(
    age: int,
    annual_income: float,
    dependents: int,
    mortgage_balance: float,
    other_debts: float,
    existing_coverage: Dict[str, float],
    health_status: str,
    occupation_risk: str,
    years_until_retirement: int,
    spouse_income: float,
    college_funding_need: float,
    final_expenses: float
) -> str:
    """
    Calculate comprehensive insurance needs including life, disability, and long-term care
    based on income replacement, debt coverage, dependent needs, and risk factors.

    Args:
        age: Current age in years. Must be 18-80. Example: 42
        annual_income: Gross annual income to replace. Positive. Example: 125000
        dependents: Number of dependents relying on income. 0-10. Example: 3
        mortgage_balance: Outstanding mortgage balance. Non-negative. Example: 285000
        other_debts: Other debts (car, student, credit card). Non-negative. Example: 35000
        existing_coverage: Dictionary of existing insurance by type with coverage amounts.
            Keys: "life", "disability", "ltc" (long-term care). Values: coverage amounts.
            Example: {"life": 250000, "disability": 0, "ltc": 0}
        health_status: Current health status. Must be one of:
            - "excellent": No health issues
            - "good": Minor manageable conditions
            - "fair": Some chronic conditions
            - "poor": Significant health issues
        occupation_risk: Occupation risk level. Must be one of:
            - "low": Office/desk work
            - "medium": Mixed physical/mental
            - "high": Physical labor, hazardous
        years_until_retirement: Years until retirement. 1-40. Example: 23
        spouse_income: Spouse's annual income if applicable. Non-negative. Example: 75000
        college_funding_need: Total college funding needed for dependents. Non-negative. Example: 150000
        final_expenses: Estimated funeral and final expenses. Positive. Example: 15000

    Returns:
        String with recommended coverage amounts and estimated premiums for each insurance type.
        Format: "Life: $850,000 ($95/mo), Disability: $6,250/mo ($180/mo), LTC: $4,500/mo ($220/mo), Total Premium: $495/mo"
    """
    if age < 18 or age > 80:
        return "Error: Age must be 18-80"
    if annual_income <= 0:
        return "Error: Annual income must be positive"
    if dependents < 0 or dependents > 10:
        return "Error: Dependents must be 0-10"
    if mortgage_balance < 0 or other_debts < 0:
        return "Error: Debts cannot be negative"
    if health_status not in ["excellent", "good", "fair", "poor"]:
        return "Error: Invalid health status"
    if occupation_risk not in ["low", "medium", "high"]:
        return "Error: Invalid occupation risk"
    if years_until_retirement < 1 or years_until_retirement > 40:
        return "Error: Years until retirement must be 1-40"
    if spouse_income < 0 or college_funding_need < 0:
        return "Error: Spouse income and college funding cannot be negative"
    if final_expenses <= 0:
        return "Error: Final expenses must be positive"
    
    # Calculate life insurance need (DIME method + adjustments)
    debt_coverage = mortgage_balance + other_debts
    income_replacement = annual_income * min(years_until_retirement, 10)
    dependent_expenses = dependents * 20000 * min(years_until_retirement, 18)
    
    life_need = debt_coverage + income_replacement + college_funding_need + final_expenses + dependent_expenses
    
    # Adjust for spouse income
    if spouse_income > 0:
        life_need *= 0.75  # Reduce need if spouse has income
    
    # Calculate gap
    existing_life = existing_coverage.get("life", 0)
    life_gap = max(0, life_need - existing_life)
    
    # Calculate life insurance premium
    life_premium_monthly = (life_gap / 1000) * age * INSURANCE_FACTORS["life"]["age_mult"]
    life_premium_monthly *= INSURANCE_FACTORS["life"]["health_mult"][health_status]
    
    # Calculate disability insurance need (60-70% of income)
    disability_monthly_need = (annual_income * 0.65) / 12
    existing_disability = existing_coverage.get("disability", 0)
    disability_gap = max(0, disability_monthly_need - existing_disability)
    
    # Calculate disability premium
    disability_premium_monthly = disability_gap * INSURANCE_FACTORS["disability"]["age_mult"]
    disability_premium_monthly *= INSURANCE_FACTORS["disability"]["occupation_mult"][occupation_risk]
    
    # Calculate long-term care need (if age > 50)
    ltc_monthly_need = 0
    ltc_premium_monthly = 0
    if age >= 50:
        ltc_monthly_need = 4500  # Average LTC cost
        existing_ltc = existing_coverage.get("ltc", 0)
        ltc_gap = max(0, ltc_monthly_need - existing_ltc)
        
        ltc_premium_monthly = ltc_gap * INSURANCE_FACTORS["ltc"]["age_mult"] * (age / 100)
        ltc_premium_monthly *= INSURANCE_FACTORS["ltc"]["health_mult"][health_status]
    
    total_premium = life_premium_monthly + disability_premium_monthly + ltc_premium_monthly
    
    result = f"Life: ${life_gap:,.0f} (${life_premium_monthly:.0f}/mo), Disability: ${disability_gap:,.0f}/mo (${disability_premium_monthly:.0f}/mo)"
    if age >= 50:
        result += f", LTC: ${ltc_gap:,.0f}/mo (${ltc_premium_monthly:.0f}/mo)"
    result += f", Total Premium: ${total_premium:.0f}/mo"
    
    return result


def calculate_education_funding(
    child_current_age: int,
    college_start_age: int,
    years_of_college: int,
    current_annual_cost: float,
    education_inflation_rate: float,
    current_savings: float,
    monthly_contribution: float,
    expected_return: float,
    financial_aid_expected: float,
    student_contribution_percent: float,
    state_residency: str,
    school_type: str
) -> str:
    """
    Calculate education funding needs and project savings for college expenses considering
    inflation, investment returns, financial aid, student contributions, and school type.

    Args:
        child_current_age: Child's current age in years. Must be 0-17. Example: 8
        college_start_age: Age when college starts. Must be > current_age, typically 18. Example: 18
        years_of_college: Number of years of college. 2-6. Example: 4
        current_annual_cost: Current annual cost of target school. Positive. Example: 35000
        education_inflation_rate: Annual education cost inflation rate. 0.03-0.08. Example: 0.05
        current_savings: Current 529/education savings. Non-negative. Example: 25000
        monthly_contribution: Monthly contribution to education savings. Non-negative. Example: 500
        expected_return: Expected annual investment return. 0.04-0.10. Example: 0.07
        financial_aid_expected: Expected annual financial aid/scholarships. Non-negative. Example: 8000
        student_contribution_percent: Percentage of costs student will cover (work/loans). 0.0-0.50. Example: 0.15
        state_residency: State of residency for in-state tuition. Two-letter code. Example: "CA"
        school_type: Type of school. Must be one of:
            - "public_instate": Public university, in-state tuition
            - "public_outstate": Public university, out-of-state tuition
            - "private": Private university
            - "community": Community college

    Returns:
        String with projected savings, total cost, funding gap/surplus, and monthly contribution needed.
        Format: "Total Cost: $185,000, Projected Savings: $142,000, Gap: $43,000, Additional Monthly Needed: $285"
    """
    if child_current_age < 0 or child_current_age > 17:
        return "Error: Child age must be 0-17"
    if college_start_age <= child_current_age:
        return "Error: College start age must be > current age"
    if years_of_college < 2 or years_of_college > 6:
        return "Error: Years of college must be 2-6"
    if current_annual_cost <= 0:
        return "Error: Annual cost must be positive"
    if education_inflation_rate < 0.03 or education_inflation_rate > 0.08:
        return "Error: Education inflation must be 3%-8%"
    if current_savings < 0 or monthly_contribution < 0:
        return "Error: Savings and contributions cannot be negative"
    if expected_return < 0.04 or expected_return > 0.10:
        return "Error: Expected return must be 4%-10%"
    if financial_aid_expected < 0:
        return "Error: Financial aid cannot be negative"
    if student_contribution_percent < 0 or student_contribution_percent > 0.50:
        return "Error: Student contribution must be 0%-50%"
    if school_type not in ["public_instate", "public_outstate", "private", "community"]:
        return "Error: Invalid school type"
    
    # Adjust cost based on school type
    cost_multipliers = {
        "public_instate": 1.0,
        "public_outstate": 1.8,
        "private": 2.2,
        "community": 0.4
    }
    adjusted_annual_cost = current_annual_cost * cost_multipliers[school_type]
    
    # Calculate years until college
    years_until_college = college_start_age - child_current_age
    
    # Project savings at college start
    fv_current = current_savings * ((1 + expected_return) ** years_until_college)
    fv_contributions = monthly_contribution * 12 * (((1 + expected_return) ** years_until_college - 1) / expected_return)
    projected_savings_at_start = fv_current + fv_contributions
    
    # Calculate total college cost (inflated)
    total_cost = 0
    for year in range(years_of_college):
        year_cost = adjusted_annual_cost * ((1 + education_inflation_rate) ** (years_until_college + year))
        year_cost -= financial_aid_expected  # Subtract aid
        year_cost *= (1 - student_contribution_percent)  # Subtract student contribution
        total_cost += year_cost
    
    # Calculate gap
    gap = total_cost - projected_savings_at_start
    
    # Calculate additional monthly contribution needed if gap exists
    if gap > 0 and years_until_college > 0:
        # Calculate monthly payment needed to close gap
        additional_monthly = (gap * expected_return) / (((1 + expected_return) ** years_until_college - 1) * 12)
    else:
        additional_monthly = 0
    
    if gap > 0:
        return f"Total Cost: ${total_cost:,.0f}, Projected Savings: ${projected_savings_at_start:,.0f}, Gap: ${gap:,.0f}, Additional Monthly Needed: ${additional_monthly:.0f}"
    else:
        surplus = abs(gap)
        return f"Total Cost: ${total_cost:,.0f}, Projected Savings: ${projected_savings_at_start:,.0f}, Surplus: ${surplus:,.0f}, On Track"


def calculate_currency_exchange_arbitrage(
    base_currency: str,
    target_currency: str,
    amount: float,
    exchange_method: str,
    transfer_fee_percent: float,
    transfer_fee_fixed: float,
    intermediate_currency: Optional[str],
    spot_rate_override: Optional[float],
    forward_contract_months: int,
    hedging_strategy: str,
    tax_reporting_required: bool
) -> str:
    """
    Calculate optimal currency exchange considering direct vs. intermediate routes, fees,
    forward contracts, hedging strategies, and tax implications.

    Args:
        base_currency: Source currency code (3-letter ISO). Must be in EXCHANGE_RATES. Example: "USD"
        target_currency: Destination currency code (3-letter ISO). Must be in EXCHANGE_RATES. Example: "EUR"
        amount: Amount to exchange in base currency. Positive. Example: 50000
        exchange_method: Exchange method. Must be one of:
            - "bank": Traditional bank transfer
            - "forex_broker": Forex broker
            - "crypto_bridge": Cryptocurrency intermediary
            - "wire": International wire transfer
        transfer_fee_percent: Percentage fee for transfer. 0.0-0.05. Example: 0.015
        transfer_fee_fixed: Fixed fee for transfer in base currency. Non-negative. Example: 25
        intermediate_currency: Optional intermediate currency for triangular arbitrage. Example: "GBP"
        spot_rate_override: Optional custom spot rate. If None, uses market rate. Example: 0.93
        forward_contract_months: Months for forward contract (0 for spot). 0-24. Example: 6
        hedging_strategy: Hedging approach. Must be one of:
            - "none": No hedging
            - "forward": Lock in forward rate
            - "option": Currency option
            - "collar": Option collar strategy
        tax_reporting_required: Whether transaction requires tax reporting (>$10k). Boolean.

    Returns:
        String with net amount received, effective rate, and cost breakdown.
        Format: "Received: €46,250, Effective Rate: 0.925, Total Cost: $1,250 (Fee: $775, Spread: $475), Method: Direct"
    """
    if base_currency not in EXCHANGE_RATES or target_currency not in EXCHANGE_RATES:
        return f"Error: Invalid currency. Supported: {', '.join(EXCHANGE_RATES.keys())}"
    if amount <= 0:
        return "Error: Amount must be positive"
    if exchange_method not in ["bank", "forex_broker", "crypto_bridge", "wire"]:
        return "Error: Invalid exchange method"
    if transfer_fee_percent < 0 or transfer_fee_percent > 0.05:
        return "Error: Transfer fee percent must be 0%-5%"
    if transfer_fee_fixed < 0:
        return "Error: Fixed fee cannot be negative"
    if intermediate_currency and intermediate_currency not in EXCHANGE_RATES:
        return "Error: Invalid intermediate currency"
    if forward_contract_months < 0 or forward_contract_months > 24:
        return "Error: Forward contract months must be 0-24"
    if hedging_strategy not in ["none", "forward", "option", "collar"]:
        return "Error: Invalid hedging strategy"
    
    # Get exchange rates
    base_to_usd = EXCHANGE_RATES[base_currency]
    target_to_usd = EXCHANGE_RATES[target_currency]
    
    if spot_rate_override:
        direct_rate = spot_rate_override
    else:
        direct_rate = target_to_usd / base_to_usd
    
    # Apply spread based on method
    spreads = {"bank": 0.02, "forex_broker": 0.005, "crypto_bridge": 0.015, "wire": 0.018}
    spread = spreads[exchange_method]
    effective_rate = direct_rate * (1 - spread)
    
    # Calculate direct exchange
    gross_amount = amount * effective_rate
    fee_percent = amount * transfer_fee_percent
    fee_total = fee_percent + transfer_fee_fixed
    net_direct = gross_amount - (fee_total * effective_rate)
    
    # Check triangular arbitrage if intermediate currency specified
    if intermediate_currency:
        inter_to_usd = EXCHANGE_RATES[intermediate_currency]
        rate1 = inter_to_usd / base_to_usd
        rate2 = target_to_usd / inter_to_usd
        triangular_rate = rate1 * rate2 * (1 - spread) * (1 - spread)  # Double spread
        gross_triangular = amount * triangular_rate
        net_triangular = gross_triangular - (fee_total * 2 * triangular_rate)  # Double fees
        
        if net_triangular > net_direct:
            net_amount = net_triangular
            method = f"Triangular via {intermediate_currency}"
            effective_rate = triangular_rate
        else:
            net_amount = net_direct
            method = "Direct"
    else:
        net_amount = net_direct
        method = "Direct"
    
    # Apply forward contract adjustment
    if forward_contract_months > 0:
        # Forward points (simplified)
        forward_adjustment = forward_contract_months * 0.001
        effective_rate *= (1 + forward_adjustment)
        net_amount *= (1 + forward_adjustment)
    
    # Hedging cost
    hedging_cost = 0
    if hedging_strategy == "forward":
        hedging_cost = amount * 0.002
    elif hedging_strategy == "option":
        hedging_cost = amount * 0.015
    elif hedging_strategy == "collar":
        hedging_cost = amount * 0.008
    
    net_amount -= hedging_cost * effective_rate
    
    # Calculate total cost
    spread_cost = amount * spread
    total_cost = fee_total + spread_cost + hedging_cost
    
    # Format currency symbol
    currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "CHF", "CAD": "C$", "AUD": "A$", "CNY": "¥", "INR": "₹", "BRL": "R$"}
    target_symbol = currency_symbols.get(target_currency, target_currency)
    
    return f"Received: {target_symbol}{net_amount:,.0f}, Effective Rate: {effective_rate:.4f}, Total Cost: ${total_cost:,.0f} (Fee: ${fee_total:.0f}, Spread: ${spread_cost:.0f}, Hedge: ${hedging_cost:.0f}), Method: {method}"




# =============================================================================
# TOOL RUNNER
# =============================================================================

# =============================================================================
# TOOL FUNCTIONS EXPORT LIST
# =============================================================================
# This list is used by the GRPO trainer for dynamic tool function loading.
# Add new tool functions to this list to make them available for training.

TOOL_FUNCTIONS = [
    calculate_portfolio_allocation,
    calculate_mortgage_affordability,
    optimize_tax_strategy,
    calculate_retirement_readiness,
    analyze_debt_payoff_strategy,
    calculate_insurance_needs,
    calculate_education_funding,
    calculate_currency_exchange_arbitrage,
]


TOOLS = {
    "calculate_portfolio_allocation": calculate_portfolio_allocation,
    "calculate_mortgage_affordability": calculate_mortgage_affordability,
    "optimize_tax_strategy": optimize_tax_strategy,
    "calculate_retirement_readiness": calculate_retirement_readiness,
    "analyze_debt_payoff_strategy": analyze_debt_payoff_strategy,
    "calculate_insurance_needs": calculate_insurance_needs,
    "calculate_education_funding": calculate_education_funding,
    "calculate_currency_exchange_arbitrage": calculate_currency_exchange_arbitrage,
}


def run_tool(tool_call: Union[str, dict]) -> str:
    """
    Execute a tool based on a JSON tool call specification.
    
    Args:
        tool_call: Either a JSON string or dict with format:
                   {"name": "function_name", "arguments": {...}}
    
    Returns:
        Simple string result for training data comparison.
    """
    if isinstance(tool_call, str):
        try:
            tool_call = json.loads(tool_call)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON - {str(e)}"
    
    if not isinstance(tool_call, dict):
        return "Error: Tool call must be a dictionary"
    if "name" not in tool_call:
        return "Error: Tool call must have 'name' field"
    
    tool_name = tool_call["name"]
    arguments = tool_call.get("arguments", {})
    
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'. Available: {', '.join(TOOLS.keys())}"
    
    try:
        return TOOLS[tool_name](**arguments)
    except TypeError as e:
        return f"Error: Invalid arguments - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TEST CASES
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FINANCIAL TOOLS - COMPLEX ARGUMENT STRUCTURES")
    print("=" * 80)
    
    # Test 1: Portfolio Allocation
    print("\n[TEST 1] Calculate Portfolio Allocation")
    test1 = {
        "name": "calculate_portfolio_allocation",
        "arguments": {
            "total_investment": 500000,
            "risk_tolerance": "moderate_aggressive",
            "time_horizon_years": 25,
            "current_age": 45,
            "retirement_age": 65,
            "income_need_percentage": 0.04,
            "existing_allocations": {"equity_us_large": 0.30, "fixed_income_govt": 0.20},
            "esg_preference": True,
            "tax_loss_harvesting": True,
            "rebalancing_frequency": "quarterly",
            "inflation_protection": True,
            "liquidity_requirement": "long_term"
        }
    }
    print(f"Answer: {run_tool(test1)}")
    
    # Test 2: Mortgage Affordability
    print("\n[TEST 2] Calculate Mortgage Affordability")
    test2 = {
        "name": "calculate_mortgage_affordability",
        "arguments": {
            "annual_income": 120000,
            "monthly_debts": 850,
            "down_payment": 60000,
            "credit_score": 740,
            "loan_type": "conventional",
            "property_state": "CA",
            "property_tax_rate": 0.012,
            "hoa_fees": 250,
            "homeowners_insurance": 150,
            "pmi_required": True,
            "interest_rate_override": None,
            "loan_term_years": 30
        }
    }
    print(f"Answer: {run_tool(test2)}")
    
    # Test 3: Tax Strategy
    print("\n[TEST 3] Optimize Tax Strategy")
    test3 = {
        "name": "optimize_tax_strategy",
        "arguments": {
            "gross_income": 150000,
            "filing_status": "married",
            "state": "CA",
            "retirement_contributions": {"traditional_401k": 23000, "hsa": 4150},
            "capital_gains_short": 5000,
            "capital_gains_long": 15000,
            "dividend_income_qualified": 3000,
            "dividend_income_ordinary": 1200,
            "itemized_deductions": 18000,
            "dependents": 2,
            "self_employment_income": 25000,
            "rental_income": 8000
        }
    }
    print(f"Answer: {run_tool(test3)}")
    
    # Test 4: Retirement Readiness
    print("\n[TEST 4] Calculate Retirement Readiness")
    test4 = {
        "name": "calculate_retirement_readiness",
        "arguments": {
            "current_age": 45,
            "retirement_age": 67,
            "current_savings": 350000,
            "annual_contribution": 25000,
            "employer_match_percent": 0.50,
            "expected_return": 0.07,
            "inflation_rate": 0.03,
            "desired_retirement_income": 80000,
            "social_security_estimate": 28000,
            "pension_income": 15000,
            "healthcare_cost_annual": 12000,
            "life_expectancy": 90
        }
    }
    print(f"Answer: {run_tool(test4)}")
    
    # Test 5: Debt Payoff
    print("\n[TEST 5] Analyze Debt Payoff Strategy")
    test5 = {
        "name": "analyze_debt_payoff_strategy",
        "arguments": {
            "debts": [
                {"balance": 5000, "rate": 0.18, "minimum": 150, "type": "credit_card"},
                {"balance": 15000, "rate": 0.06, "minimum": 200, "type": "auto_loan"},
                {"balance": 8000, "rate": 0.22, "minimum": 180, "type": "credit_card"}
            ],
            "monthly_payment_budget": 1200,
            "strategy": "avalanche",
            "extra_payment_allocation": "single_focus",
            "interest_rate_threshold": 0.12,
            "consolidation_available": True,
            "consolidation_rate": 0.08,
            "balance_transfer_fee": 0.03,
            "credit_score_impact_weight": 0.3
        }
    }
    print(f"Answer: {run_tool(test5)}")
    
    print("\n" + "=" * 80)

