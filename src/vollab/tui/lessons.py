"""The findings, as readable lessons.

Each entry pairs a short explanation with the numbers the report generated, so
the lab teaches the result rather than only displaying it. Text is written once
here and used by the TUI; the numbers always come from artifacts/findings.json,
never from prose, so a lesson cannot quote a figure the code did not produce.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Lesson:
    key: str
    title: str
    question: str
    mechanism: str
    takeaway: str


LESSONS = [
    Lesson(
        key="f1_discretisation",
        title="1. Hedging error and frequency",
        question="If I hedge twice as often, how much less risk do I carry?",
        mechanism=(
            "Freeze gamma over a step and write dS/S = s*sqrt(dt)*z. One interval "
            "contributes 0.5*Gamma*S^2*s^2*dt*(z^2 - 1), which has mean zero and, "
            "because Var(z^2 - 1) = 2, variance proportional to dt^2. Summing T/dt "
            "of them gives total variance proportional to 1/N."
        ),
        takeaway=(
            "Dispersion falls as the inverse square root of frequency, so halving "
            "your risk costs four times the trading. This is also the engine's "
            "primary correctness test: a sign error or a misaligned shift moves "
            "the fitted slope off -0.5."
        ),
    ),
    Lesson(
        key="f2_lockin",
        title="2. Hedging at implied against realized",
        question="I sold vol at 35 and it realizes 25. Which volatility do I hedge at?",
        mechanism=(
            "Hedge at realized and the terminal P&L converges to BS(s_imp) - "
            "BS(s_real), a number known at inception; what stays random is the "
            "path you take to reach it. Hedge at implied and the P&L becomes "
            "genuinely path dependent, earning the same amount on average."
        ),
        takeaway=(
            "Same expected profit, several times the dispersion. The pathwise sign "
            "follows s_imp - s_real: a short position profits when realized comes "
            "in below implied, which is why every path here is profitable rather "
            "than merely most of them."
        ),
    ),
    Lesson(
        key="f3_attribution",
        title="3. Where the P&L actually comes from",
        question="My book made money. Which greek earned it?",
        mechanism=(
            "Decompose each step into delta, gamma, theta, vega, carry and cost "
            "terms, and whatever is left is the residual. The expansion is exact "
            "to second order in dS, so under diffusion the residual is third order."
        ),
        takeaway=(
            "Gamma and theta nearly cancel: you are paid time decay to carry short "
            "convexity. Watch the delta column, though. Hedging every step at the "
            "mark volatility makes that term zero by construction, not by "
            "measurement, so a near-zero figure there establishes nothing."
        ),
    ),
    Lesson(
        key="f4_jump_floor",
        title="4. Gap risk is not a frequency problem",
        question="Can I hedge my way out of a jump?",
        mechanism=(
            "Between jumps the diffusive error hedges away exactly as in lesson 1. "
            "Each Poisson jump delivers a convexity loss that arrives between "
            "rehedges however close together they are, and the variance it "
            "contributes depends on intensity and jump size, not on frequency."
        ),
        takeaway=(
            "Total variance converges to a floor instead of to zero. Hedging 226 "
            "times more often cuts diffusive error by a factor of eleven and jump "
            "error by about 1.6. For a crypto book this is the finding that "
            "matters: no rehedging discipline converts gap risk into something else."
        ),
    ),
    Lesson(
        key="f5_schedules",
        title="5. Under costs, where you trade beats how often",
        question="Costs are 10 bps. What is my hedging rule?",
        mechanism=(
            "Hedging on a clock spreads trades evenly across the calendar. Hedging "
            "on a band spends them where gamma is large, which is where the hedge "
            "is actually doing work."
        ),
        takeaway=(
            "The band rules reach a better mean on far less turnover. On a pure "
            "frequency sweep the risk-adjusted objective is U-shaped, but the "
            "bootstrap interval on the location of that minimum is wide: the "
            "optimum is a region, not a number."
        ),
    ),
    Lesson(
        key="f6_surface",
        title="6. A fitted smile can imply negative probabilities",
        question="My SVI fit has a low RMSE. Is it a set of prices?",
        mechanism=(
            "Durrleman's condition is non-negative exactly when the slice implies "
            "a non-negative risk-neutral density. Fitting freely and checking "
            "afterwards is not the same as constraining the fit."
        ),
        takeaway=(
            "Ordinary quote noise pushes a substantial fraction of unconstrained "
            "fits into negative density, and such a slice prices a butterfly at a "
            "negative number. Imposing the constraint during the fit removes every "
            "violation for a few thousandths of a vol point, far below any spread."
        ),
    ),
    Lesson(
        key="f7_market_making",
        title="7. Quoting against inventory",
        question="I am long 10 lots. Where do I quote?",
        mechanism=(
            "Centre quotes on a reservation price that leans against inventory: a "
            "long dealer quotes a closer ask and a further bid. Avellaneda-Stoikov "
            "solves a finite-horizon problem and widens as the horizon lengthens; "
            "the Gueant-Lehalle-Fernandez-Tapia form is the steady state and does "
            "not."
        ),
        takeaway=(
            "Skewing roughly halves P&L dispersion against a never-skewed control "
            "quoting the same width. The horizon term costs fills for no risk "
            "benefit, which is why the steady-state form keeps improving as risk "
            "aversion rises while the classical one collapses."
        ),
    ),
    Lesson(
        key="f8_inverse",
        title="8. Coin-settled options need their own delta",
        question="Deribit options settle in BTC. Does that change my hedge?",
        mechanism=(
            "Multiplying the coin payoff by the settlement price recovers the "
            "vanilla dollar payoff, so the coin price is just the vanilla value "
            "over spot. Differentiating that gives Delta/S - C/S^2, and the second "
            "term does not vanish."
        ),
        takeaway=(
            "Converting a vanilla delta by dividing by spot drops that term, an "
            "error equal to the option's own value over spot squared and largest "
            "where the option is worth most. Also: the expectation of the coin "
            "payoff under the dollar measure is not the coin price."
        ),
    ),
    Lesson(
        key="f9_rough_vol_floor",
        title="9. Vol-of-vol floors the hedging error",
        question="If I rehedge fast enough, does the risk go away?",
        mechanism=(
            "A delta hedge removes the dS term. Under stochastic volatility the "
            "option also moves with the variance, and no amount of trading the "
            "underlying removes that. Refining the grid drives the discretisation "
            "term to zero and leaves the vega term, whose variance does not depend "
            "on how often you trade."
        ),
        takeaway=(
            "The error stalls at a floor set by eta, the volatility of volatility. "
            "Roughness H does not move the floor: a nearly smooth variance floors in "
            "the same place. That is worth knowing, because roughness is usually "
            "introduced as a hedging story and it is really a smile story."
        ),
    ),
    Lesson(
        key="f10_rough_skew",
        title="10. Rough variance and the short-dated skew",
        question="Why does the one-week skew steepen faster than my model says?",
        mechanism=(
            "When the variance is a diffusion its increments over a week are tiny, "
            "so the at-the-money skew flattens out as maturity shrinks. A Volterra "
            "kernel with exponent H - 1/2 lets the variance move enough at short "
            "horizons that the skew scales as T^(H - 1/2) instead."
        ),
        takeaway=(
            "Measured against a Heston control with the same correlation and initial "
            "variance: the rough model's skew follows a power law close to the "
            "theoretical exponent, and Heston's is nearly flat. For anyone quoting "
            "weeklies that is the difference between a surface that fits and one "
            "that cannot."
        ),
    ),
    Lesson(
        key="f11_unwind",
        title="11. What it costs to go home flat",
        question="My backtest marks leftover inventory at the mid. Is that fair?",
        mechanism=(
            "The mid is the one price at which nobody trades. A dealer closing a "
            "position crosses the spread and moves the market it is crossing, and "
            "the strategy left holding the most inventory is the one that subsidy "
            "flatters most. Here that is the never-skewed control."
        ),
        takeaway=(
            "Charge a half-spread plus a quadratic impact term and the mean-P&L "
            "comparison that straddled zero turns decisive for the steady-state "
            "quoter. The dispersion advantage does not move, so finding 7 was real "
            "and was also measured in a world too kind to its control."
        ),
    ),
    Lesson(
        key="f12_adverse_selection",
        title="12. Adverse selection does not care how you quote",
        question="Does leaning against inventory protect me from informed flow?",
        mechanism=(
            "Inventory skew is a function of the position, and the position is a "
            "consequence of past fills. Information is a property of the next fill. "
            "Making a fraction of arrivals informed, so they trade just before the "
            "mid moves, leaves every quoting rule marking out the same amount."
        ),
        takeaway=(
            "No. The markout per fill falls by the same amount whatever the rule, "
            "and the P&L toll is the same within noise. Protection has to come from "
            "reading the flow, quoting wider when it looks informed, or standing "
            "aside, none of which this model can do."
        ),
    ),
]

BY_KEY = {lesson.key: lesson for lesson in LESSONS}
