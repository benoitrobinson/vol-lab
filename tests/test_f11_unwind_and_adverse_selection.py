"""F11 and F12: what the two missing frictions do to the skew's advantage.

F11. The idealised model lets the dealer go home flat for nothing. That is the
only reason the never-skewed control keeps up on mean P&L: it ends the session
holding four times the inventory and pays nothing to get out of it. Charge a
half-spread to unwind and the ranking flips.

F12. Informed flow costs every strategy the same. Leaning quotes against
inventory is protection against inventory risk, not against being picked off:
the quoting rule cannot see information, so it cannot avoid it. The dispersion
advantage survives adverse selection untouched, and the mean does not.

Measured on 8,000 sessions, seed 4242, phi = 0.3, unwind 0.5 per unit plus
0.005 per unit squared:

    setting        strategy     pnl     sd    sd/ctrl  markout   unwind paid
    frictionless   symmetric   67.978  13.217  1.000    0.0001   0.00
    frictionless   glft        67.738   7.080  0.536    0.0001   0.00
    unwind 0.5     symmetric   64.302  13.484  1.000    0.0001   3.68
    unwind 0.5     glft        66.853   7.079  0.525    0.0001   0.89
    informed 0.3   symmetric   64.778  13.284  1.000   -0.0333   0.00
    informed 0.3   glft        64.208   6.959  0.524   -0.0340   0.00

    glft minus control, paired: -0.240 [-0.451, -0.017] frictionless
                                +2.550 [+2.325, +2.785] with the unwind charged
"""

import numpy as np
import pytest

# Research sweep: see the slow marker in pyproject.toml.
pytestmark = pytest.mark.slow

from vollab.metrics.bootstrap import paired_bootstrap
from vollab.mm.quoting import DealerParams, MarketParams
from vollab.mm.simulate import simulate_mm

SEED, PATHS = 4242, 8000
STRATEGIES = ("symmetric", "avellaneda_stoikov", "glft")


def _runs(phi, liq_cost, liq_impact):
    market = MarketParams(phi=phi)
    dealer = DealerParams(liq_cost=liq_cost, liq_impact=liq_impact)
    return {s: simulate_mm(market, dealer, s, seed=SEED, n_paths=PATHS) for s in STRATEGIES}


@pytest.fixture(scope="module")
def settings():
    return {
        "frictionless": _runs(0.0, 0.0, 0.0),
        "unwind": _runs(0.0, 0.5, 0.005),
        "informed": _runs(0.3, 0.0, 0.0),
        "both": _runs(0.3, 0.5, 0.005),
    }


def test_the_free_unwind_is_what_kept_the_control_competitive(settings):
    """F11. Frictionless, the control's mean P&L is not worse than the skew's.
    Charge for the unwind and the skew wins, with the interval clear of zero."""
    free = settings["frictionless"]
    free_diff, _, free_hi = paired_bootstrap(free["glft"].pnl, free["symmetric"].pnl,
                                                   n_boot=2000)
    assert free_hi < 0.5, "frictionless, the skew has no mean advantage to speak of"

    charged = settings["unwind"]
    diff, lo, _ = paired_bootstrap(charged["glft"].pnl, charged["symmetric"].pnl, n_boot=2000)
    assert diff > 2.0
    assert lo > 0.0
    assert diff > free_diff + 2.0


def test_the_control_pays_most_of_the_unwind(settings):
    paid = {s: r.liq_paid.mean() for s, r in settings["unwind"].items()}
    assert paid["symmetric"] > 3 * paid["glft"]
    assert paid["symmetric"] > 2 * paid["avellaneda_stoikov"]


def test_the_dispersion_advantage_survives_both_frictions(settings):
    """F7 measured the halving on the idealised model. It is not an artefact of
    the free unwind or of uninformed flow: it is there in all four settings."""
    for name, runs in settings.items():
        control = runs["symmetric"].pnl.std(ddof=1)
        for skewed in ("avellaneda_stoikov", "glft"):
            ratio = runs[skewed].pnl.std(ddof=1) / control
            assert ratio < 0.6, f"{name}/{skewed} dispersion ratio {ratio:.3f}"


def test_adverse_selection_costs_every_strategy_the_same(settings):
    """F12, and it is a negative result worth stating: inventory skew is not
    protection against informed flow. The quoting rule cannot see information,
    so all three strategies mark out identically."""
    marks = np.array([settings["informed"][s].markout_per_fill() for s in STRATEGIES])
    assert (marks < -0.02).all()
    assert np.ptp(marks) < 0.1 * abs(marks.mean())


def test_informed_flow_takes_the_same_toll_from_the_mean_everywhere(settings):
    tolls = {s: settings["frictionless"][s].pnl.mean() - settings["informed"][s].pnl.mean()
             for s in STRATEGIES}
    assert all(t > 2.0 for t in tolls.values())
    assert max(tolls.values()) - min(tolls.values()) < 1.0


def test_with_both_frictions_the_skew_still_pays(settings):
    both = settings["both"]
    diff, lo, _ = paired_bootstrap(both["glft"].pnl, both["symmetric"].pnl, n_boot=2000)
    assert lo > 0.0
    assert diff > 1.5
