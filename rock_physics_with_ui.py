
[ ]
import numpy as np
import matplotlib.pyplot as plt

MINERALS = {
    "quartz":  dict(K=36.6, Mu=45.0, rho=2.65),
    "clay":    dict(K=21.0, Mu=7.0,  rho=2.58),
    "calcite": dict(K=76.8, Mu=32.0, rho=2.71),
}
# Order (VSAND, VSH, VCARB)
MINERAL_ORDER = ["quartz", "clay", "calcite"]
K_MIN   = np.array([MINERALS[m]["K"]   for m in MINERAL_ORDER])
MU_MIN  = np.array([MINERALS[m]["Mu"]  for m in MINERAL_ORDER])
RHO_MIN = np.array([MINERALS[m]["rho"] for m in MINERAL_ORDER])


def vrh_average(fractions, K, Mu):
    '''Voigt-Reuss-Hill average of bulk (K) and shear (Mu) moduli.

    Parameters
    ----------
    fractions : array_like, shape (..., n_minerals)
        Volume fractions of each mineral end-member (should sum to 1 along the last axis).
    K, Mu : array_like, shape (n_minerals,)
        Bulk and shear modulus of each mineral end-member (GPa).

    Returns
    -------
    K_vrh, Mu_vrh : ndarray
        Voigt-Reuss-Hill averaged bulk and shear modulus (GPa), broadcasting over any
        leading dimensions of `fractions`.
    '''
    fractions = np.asarray(fractions, dtype=float)
    K = np.asarray(K, dtype=float)
    Mu = np.asarray(Mu, dtype=float)

    K_voigt = np.sum(fractions * K, axis=-1)
    K_reuss = 1.0 / np.sum(fractions / K, axis=-1)
    Mu_voigt = np.sum(fractions * Mu, axis=-1)
    Mu_reuss = 1.0 / np.sum(fractions / Mu, axis=-1)

    K_vrh = 0.5 * (K_voigt + K_reuss)
    Mu_vrh = 0.5 * (Mu_voigt + Mu_reuss)
    return K_vrh, Mu_vrh


def solid_density(fractions, rho):
    '''Volumetric (mass-weighted) average mineral density (g/cc).'''
    return np.sum(np.asarray(fractions) * np.asarray(rho), axis=-1)

[ ]
def solid_poisson_ratio(Ks, Mus):
    '''Poisson's ratio of the solid mineral from its bulk & shear modulus.'''
    return (3 * Ks - 2 * Mus) / (2 * (3 * Ks + Mus))


def hertz_mindlin(Ks, Mus, phi_c, C, gamma, P_MPa):
    '''Hertz-Mindlin dry-frame moduli AT the critical porosity phi_c.

    Parameters
    ----------
    Ks, Mus : solid (mineral) bulk & shear modulus (GPa), from VRH averaging.
    phi_c : critical porosity (fraction, e.g. 0.4).
    C : coordination number (average grain contacts per grain).
    gamma : contact friction/slip coefficient, 0 (frictionless) to 1 (no-slip).
    P_MPa : differential (effective) pressure (MPa).

    Returns
    -------
    K_HM, Mu_HM : Hertz-Mindlin bulk & shear modulus at phi_c (GPa).
    '''
    sigma = solid_poisson_ratio(Ks, Mus)
    P = P_MPa / 1000.0  # MPa -> GPa, so units are consistent with Ks/Mus (GPa)

    K_HM = (C**2 * (1 - phi_c)**2 * Mus**2 / (18 * np.pi**2 * (1 - sigma)**2) * P) ** (1/3)

    coeff = (2 + 3*gamma - sigma*(1 + 3*gamma)) / (5 * (2 - sigma))
    Mu_HM = coeff * (3 * C**2 * (1 - phi_c)**2 * Mus**2 / (2 * np.pi**2 * (1 - sigma)**2) * P) ** (1/3)

    return K_HM, Mu_HM


def modified_hashin_shtrikman_frame(phi, phi_c, Ks, Mus, K_HM, Mu_HM):
    '''Interpolates dry-frame moduli between the solid (phi=0) and Hertz-Mindlin
    (phi=phi_c) endpoints, via the modified lower Hashin-Shtrikman bound
    (paper Eqs. 1-3).
    '''
    phi = np.asarray(phi, dtype=float)
    ratio = np.clip(phi / phi_c, 0.0, 1.0)  # clip: undefined for phi > phi_c (unconsolidated/suspension regime)

    K_fr = (ratio / (K_HM + 4/3*Mus) + (1 - ratio) / (Ks + 4/3*Mus)) ** (-1) - 4/3*Mus

    xi = (Mus / 6.0) * (9*Ks + 8*Mus) / (Ks + 2*Mus)
    Mu_fr = (ratio / (Mu_HM + xi) + (1 - ratio) / (Mus + xi)) ** (-1) - xi

    return K_fr, Mu_fr

[ ]
K_BRINE, RHO_BRINE = 2.80, 1.03   # GPa, g/cc
K_OIL, RHO_OIL = 0.90, 0.85       # GPa, g/cc


def fluid_properties(sw, K_w=K_BRINE, rho_w=RHO_BRINE, K_hc=K_OIL, rho_hc=RHO_OIL):
    '''Effective fluid bulk modulus and density from water saturation, via Wood's equation.

    Parameters
    ----------
    sw : array_like
        Water saturation (fraction, 0-1).
    K_w, rho_w : brine bulk modulus (GPa) and density (g/cc).
    K_hc, rho_hc : hydrocarbon bulk modulus (GPa) and density (g/cc).

    Returns
    -------
    K_fluid, rho_fluid : effective mixed-fluid bulk modulus (GPa) and density (g/cc).
    '''
    sw = np.asarray(sw, dtype=float)
    K_fluid = 1.0 / (sw / K_w + (1 - sw) / K_hc)
    rho_fluid = sw * rho_w + (1 - sw) * rho_hc
    return K_fluid, rho_fluid

[ ]
def gassmann_saturation(K_fr, Mu_fr, K_mineral, K_fluid, phi):
    '''Gassmann fluid substitution: dry-frame moduli -> fluid-saturated moduli.

    Parameters
    ----------
    K_fr, Mu_fr : dry-frame bulk & shear modulus (GPa), from Section 3.
    K_mineral : solid mineral bulk modulus (GPa), from Section 2 (VRH).
    K_fluid : effective fluid bulk modulus (GPa), from Section 4 (Wood's equation).
    phi : porosity (fraction).

    Returns
    -------
    K_sat, Mu_sat : saturated bulk & shear modulus (GPa). Mu_sat == Mu_fr by
    Gassmann's shear-modulus-unaffected-by-fluid assumption.
    '''
    phi = np.asarray(phi, dtype=float)
    K_sat = K_fr + (1 - K_fr / K_mineral)**2 / (
        phi / K_fluid + (1 - phi) / K_mineral - K_fr / K_mineral**2
    )
    Mu_sat = Mu_fr
    return K_sat, Mu_sat


def bulk_density(phi, rho_mineral, rho_fluid):
    '''Saturated bulk density (g/cc): volumetric mix of solid and fluid.'''
    return (1 - phi) * rho_mineral + phi * rho_fluid


def elastic_velocities(K_sat, Mu_sat, rho_bulk):
    '''Vp, Vs (km/s) from saturated moduli (GPa) and bulk density (g/cc).'''
    Vp = np.sqrt((K_sat + 4/3 * Mu_sat) / rho_bulk)
    Vs = np.sqrt(Mu_sat / rho_bulk)
    return Vp, Vs

[ ]
#Hertz-Mindlin 
LITHOLOGY_PARAMS = {
    #              phi_c    C      gamma
    "shale":     dict(phi_c=0.33, C=2.0, gamma=0.56),
    "sand":      dict(phi_c=0.42, C=6.7, gamma=0.01),
    "carbonate": dict(phi_c=0.40, C=6.7, gamma=0.64),
}
LITHOLOGY_ORDER = ["sand", "shale", "carbonate"]  # matches (VSAND, VSH, VCARB) convention


def effective_hm_parameters(vsand, vsh, vcarb):
    '''Volume-fraction-weighted average of phi_c, C, gamma across the three lithology
    end-members (Table 1). Explicit modeling choice -- see Section 6 discussion.
    '''
    fracs = {"sand": vsand, "shale": vsh, "carbonate": vcarb}
    phi_c = sum(fracs[l] * LITHOLOGY_PARAMS[l]["phi_c"] for l in LITHOLOGY_ORDER)
    C = sum(fracs[l] * LITHOLOGY_PARAMS[l]["C"] for l in LITHOLOGY_ORDER)
    gamma = sum(fracs[l] * LITHOLOGY_PARAMS[l]["gamma"] for l in LITHOLOGY_ORDER)
    return phi_c, C, gamma


def rock_physics_forward_model(vsh, vsand, vcarb, phi, sw, pd_mpa):
    '''Full theoretical rock-physics forward model (paper Figure 1), from scratch.

    Parameters
    ----------
    vsh, vsand, vcarb : lithology volume fractions (should sum to 1).
    phi : porosity (fraction).
    sw : water saturation (fraction).
    pd_mpa : differential (effective) pressure (MPa).

    Returns
    -------
    dict with keys: Vp, Vs, rho, Ks, mus, K_fr, Mu_fr, K_sat, Mu_sat,
    K_fluid, rho_fluid, phi_c, C, gamma  (all intermediate results, for inspection/plotting).
    '''
    vsh = np.asarray(vsh, dtype=float)
    vsand = np.asarray(vsand, dtype=float)
    vcarb = np.asarray(vcarb, dtype=float)
    phi = np.asarray(phi, dtype=float)
    sw = np.asarray(sw, dtype=float)
    pd_mpa = np.asarray(pd_mpa, dtype=float)

    fractions = np.stack([vsand, vsh, vcarb], axis=-1)
    Ks, Mus = vrh_average(fractions, K_MIN, MU_MIN)
    rho_s = solid_density(fractions, RHO_MIN)

    phi_c, C, gamma = effective_hm_parameters(vsand, vsh, vcarb)
    K_HM, Mu_HM = hertz_mindlin(Ks, Mus, phi_c, C, gamma, pd_mpa)
    K_fr, Mu_fr = modified_hashin_shtrikman_frame(phi, phi_c, Ks, Mus, K_HM, Mu_HM)

    K_fluid, rho_fluid = fluid_properties(sw)

    K_sat, Mu_sat = gassmann_saturation(K_fr, Mu_fr, Ks, K_fluid, phi)
    rho_bulk = bulk_density(phi, rho_s, rho_fluid)
    Vp, Vs = elastic_velocities(K_sat, Mu_sat, rho_bulk)

    return dict(
        Vp=Vp, Vs=Vs, rho=rho_bulk,
        Ks=Ks, mus=Mus, K_fr=K_fr, Mu_fr=Mu_fr,
        K_sat=K_sat, Mu_sat=Mu_sat,
        K_fluid=K_fluid, rho_fluid=rho_fluid,
        phi_c=phi_c, C=C, gamma=gamma,
    )

[ ]
def _smooth_random_walk(n, step_std, lo, hi, seed, start=None):
    '''A bounded random walk, used to create geologically-plausible smooth log curves.'''
    rng = np.random.default_rng(seed)
    if start is None:
        start = rng.uniform(lo, hi)
    walk = np.empty(n)
    walk[0] = np.clip(start, lo, hi)
    for i in range(1, n):
        walk[i] = np.clip(walk[i-1] + rng.normal(0, step_std), lo, hi)
    return walk


def generate_synthetic_well(well_name, n_samples, depth_start, depth_step, litho_character, seed):
    '''Generates one synthetic well: smooth VSH/VSAND/VCARB, POR, SW, PD curves, passed
    through the from-scratch rock_physics_forward_model to get exact Vp, Vs, RHO.

    Parameters
    ----------
    well_name : str, label only.
    n_samples : number of depth samples.
    depth_start : starting depth (m).
    depth_step : sample spacing (m).
    litho_character : one of "shale-dominated", "sand-dominated", "mixed", "carbonate-rich".
    seed : RNG seed (reproducibility).

    Returns
    -------
    dict of 1-D arrays: DEPTH, VSH, VSAND, VCARB, POR, SW, PD, VP, VS, RHO
    '''
    rng = np.random.default_rng(seed)
    depth = depth_start + np.arange(n_samples) * depth_step

    centers = {
        "shale-dominated":  dict(vsh=0.65, vsand=0.30, vcarb=0.05),
        "sand-dominated":   dict(vsh=0.25, vsand=0.65, vcarb=0.10),
        "mixed":            dict(vsh=0.40, vsand=0.40, vcarb=0.20),
        "carbonate-rich":   dict(vsh=0.15, vsand=0.25, vcarb=0.60),
    }[litho_character]

    vsh_raw = _smooth_random_walk(n_samples, 0.03, 0.0, 1.0, seed + 1, start=centers["vsh"])
    vsand_raw = _smooth_random_walk(n_samples, 0.03, 0.0, 1.0, seed + 2, start=centers["vsand"])
    vcarb_raw = _smooth_random_walk(n_samples, 0.02, 0.0, 1.0, seed + 3, start=centers["vcarb"])
    total = vsh_raw + vsand_raw + vcarb_raw
    vsh, vsand, vcarb = vsh_raw/total, vsand_raw/total, vcarb_raw/total

    # porosity
    phi_c_eff, _, _ = effective_hm_parameters(vsand.mean(), vsh.mean(), vcarb.mean())
    phi_start = rng.uniform(0.18, 0.30)
    phi_trend = np.linspace(phi_start, phi_start * rng.uniform(0.45, 0.7), n_samples)
    phi_noise = _smooth_random_walk(n_samples, 0.01, -0.05, 0.05, seed + 4, start=0.0)
    phi = np.clip(phi_trend + phi_noise, 0.02, 0.95 * phi_c_eff)

    # water saturation
    sw_background = rng.uniform(0.75, 1.0)
    sw = np.full(n_samples, sw_background)
    if rng.random() < 0.7:  # 70% chance of a hydrocarbon zone
        zone_len = rng.integers(n_samples // 6, n_samples // 2)
        zone_start = rng.integers(0, max(1, n_samples - zone_len))
        sw_pay = rng.uniform(0.15, 0.45)
        sw[zone_start:zone_start + zone_len] = sw_pay
    sw = np.clip(sw + rng.normal(0, 0.02, n_samples), 0.05, 1.0)

    # differential pressure
    pd_start = rng.uniform(8, 15)
    pd_end = pd_start + rng.uniform(15, 30)
    pd = np.linspace(pd_start, pd_end, n_samples) + rng.normal(0, 0.8, n_samples)
    pd = np.clip(pd, 2.0, None)

    out = rock_physics_forward_model(vsh, vsand, vcarb, phi, sw, pd)

    return dict(
        name=well_name, DEPTH=depth, VSH=vsh, VSAND=vsand, VCARB=vcarb,
        POR=phi, SW=sw, PD=pd, VP=out["Vp"], VS=out["Vs"], RHO=out["rho"],
    )





#software enabling user to input data and view results
