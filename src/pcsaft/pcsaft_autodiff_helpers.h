#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/utils/gradient.hpp>

#include "pcsaft_electrolyte.h"

namespace pcsaft_autodiff {

using std::vector;
using autodiff::dual;
using autodiff::derivative;
using autodiff::dual2nd;
using autodiff::hessian;
using autodiff::wrt;
using autodiff::at;

inline double scalar_value(double x)
{
    return x;
}

template<typename T>
inline auto scalar_value(const T& x) -> decltype(autodiff::val(x))
{
    return autodiff::val(x);
}

template<typename T>
inline bool scalar_isfinite(const T& x)
{
    return std::isfinite(static_cast<double>(scalar_value(x)));
}

template<typename T>
inline auto ad_abs(const T& x)
{
    using std::abs;
    return abs(x);
}

template<typename T>
inline auto ad_exp(const T& x)
{
    using std::exp;
    return exp(x);
}

template<typename T>
inline auto ad_log(const T& x)
{
    using std::log;
    return log(x);
}

template<typename T>
inline auto ad_sqrt(const T& x)
{
    using std::sqrt;
    return sqrt(x);
}

template<typename T, typename U>
inline auto ad_pow(const T& x, const U& y)
{
    using std::pow;
    return pow(x, y);
}

inline bool ad_is_ion_species(const add_args& cppargs, int i)
{
    return std::abs(cppargs.z[i]) > 1e-12;
}

template<typename Scalar>
inline Scalar ad_compute_ion_diameter(int i, const Scalar& t, const add_args& cppargs)
{
    if (!ad_is_ion_species(cppargs, i)) {
        return Scalar(cppargs.s[i]);
    }
    int mode = cppargs.d_ion_mode;
    double sigma_i = cppargs.s[i];
    if (sigma_i <= 0.0) {
        throw ValueError("DH/ion diameter requires positive ionic sigma_i.");
    }
    if (mode == 0) {
        return Scalar(sigma_i);
    }
    if (mode == 1) {
        return Scalar(sigma_i * (1.0 - 0.12));
    }
    if (mode == 2) {
        return Scalar(sigma_i) * (1.0 - 0.12 * ad_exp(-3.0 * cppargs.e[i] / t));
    }
    throw ValueError("Unknown d_ion_mode. Supported values are 0, 1, 2.");
}

template<typename Scalar>
inline Scalar ad_compute_ion_born_radius(int i, const Scalar& t, const add_args& cppargs)
{
    if (!ad_is_ion_species(cppargs, i)) {
        return Scalar(cppargs.s[i]);
    }
    int mode = cppargs.d_born_mode;
    double sigma_i = cppargs.s[i];
    if (sigma_i <= 0.0) {
        throw ValueError("Born term requires positive ionic sigma_i.");
    }
    if (mode == 0) {
        return Scalar(sigma_i);
    }
    if (mode == 1) {
        return Scalar(sigma_i * (1.0 - 0.12));
    }
    if (mode == 2) {
        return Scalar(sigma_i) * (1.0 - 0.12 * ad_exp(-3.0 * cppargs.e[i] / t));
    }
    if (mode == 3) {
        if (cppargs.d_born.size() <= static_cast<size_t>(i) || cppargs.d_born[i] <= 0.0) {
            throw ValueError("d_Born_mode=fitted_param requires positive ionic params['d_born'] values.");
        }
        return Scalar(cppargs.d_born[i]);
    }
    throw ValueError("Unknown d_Born_mode. Supported values are 0, 1, 2, 3.");
}

template<typename Scalar>
struct BornSSMDSData {
    vector<Scalar> d_born;
    vector<Scalar> D;
    vector<Scalar> ddelta_prefac;
    vector<double> f_k;
    vector<Scalar> bracket;
    Scalar sum_bracket = 0.0;
    Scalar sum_invD = 0.0;
    Scalar sum_gap = 0.0;
    Scalar sum_dpref_over_D2 = 0.0;
};

template<typename Scalar>
struct ContributionTerms {
    Scalar hc = 0.0;
    Scalar disp = 0.0;
    Scalar polar = 0.0;
    Scalar assoc = 0.0;
};

struct ResidualDerivatives {
    double ares = 0.0;
    double dadt = 0.0;
    double d2adt2 = 0.0;
    vector<double> dadx;
    vector<double> d2adtdx;
    vector<double> hessian_x;
};

template<typename Scalar>
Scalar compute_eps_aqueous_organic_mixed(const vector<Scalar>& x, const add_args& cppargs)
{
    int ncomp = static_cast<int>(x.size());
    if (cppargs.z.size() != static_cast<size_t>(ncomp)) {
        throw ValueError("dielc_rule=8 requires params['z'] as an array with length equal to ncomp.");
    }
    if (cppargs.mixed_rel_perm_a.size() != static_cast<size_t>(ncomp) ||
        cppargs.mixed_rel_perm_b.size() != static_cast<size_t>(ncomp) ||
        cppargs.mixed_rel_perm_c.size() != static_cast<size_t>(ncomp) ||
        cppargs.mixed_rel_perm_mask.size() != static_cast<size_t>(ncomp)) {
        throw ValueError("dielc_rule=8 requires mixed relative-permittivity arrays with length equal to ncomp.");
    }

    Scalar x_sol = 0.0;
    Scalar x_water = 0.0;
    Scalar x_org = 0.0;
    Scalar eps_org_num = 0.0;
    Scalar a_num = 0.0;
    Scalar b_num = 0.0;
    Scalar c_num = 0.0;
    bool needs_coeffs = false;

    int water_idx = cppargs.mixed_rel_perm_water_index;
    bool has_water_component = (water_idx >= 0 && water_idx < ncomp && std::abs(cppargs.z[water_idx]) <= 1e-12);

    for (int i = 0; i < ncomp; i++) {
        bool is_solvent = std::abs(cppargs.z[i]) <= 1e-12;
        if (!is_solvent) {
            continue;
        }
        x_sol += x[i];
        if (i == water_idx) {
            x_water += x[i];
        }
        else {
            x_org += x[i];
            eps_org_num += x[i] * cppargs.dielc[i];
            a_num += x[i] * cppargs.mixed_rel_perm_a[i];
            b_num += x[i] * cppargs.mixed_rel_perm_b[i];
            c_num += x[i] * cppargs.mixed_rel_perm_c[i];
            if (cppargs.mixed_rel_perm_mask[i] != 0) {
                needs_coeffs = true;
            }
        }
    }

    if (!has_water_component) {
        throw ValueError("dielc_rule=8 requires a designated water component in the solvent set.");
    }
    if (x_sol <= 0.0) {
        throw ValueError("dielc_rule=8 requires positive total solvent fraction.");
    }
    if (x_water <= 0.0) {
        throw ValueError("dielc_rule=8 requires positive water solvent fraction.");
    }

    Scalar eps_org = 0.0;
    if (x_org > 0.0) {
        eps_org = eps_org_num / x_org;
    }

    Scalar eps_sf = cppargs.dielc[water_idx];
    if (x_org > 0.0) {
        eps_sf = (x_water * cppargs.dielc[water_idx] + x_org * eps_org) / x_sol;
    }

    if (!needs_coeffs || x_org <= 0.0) {
        return eps_sf;
    }

    Scalar a = a_num / x_org;
    Scalar b = b_num / x_org;
    Scalar c = c_num / x_org;
    Scalar x_org_sol = x_org / x_sol;
    return eps_sf / (1.0 + a * x_org_sol + b * ad_pow(x_org_sol, 2) + c * ad_pow(x_org_sol, 3));
}

template<typename Scalar>
Scalar compute_eps_rule(int rule, const vector<Scalar>& x, const add_args& cppargs)
{
    const double alpha = 7.01;
    int ncomp = static_cast<int>(x.size());
    if (rule == 0) {
        if (cppargs.dielc.empty()) {
            throw ValueError("dielc_rule=0 requires params['dielc'].");
        }
        return cppargs.dielc[0];
    }
    if (rule == 1) {
        Scalar eps = 0.0;
        for (int i = 0; i < ncomp; i++) {
            eps += x[i] * cppargs.dielc[i];
        }
        return eps;
    }
    if (rule == 7) {
        vector<int> idx_sol;
        vector<int> idx_ion;
        for (int i = 0; i < ncomp; i++) {
            if (std::abs(cppargs.z[i]) <= 1e-12) idx_sol.push_back(i);
            else idx_ion.push_back(i);
        }
        if (idx_sol.empty() || idx_ion.empty()) {
            throw ValueError("dielc_rule=7 requires both solvent and ionic species.");
        }
        Scalar x_sol = 0.0;
        Scalar eps_sol_num = 0.0;
        for (int idx : idx_sol) {
            x_sol += x[idx];
            eps_sol_num += x[idx] * cppargs.dielc[idx];
        }
        Scalar eps_sol = 0.0;
        if (x_sol > 1.0e-16) {
            eps_sol = eps_sol_num / x_sol;
        }
        else {
            for (int idx : idx_sol) eps_sol += cppargs.dielc[idx];
            eps_sol /= static_cast<double>(idx_sol.size());
        }
        double eps_salt = 0.0;
        for (int idx : idx_ion) eps_salt += cppargs.dielc[idx];
        eps_salt /= static_cast<double>(idx_ion.size());
        return eps_sol * x_sol + eps_salt * (1.0 - x_sol);
    }
    if (rule == 8) {
        return compute_eps_aqueous_organic_mixed(x, cppargs);
    }
    if (rule == 2) {
        Scalar mw_bar = 0.0;
        Scalar num = 0.0;
        for (int i = 0; i < ncomp; i++) {
            mw_bar += x[i] * cppargs.mw[i];
            num += x[i] * cppargs.mw[i] * cppargs.dielc[i];
        }
        if (mw_bar <= 0.0) {
            throw ValueError("Average molecular weight must be positive for dielc_rule=2.");
        }
        return num / mw_bar;
    }
    if (rule == 3) {
        vector<int> idx_sol;
        vector<int> idx_ion;
        for (int i = 0; i < ncomp; i++) {
            if (std::abs(cppargs.z[i]) <= 1e-12) idx_sol.push_back(i);
            else idx_ion.push_back(i);
        }
        if (idx_sol.empty()) {
            throw ValueError("dielc_rule=3 requires at least one solvent species (z=0).");
        }
        Scalar mw_sol = 0.0;
        Scalar eps_sol_num = 0.0;
        for (int idx : idx_sol) {
            mw_sol += x[idx] * cppargs.mw[idx];
            eps_sol_num += x[idx] * cppargs.mw[idx] * cppargs.dielc[idx];
        }
        if (mw_sol <= 0.0) {
            throw ValueError("Solvent molecular-weight denominator must be positive for dielc_rule=3.");
        }
        Scalar eps_sol_w = eps_sol_num / mw_sol;
        Scalar x_sol = 0.0;
        Scalar eps_ion = 0.0;
        for (int idx : idx_sol) x_sol += x[idx];
        for (int idx : idx_ion) eps_ion += x[idx] * cppargs.dielc[idx];
        return x_sol * eps_sol_w + eps_ion;
    }
    if (rule == 4 || rule == 5) {
        vector<int> idx_sol;
        vector<int> idx_ion;
        for (int i = 0; i < ncomp; i++) {
            if (std::abs(cppargs.z[i]) <= 1e-12) idx_sol.push_back(i);
            else idx_ion.push_back(i);
        }
        if (idx_sol.empty()) {
            throw ValueError("dielc_rule requires at least one solvent species (z=0).");
        }
        Scalar mw_sol = 0.0;
        Scalar eps_sf_num = 0.0;
        for (int idx : idx_sol) {
            mw_sol += x[idx] * cppargs.mw[idx];
            eps_sf_num += x[idx] * cppargs.mw[idx] * cppargs.dielc[idx];
        }
        if (mw_sol <= 0.0) {
            throw ValueError("Solvent molecular-weight denominator must be positive for dielc_rule.");
        }
        Scalar eps_sf = eps_sf_num / mw_sol;
        Scalar x_ion = 0.0;
        for (int idx : idx_ion) x_ion += x[idx];
        return eps_sf / (1.0 + alpha * x_ion);
    }
    if (rule == 6) {
        vector<int> idx_sol;
        vector<int> idx_ion;
        for (int i = 0; i < ncomp; i++) {
            if (std::abs(cppargs.z[i]) <= 1e-12) idx_sol.push_back(i);
            else idx_ion.push_back(i);
        }
        if (idx_sol.empty()) {
            throw ValueError("dielc_rule=6 requires at least one solvent species (z=0).");
        }
        double eps_sf_const = 0.0;
        for (int idx : idx_sol) eps_sf_const += cppargs.dielc[idx];
        eps_sf_const /= static_cast<double>(idx_sol.size());
        Scalar x_ion = 0.0;
        for (int idx : idx_ion) x_ion += x[idx];
        return eps_sf_const / (1.0 + alpha * x_ion);
    }
    throw ValueError("Unknown dielc_rule. Supported rules are 0, 1, 2, 3, 4, 5, 6, 7, 8.");
}

template<typename Scalar>
Scalar compute_eps_solvent_reference(const vector<Scalar>& x, const add_args& cppargs)
{
    int ncomp = static_cast<int>(x.size());
    Scalar x_sol = 0.0;
    Scalar eps_sol_num = 0.0;
    for (int i = 0; i < ncomp; i++) {
        if (std::abs(cppargs.z[i]) <= 1e-12) {
            x_sol += x[i];
            eps_sol_num += x[i] * cppargs.dielc[i];
        }
    }
    if (x_sol <= 0.0) {
        return compute_eps_rule(cppargs.dielc_rule, x, cppargs);
    }
    return eps_sol_num / x_sol;
}

template<typename Scalar>
BornSSMDSData<Scalar> build_born_ssmds_data(const vector<Scalar>& x, const add_args& cppargs, const Scalar& t, const Scalar& eps_r, double eps_r_ion)
{
    int ncomp = static_cast<int>(x.size());
    const bool use_ssm = (cppargs.born_solvation_shell_model != 0);
    const bool use_ds = (cppargs.born_dielectric_saturation != 0);

    BornSSMDSData<Scalar> data;
    data.d_born.assign(ncomp, 1.0);
    data.D.assign(ncomp, 1.0);
    data.ddelta_prefac.assign(ncomp, 0.0);
    data.f_k.assign(ncomp, 1.0);
    data.bracket.assign(ncomp, 0.0);

    Scalar f_mix = 0.0;
    for (int i = 0; i < ncomp; i++) {
        bool is_ion = ad_is_ion_species(cppargs, i);
        double fi = 1.0;
        if (!is_ion && cppargs.f_solv.size() > static_cast<size_t>(i)) {
            fi = cppargs.f_solv[i];
        }
        data.f_k[i] = fi;
        f_mix += x[i] * fi;

        if (is_ion) {
            data.d_born[i] = ad_compute_ion_born_radius(i, t, cppargs);
        }
        else if (cppargs.d_born.size() > static_cast<size_t>(i) && cppargs.d_born[i] > 0.0) {
            data.d_born[i] = cppargs.d_born[i];
        }
        else if (cppargs.s[i] > 0.0) {
            data.d_born[i] = cppargs.s[i];
        }
        else {
            throw ValueError("Born model requires positive solvent diameter.");
        }

        if (is_ion) {
            data.ddelta_prefac[i] = data.d_born[i] / std::abs(cppargs.z[i]);
        }
    }

    for (int i = 0; i < ncomp; i++) {
        bool is_ion = std::abs(cppargs.z[i]) > 1e-12;
        if (!is_ion) {
            data.D[i] = data.d_born[i];
            continue;
        }

        Scalar delta_di = use_ssm ? ((f_mix - 1.0) * data.ddelta_prefac[i]) : Scalar(0.0);
        data.D[i] = data.d_born[i] + delta_di;
        if (data.D[i] <= 0.0) {
            throw ValueError("Born model generated a non-positive d_born + Delta d.");
        }

        double z2 = cppargs.z[i] * cppargs.z[i];
        Scalar invD = 1.0 / data.D[i];
        Scalar gap = (1.0 / data.d_born[i] - invD);
        Scalar base_term = (1.0 - 1.0 / eps_r) * invD;
        Scalar ds_term = use_ds ? ((1.0 - 1.0 / eps_r_ion) * gap) : Scalar(0.0);

        data.bracket[i] = base_term + ds_term;
        data.sum_bracket += x[i] * z2 * data.bracket[i];
        data.sum_invD += x[i] * z2 * invD;
        data.sum_gap += x[i] * z2 * gap;
        if (use_ssm) {
            data.sum_dpref_over_D2 += x[i] * z2 * data.ddelta_prefac[i] * invD * invD;
        }
    }
    return data;
}

template<typename Scalar>
Scalar compute_born_ares_only(const Scalar& t, const vector<Scalar>& x, const add_args& cppargs)
{
    if (cppargs.born_model == 0) {
        return 0.0;
    }
    Scalar eps_mix = compute_eps_rule(cppargs.dielc_rule, x, cppargs);
    Scalar eps_born = (cppargs.born_eps_mode == 1) ? compute_eps_solvent_reference(x, cppargs) : eps_mix;
    if (cppargs.born_model == 1) {
        Scalar born_sum = 0.0;
        for (int i = 0; i < static_cast<int>(x.size()); i++) {
            if (ad_is_ion_species(cppargs, i)) {
                Scalar d_born_i = ad_compute_ion_born_radius(i, t, cppargs);
                born_sum += x[i] * cppargs.z[i] * cppargs.z[i] / d_born_i;
            }
        }
        return -E_CHRG * E_CHRG / (4.0 * PI * kb * t * perm_vac) * (1.0 - 1.0 / eps_born) * born_sum;
    }
    if (cppargs.born_model == 2) {
        const double eps_r_ion = 8.0;
        const Scalar Kborn = E_CHRG * E_CHRG / (4.0 * PI * kb * t * perm_vac);
        auto born = build_born_ssmds_data(x, cppargs, t, eps_born, eps_r_ion);
        return -Kborn * born.sum_bracket;
    }
    throw ValueError("Unknown born_model. Supported values are 0, 1, 2.");
}

template<typename Scalar>
Scalar compute_dh_ares_only(const Scalar& t, double rho, const vector<Scalar>& x, const add_args& cppargs)
{
    if (cppargs.z.empty()) {
        return 0.0;
    }
    int ncomp = static_cast<int>(x.size());
    vector<Scalar> d(ncomp, Scalar(0.0));
    for (int i = 0; i < ncomp; i++) {
        d[i] = Scalar(cppargs.s[i]) * (1.0 - 0.12 * ad_exp(-3.0 * cppargs.e[i] / t));
        if (ad_is_ion_species(cppargs, i)) {
            d[i] = ad_compute_ion_diameter(i, t, cppargs);
        }
    }

    double den = rho * N_AV / 1.0e30;
    Scalar Qsum = 0.0;
    for (int i = 0; i < ncomp; i++) {
        Qsum += cppargs.z[i] * cppargs.z[i] * x[i];
    }
    if (Qsum == 0.0) {
        return 0.0;
    }

    Scalar eps = compute_eps_rule(cppargs.dielc_rule, x, cppargs);
    Scalar kappa = ad_sqrt(den * E_CHRG * E_CHRG / kb / t / (eps * perm_vac) * Qsum);
    if (kappa == 0.0) {
        return 0.0;
    }

    Scalar S = 0.0;
    for (int i = 0; i < ncomp; i++) {
        Scalar ka = kappa * d[i];
        Scalar chi = 3 / ad_pow(ka, 3) * (1.5 + ad_log(1 + ka) - 2 * (1 + ka) + 0.5 * ad_pow(1 + ka, 2));
        S += x[i] * cppargs.z[i] * cppargs.z[i] * chi;
    }

    Scalar K0 = E_CHRG * E_CHRG / (12.0 * PI * kb * t * perm_vac);
    return -K0 * kappa / eps * S;
}

template<typename Scalar>
ContributionTerms<Scalar> compute_contribution_terms(const Scalar& t, double rho, const vector<Scalar>& x, const add_args& cppargs)
{
    ContributionTerms<Scalar> out;
    int ncomp = static_cast<int>(x.size());
    vector<Scalar> d(ncomp, Scalar(0.0));
    for (int i = 0; i < ncomp; i++) {
        d[i] = Scalar(cppargs.s[i]) * (1 - 0.12 * ad_exp(-3 * cppargs.e[i] / t));
        if (!cppargs.z.empty() && ad_is_ion_species(cppargs, i)) {
            d[i] = ad_compute_ion_diameter(i, t, cppargs);
        }
    }

    double den = rho * N_AV / 1.0e30;

    vector<Scalar> zeta(4, 0.0);
    Scalar summ = 0.0;
    for (int i = 0; i < 4; i++) {
        summ = 0.0;
        for (int j = 0; j < ncomp; j++) {
            summ += x[j] * cppargs.m[j] * ad_pow(d[j], i);
        }
        zeta[i] = PI / 6 * den * summ;
    }

    Scalar eta = zeta[3];
    Scalar m_avg = 0.0;
    for (int i = 0; i < ncomp; i++) {
        m_avg += x[i] * cppargs.m[i];
    }

    vector<Scalar> ghs(ncomp * ncomp, 0.0);
    vector<double> e_ij(ncomp * ncomp, 0.0);
    vector<double> s_ij(ncomp * ncomp, 0.0);
    Scalar m2es3 = 0.0;
    Scalar m2e2s3 = 0.0;
    int idx = -1;
    for (int i = 0; i < ncomp; i++) {
        for (int j = 0; j < ncomp; j++) {
            idx += 1;
            if (cppargs.l_ij.empty()) {
                s_ij[idx] = (cppargs.s[i] + cppargs.s[j]) / 2.0;
            }
            else {
                s_ij[idx] = (cppargs.s[i] + cppargs.s[j]) / 2.0 * (1 - cppargs.l_ij[idx]);
            }
            if (!cppargs.z.empty()) {
                if (cppargs.z[i] * cppargs.z[j] <= 0) {
                    if (cppargs.k_ij.empty()) {
                        e_ij[idx] = std::sqrt(cppargs.e[i] * cppargs.e[j]);
                    }
                    else {
                        e_ij[idx] = std::sqrt(cppargs.e[i] * cppargs.e[j]) * (1 - cppargs.k_ij[idx]);
                    }
                }
            }
            else {
                if (cppargs.k_ij.empty()) {
                    e_ij[idx] = std::sqrt(cppargs.e[i] * cppargs.e[j]);
                }
                else {
                    e_ij[idx] = std::sqrt(cppargs.e[i] * cppargs.e[j]) * (1 - cppargs.k_ij[idx]);
                }
            }
            m2es3 += x[i] * x[j] * cppargs.m[i] * cppargs.m[j] * e_ij[idx] / t * ad_pow(s_ij[idx], 3);
            m2e2s3 += x[i] * x[j] * cppargs.m[i] * cppargs.m[j] * ad_pow(e_ij[idx] / t, 2) * ad_pow(s_ij[idx], 3);
            ghs[idx] = 1 / (1 - zeta[3]) + (d[i] * d[j] / (d[i] + d[j])) * 3 * zeta[2] / ad_pow(1 - zeta[3], 2) +
                ad_pow(d[i] * d[j] / (d[i] + d[j]), 2) * 2 * zeta[2] * zeta[2] / ad_pow(1 - zeta[3], 3);
        }
    }

    Scalar ares_hs = 1 / zeta[0] * (3 * zeta[1] * zeta[2] / (1 - zeta[3]) + ad_pow(zeta[2], 3.) / (zeta[3] * ad_pow(1 - zeta[3], 2))
        + (ad_pow(zeta[2], 3.) / ad_pow(zeta[3], 2.) - zeta[0]) * ad_log(1 - zeta[3]));

    static double a0[7] = { 0.9105631445, 0.6361281449, 2.6861347891, -26.547362491, 97.759208784, -159.59154087, 91.297774084 };
    static double a1[7] = { -0.3084016918, 0.1860531159, -2.5030047259, 21.419793629, -65.255885330, 83.318680481, -33.746922930 };
    static double a2[7] = { -0.0906148351, 0.4527842806, 0.5962700728, -1.7241829131, -4.1302112531, 13.776631870, -8.6728470368 };
    static double b0[7] = { 0.7240946941, 2.2382791861, -4.0025849485, -21.003576815, 26.855641363, 206.55133841, -355.60235612 };
    static double b1[7] = { -0.5755498075, 0.6995095521, 3.8925673390, -17.215471648, 192.67226447, -161.82646165, -165.20769346 };
    static double b2[7] = { 0.0976883116, -0.2557574982, -9.1558561530, 20.642075974, -38.804430052, 93.626774077, -29.666905585 };

    vector<Scalar> a(7, 0.0);
    vector<Scalar> b(7, 0.0);
    for (int i = 0; i < 7; i++) {
        a[i] = a0[i] + (m_avg - 1.) / m_avg * a1[i] + (m_avg - 1.) / m_avg * (m_avg - 2.) / m_avg * a2[i];
        b[i] = b0[i] + (m_avg - 1.) / m_avg * b1[i] + (m_avg - 1.) / m_avg * (m_avg - 2.) / m_avg * b2[i];
    }

    Scalar I1 = 0.0;
    Scalar I2 = 0.0;
    for (int i = 0; i < 7; i++) {
        I1 += a[i] * ad_pow(eta, i);
        I2 += b[i] * ad_pow(eta, i);
    }
    Scalar C1 = 1. / (1. + m_avg * (8 * eta - 2 * eta * eta) / ad_pow(1 - eta, 4) + (1 - m_avg) * (20 * eta - 27 * eta * eta + 12 * ad_pow(eta, 3) - 2 * ad_pow(eta, 4)) / ad_pow((1 - eta) * (2 - eta), 2.0));

    summ = 0.0;
    for (int i = 0; i < ncomp; i++) {
        summ += x[i] * (cppargs.m[i] - 1) * ad_log(ghs[i * ncomp + i]);
    }
    out.hc = m_avg * ares_hs - summ;
    out.disp = -2 * PI * den * I1 * m2es3 - PI * den * m_avg * C1 * I2 * m2e2s3;

    if (!cppargs.dipm.empty()) {
        Scalar A2 = 0.0;
        Scalar A3 = 0.0;
        vector<double> dipmSQ(ncomp, 0.0);

        static double a0dip[5] = { 0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308 };
        static double a1dip[5] = { 0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135 };
        static double a2dip[5] = { -1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575 };
        static double b0dip[5] = { 0.2187939, -1.1896431, 1.1626889, 0, 0 };
        static double b1dip[5] = { -0.5873164, 1.2489132, -0.5085280, 0, 0 };
        static double b2dip[5] = { 3.4869576, -14.915974, 15.372022, 0, 0 };
        static double c0dip[5] = { -0.0646774, 0.1975882, -0.8087562, 0.6902849, 0 };
        static double c1dip[5] = { -0.9520876, 2.9924258, -2.3802636, -0.2701261, 0 };
        static double c2dip[5] = { -0.6260979, 1.2924686, 1.6542783, -3.4396744, 0 };

        const static double conv = 7242.702976750923;
        for (int i = 0; i < ncomp; i++) {
            dipmSQ[i] = ad_pow(cppargs.dipm[i], 2.) / (cppargs.m[i] * cppargs.e[i] * ad_pow(cppargs.s[i], 3.)) * conv;
        }

        vector<Scalar> adip(5, 0.0);
        vector<Scalar> bdip(5, 0.0);
        vector<Scalar> cdip(5, 0.0);
        Scalar J2, J3;
        double m_ij, m_ijk;
        for (int i = 0; i < ncomp; i++) {
            for (int j = 0; j < ncomp; j++) {
                m_ij = std::sqrt(cppargs.m[i] * cppargs.m[j]);
                if (m_ij > 2) {
                    m_ij = 2;
                }
                J2 = 0.0;
                for (int l = 0; l < 5; l++) {
                    adip[l] = a0dip[l] + (m_ij - 1) / m_ij * a1dip[l] + (m_ij - 1) / m_ij * (m_ij - 2) / m_ij * a2dip[l];
                    bdip[l] = b0dip[l] + (m_ij - 1) / m_ij * b1dip[l] + (m_ij - 1) / m_ij * (m_ij - 2) / m_ij * b2dip[l];
                    J2 += (adip[l] + bdip[l] * e_ij[j * ncomp + j] / t) * ad_pow(eta, l);
                }
                A2 += x[i] * x[j] * e_ij[i * ncomp + i] / t * e_ij[j * ncomp + j] / t * ad_pow(s_ij[i * ncomp + i], 3) * ad_pow(s_ij[j * ncomp + j], 3) /
                    ad_pow(s_ij[i * ncomp + j], 3) * cppargs.dip_num[i] * cppargs.dip_num[j] * dipmSQ[i] * dipmSQ[j] * J2;

                for (int k = 0; k < ncomp; k++) {
                    m_ijk = ad_pow((cppargs.m[i] * cppargs.m[j] * cppargs.m[k]), 1 / 3.);
                    if (m_ijk > 2) {
                        m_ijk = 2;
                    }
                    J3 = 0.0;
                    for (int l = 0; l < 5; l++) {
                        cdip[l] = c0dip[l] + (m_ijk - 1) / m_ijk * c1dip[l] + (m_ijk - 1) / m_ijk * (m_ijk - 2) / m_ijk * c2dip[l];
                        J3 += cdip[l] * ad_pow(eta, l);
                    }
                    A3 += x[i] * x[j] * x[k] * e_ij[i * ncomp + i] / t * e_ij[j * ncomp + j] / t * e_ij[k * ncomp + k] / t *
                        ad_pow(s_ij[i * ncomp + i], 3) * ad_pow(s_ij[j * ncomp + j], 3) * ad_pow(s_ij[k * ncomp + k], 3) / s_ij[i * ncomp + j] / s_ij[i * ncomp + k] /
                        s_ij[j * ncomp + k] * cppargs.dip_num[i] * cppargs.dip_num[j] * cppargs.dip_num[k] * dipmSQ[i] *
                        dipmSQ[j] * dipmSQ[k] * J3;
                }
            }
        }

        A2 = -PI * den * A2;
        A3 = -4 / 3. * PI * PI * den * den * A3;
        if (A2 != 0) {
            out.polar = A2 / (1 - A3 / A2);
        }
    }

    if (!cppargs.e_assoc.empty()) {
        int num_sites = 0;
        vector<int> iA;
        for (vector<int>::const_iterator it = cppargs.assoc_num.begin(); it != cppargs.assoc_num.end(); ++it) {
            num_sites += *it;
            for (int i = 0; i < *it; i++) {
                iA.push_back(static_cast<int>(it - cppargs.assoc_num.begin()));
            }
        }

        vector<Scalar> x_assoc(num_sites);
        for (int i = 0; i < num_sites; i++) {
            x_assoc[i] = x[iA[i]];
        }

        vector<Scalar> XA(num_sites, 0.0);
        vector<Scalar> delta_ij(num_sites * num_sites, 0.0);
        int idxa = 0;
        int idxi = 0;
        int idxj = 0;
        for (int i = 0; i < num_sites; i++) {
            idxi = iA[i] * ncomp + iA[i];
            for (int j = 0; j < num_sites; j++) {
                idxj = iA[j] * ncomp + iA[j];
                if (cppargs.assoc_matrix[idxa] != 0) {
                    double eABij = (cppargs.e_assoc[iA[i]] + cppargs.e_assoc[iA[j]]) / 2.;
                    double volABij = HUGE_VAL;
                    if (cppargs.k_hb.empty()) {
                        volABij = std::sqrt(cppargs.vol_a[iA[i]] * cppargs.vol_a[iA[j]]) * ad_pow(std::sqrt(s_ij[idxi] *
                            s_ij[idxj]) / (0.5 * (s_ij[idxi] + s_ij[idxj])), 3);
                    }
                    else {
                        volABij = std::sqrt(cppargs.vol_a[iA[i]] * cppargs.vol_a[iA[j]]) * ad_pow(std::sqrt(s_ij[idxi] *
                            s_ij[idxj]) / (0.5 * (s_ij[idxi] + s_ij[idxj])), 3) * (1 - cppargs.k_hb[iA[i] * ncomp + iA[j]]);
                    }
                    delta_ij[idxa] = ghs[iA[i] * ncomp + iA[j]] * (ad_exp(eABij / t) - 1) * ad_pow(s_ij[iA[i] * ncomp + iA[j]], 3) * volABij;
                }
                idxa += 1;
            }
            XA[i] = (-1 + ad_sqrt(1 + 8 * den * delta_ij[i * num_sites + i])) / (4 * den * delta_ij[i * num_sites + i]);
            if (!scalar_isfinite(XA[i])) {
                XA[i] = 0.02;
            }
        }

        int ctr = 0;
        double dif = 1000.0;
        vector<Scalar> XA_old = XA;
        while ((ctr < 100) && (dif > 1e-15)) {
            ctr += 1;
            idxa = 0;
            for (int i = 0; i < num_sites; i++) {
                Scalar assoc_sum = 0.0;
                for (int j = 0; j < num_sites; j++) {
                    assoc_sum += den * x_assoc[j] * XA_old[j] * delta_ij[idxa];
                    idxa += 1;
                }
                XA[i] = 1. / (1. + assoc_sum);
            }
            dif = 0.0;
            for (int i = 0; i < num_sites; i++) {
                dif += std::abs(static_cast<double>(scalar_value(XA[i] - XA_old[i])));
            }
            for (int i = 0; i < num_sites; i++) {
                XA_old[i] = (XA[i] + XA_old[i]) / 2.0;
            }
        }

        out.assoc = 0.0;
        for (int i = 0; i < num_sites; i++) {
            out.assoc += x[iA[i]] * (ad_log(XA[i]) - 0.5 * XA[i] + 0.5);
        }
    }

    return out;
}

template<typename Scalar>
Scalar compute_residual_ares(const Scalar& t, double rho, const vector<Scalar>& x, const add_args& cppargs)
{
    ContributionTerms<Scalar> terms = compute_contribution_terms(t, rho, x, cppargs);
    Scalar ares = terms.hc + terms.disp + terms.polar + terms.assoc;
    if (!cppargs.z.empty()) {
        ares += compute_dh_ares_only(t, rho, x, cppargs);
        ares += compute_born_ares_only(t, x, cppargs);
    }
    return ares;
}

inline ResidualDerivatives compute_residual_derivatives(double t, double rho, const vector<double>& x, const add_args& cppargs)
{
    int ncomp = static_cast<int>(x.size());
    vector<dual2nd> vars(ncomp + 1);
    vars[0] = t;
    for (int i = 0; i < ncomp; i++) {
        vars[i + 1] = x[i];
    }

    auto f = [&](const vector<dual2nd>& vars_in) -> dual2nd {
        vector<dual2nd> x_ad(ncomp);
        for (int i = 0; i < ncomp; i++) {
            x_ad[i] = vars_in[i + 1];
        }
        return compute_residual_ares(vars_in[0], rho, x_ad, cppargs);
    };

    dual2nd u;
    Eigen::VectorXd g;
    auto H = hessian(f, wrt(vars), at(vars), u, g);

    ResidualDerivatives out;
    out.ares = scalar_value(u);
    out.dadx.assign(ncomp, 0.0);
    out.d2adtdx.assign(ncomp, 0.0);
    out.hessian_x.assign(ncomp * ncomp, 0.0);
    if (g.size() != ncomp + 1 || H.rows() != ncomp + 1 || H.cols() != ncomp + 1) {
        throw ValueError("Unexpected autodiff derivative shape for residual Helmholtz derivatives.");
    }
    out.dadt = g(0);
    out.d2adt2 = H(0, 0);
    for (int i = 0; i < ncomp; i++) {
        out.dadx[i] = g(i + 1);
        out.d2adtdx[i] = H(0, i + 1);
        for (int j = 0; j < ncomp; j++) {
            out.hessian_x[i * ncomp + j] = H(i + 1, j + 1);
        }
    }
    return out;
}

template<typename Eval>
vector<double> compute_gradient(const vector<double>& x, Eval&& eval, const char* err_msg)
{
    int ncomp = static_cast<int>(x.size());
    vector<double> grad(ncomp, 0.0);
    for (int i = 0; i < ncomp; i++) {
        dual xi = x[i];
        auto fi = [&](const dual& x_i) -> dual {
            vector<dual> x_ad(ncomp);
            for (int j = 0; j < ncomp; j++) {
                x_ad[j] = x[j];
            }
            x_ad[i] = x_i;
            return eval(x_ad);
        };
        double g = derivative(fi, wrt(xi), at(xi));
        if (!std::isfinite(g)) {
            throw ValueError(err_msg);
        }
        grad[i] = g;
    }
    return grad;
}

} // namespace pcsaft_autodiff
