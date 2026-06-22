# -*- coding: utf-8 -*-
# setuptools: language=c++
"""
Created on Thu Jul 19 14:23:00 2018

@author: Zach Baird
"""
from libcpp.vector cimport vector

cdef extern from "pcsaft_electrolyte.cpp":
    double pcsaft_p_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    double pcsaft_Z_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    vector[double] pcsaft_lnfug_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    vector[double] pcsaft_lnfug_terms_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    vector[double] pcsaft_fugcoef_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    double pcsaft_den_cpp(double t, double p, vector[double] x, int phase, add_args &cppargs)
    double pcsaft_ares_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    double pcsaft_dadt_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    vector[double] pcsaft_autodiff_dadx_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    double pcsaft_autodiff_d2adt2_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    vector[double] pcsaft_autodiff_d2adtdx_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    vector[double] pcsaft_autodiff_hessian_x_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    double pcsaft_hres_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    double pcsaft_sres_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    double pcsaft_gres_cpp(double t, double rho, vector[double] x, add_args &cppargs)
    vector[double] flashTQ_cpp(double t, double Q, vector[double] x, add_args &cppargs) except +
    vector[double] flashTQ_cpp(double t, double Q, vector[double] x, add_args &cppargs, double p_guess) except +
    vector[double] flashPQ_cpp(double p, double Q, vector[double] x, add_args &cppargs) except +
    vector[double] flashPQ_cpp(double p, double Q, vector[double] x, add_args &cppargs, double t_guess) except +
    double pcsaft_dielc_eps_cpp(vector[double] x, add_args &cppargs)
    vector[double] pcsaft_dielc_diff_cpp(vector[double] x, add_args &cppargs)

    ctypedef struct add_args:
        vector[double] m
        vector[double] s
        vector[double] e
        vector[double] k_ij
        vector[double] e_assoc
        vector[double] vol_a
        vector[double] dipm
        vector[double] dip_num
        vector[double] z
        vector[double] dielc
        vector[double] mw
        vector[double] mixed_rel_perm_a
        vector[double] mixed_rel_perm_b
        vector[double] mixed_rel_perm_c
        vector[int] mixed_rel_perm_mask
        int mixed_rel_perm_water_index
        int dielc_rule
        int dielc_diff_mode
        int hc_dadx_diff_mode
        int disp_dadx_diff_mode
        int assoc_dadx_diff_mode
        int polar_dadx_diff_mode
        int dadt_diff_mode
        int d_ion_mode
        int mu_DH_diff_mode
        int mu_DH_comp_dep_rel_perm
        int mu_DH_include_sum_term
        int include_born_model
        int d_born_mode
        int born_solvation_shell_model
        int born_dielectric_saturation
        int born_bulk_mode
        int mu_born_diff_mode
        int mu_born_comp_dep_rel_perm
        int mu_born_include_sum_term
        int mu_born_comp_dep_delta_d
        vector[double] d_born
        vector[double] f_solv
        int born_model
        int born_radius_model
        int born_diff_mode
        int born_eps_mode
        int DH_model
        int debug
        vector[int] assoc_num
        vector[int] assoc_matrix
        vector[double] k_hb
        vector[double] l_ij
