# -*- coding: utf-8 -*-
# setuptools: language=c++
"""
Created on Thu Jul 19 14:23:00 2018

@author: Zach Baird
"""
from libcpp.vector cimport vector

cdef extern from "pcsaft_electrolyte.cpp":
    double pcsaft_p_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    double pcsaft_Z_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_Z_contrib_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_ion_dh_debug_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_lnfug_debug_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_lnfug_terms_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_dielc_debug_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_lnfug_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_mu_res_contrib_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_fugcoef_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    double pcsaft_den_cpp(double t, double p, vector[double] x, int phase, add_args &cppargs) except +
    double pcsaft_ares_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_ares_contrib_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    double pcsaft_dadt_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    double pcsaft_hres_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    double pcsaft_sres_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    double pcsaft_gres_cpp(double t, double rho, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_lnfugcoef_inf_dil_cpp(double t, double p, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_gsolv_cpp(double t, double p, vector[double] x, add_args &cppargs) except +
    vector[double] pcsaft_gtransfer_cpp(double t, double p, vector[double] x1, add_args &cppargs1, vector[double] x2, add_args &cppargs2) except +
    vector[double] flashTQ_cpp(double t, double Q, vector[double] x, add_args &cppargs) except +
    vector[double] flashTQ_cpp(double t, double Q, vector[double] x, add_args &cppargs, double p_guess) except +
    vector[double] flashPQ_cpp(double p, double Q, vector[double] x, add_args &cppargs) except +
    vector[double] flashPQ_cpp(double p, double Q, vector[double] x, add_args &cppargs, double t_guess) except +

    ctypedef struct add_args:
        vector[double] m
        vector[double] s
        vector[double] e
        vector[double] k_ij
        vector[double] e_assoc
        vector[double] vol_a
        vector[double] dipm
        vector[double] dip_num
        vector[double] MW
        vector[double] z
        vector[double] dielc
        vector[double] d_born
        vector[double] f_solv
        int born_model
        int born_enabled
        int dielc_rule
        int bjerrum_model
        double dielc_ion
        vector[int] assoc_num
        vector[int] assoc_matrix
        vector[double] k_hb
        vector[double] l_ij
