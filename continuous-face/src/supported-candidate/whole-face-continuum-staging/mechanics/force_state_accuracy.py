"""Explicit local force-plus-state accuracy fallback; no physical-law change."""
import numpy as np
from scipy.sparse import diags
from linear_equilibration import solve_kkt
POLICY='absolute_force_or_full_qp_force_state_accuracy_v1'
RELATIVE_TOLERANCE=1e-5
SITE_CORRECTION_TOLERANCE_MM=.0005


def accuracy_fallback(solver,q,direction,mixed_gradient,dt,generalized,epigraph,regularization):
    valid=(epigraph.get('qp_solved') is True and epigraph.get('search_only') is False
        and epigraph.get('usable_for_generalized_stationarity') is True
        and epigraph.get('energy_law_unchanged') is True
        and epigraph.get('seam_rows_are_source_support_pins') is False
        and epigraph.get('actual_comaximality_satisfied') is True
        and epigraph.get('branch_weights_nonnegative') is True
        and epigraph.get('group_weight_sum_error',float('inf'))<=1e-10
        and regularization==0.)
    direction=np.asarray(direction,float).reshape(q.shape)
    relative=generalized['relative_dynamic_balance_max']
    preliminary=valid and np.isfinite(direction).all() and np.isfinite(relative) and relative<=RELATIVE_TOLERANCE
    report=dict(policy=POLICY,route='full_qp_force_state_fallback',qualified=False,
        full_qp_actual_reaction_qualified=bool(valid),numerical_relative_diagonal_regularization=float(regularization),
        qp_solved=epigraph.get('qp_solved'),search_only=epigraph.get('search_only'),
        actual_comaximality_satisfied=epigraph.get('actual_comaximality_satisfied'),
        branch_weights_nonnegative=epigraph.get('branch_weights_nonnegative'),
        group_weight_sum_error=epigraph.get('group_weight_sum_error'),
        relative_force_balance=float(relative),relative_force_tolerance=RELATIVE_TOLERANCE,
        maximum_occupied_site_correction_tolerance_mm=SITE_CORRECTION_TOLERANCE_MM,
        scope='Local unregularized coupled correction indicator, not a rigorous total state or trajectory error bound',
        residual_projection_scope='source_origin_support_reactions_only',all_physical_contact_and_barrier_reactions_included=True)
    if not preliminary:return report
    correction=float(solver.guard.max_physical_site_difference(q,q+direction))
    C=None if solver.constraints is None else solver.constraints.C3
    source_error=0. if C is None else float(np.max(abs(C@direction.ravel()),initial=0))
    report.update(maximum_occupied_site_correction_mm=correction,
        maximum_occupied_site_velocity_correction_mm_s=correction/dt,
        maximum_carrier_correction_mm=float(np.linalg.norm(direction,axis=1).max(initial=0)),
        source_constraint_direction_error_mm=source_error)
    qualified=bool(np.isfinite(correction) and correction<=SITE_CORRECTION_TOLERANCE_MM and source_error<=1e-8)
    report['qualified']=qualified
    if qualified:
        # A separate source-compatible mass-only estimate, never the acceptance direction.
        mass_K=diags(np.repeat(solver.solid.mass/dt**2,3))
        mass_direction,_,mass_report=solve_kkt(mass_K,-np.asarray(mixed_gradient).ravel(),C,equilibrate=True)
        mass_direction=mass_direction.reshape(q.shape)
        mass_correction=float(solver.guard.max_physical_site_difference(q,q+mass_direction))
        report.update(source_constrained_mass_only_site_correction_mm=mass_correction,
            source_constrained_mass_only_velocity_correction_mm_s=mass_correction/dt,
            mass_only_indicator_scope='M delta/dt^2 + C_source^T eta = -full dynamic gradient; C_source delta=0. Diagnostic only, not nonlinear error proof',
            mass_only_linear_system=mass_report)
    return report
