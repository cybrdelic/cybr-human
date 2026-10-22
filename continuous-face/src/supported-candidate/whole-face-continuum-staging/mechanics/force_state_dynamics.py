"""Sparse backward Euler for the shared continuum, never a neutral-fit solve.

Real frame production is fail-closed until finite contact energy, endpoint and
whole-segment geometry guards are supplied. The included tests use only an
explicit obstacle-free synthetic fixture and cannot certify facial geometry.
"""
from __future__ import annotations

import time
from force_state_accuracy import accuracy_fallback, POLICY
import numpy as np
from scipy.linalg import qr
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
from linear_equilibration import solve_kkt
from epigraph_search_policy import require_full_next_search
from modified_newton_search import ModifiedNewtonSearchCache, search_context_identity
from barrier_slack_search import make_slack_anchor, check_slack_retention
from exterior_epigraph import epigraph_direction, EpigraphOperationBudget
from second_order_gap_search import make_gap_anchor, gap_corrected_candidate
from raw_predictor_geometry import cap_only_raw_endpoint


class MissingGeometryCertification(RuntimeError):
    pass


REQUIRED_GUARDS = ("source_initial_state", "rigid_bone_cartilage", "lumens",
                   "separated_exterior_sheets", "self_collision", "finite_contact_forces",
                   "whole_segment_collision", "finite_source_origin_attachments")
COARSE_SITE_GUARDS = ("coarse_physical_site_initial_state", "coarse_native_site_contact",
                      "positive_carrier_volume", "bounded_physical_site_motion",
                      "coarse_site_path_checks", "finite_source_origin_attachments")


class OriginConstraints:
    """C q = C X at verified source origin material sites, not exterior targets.

    Scalar rows apply identically to all three coordinates. Rank reduction
    removes dependent constraints, not material DOFs or physical tissue volume.
    """
    def __init__(self, matrix, reference, *, provenance, rank_rtol=1e-10):
        C = csr_matrix(matrix, dtype=float)
        self.reference = np.asarray(reference, float)
        if C.shape[1] != len(reference) or not np.isfinite(C.data).all():
            raise ValueError("finite shape-matched source-origin map required")
        if not np.allclose(np.asarray(C.sum(1)).ravel(), 1., atol=1e-10, rtol=0):
            raise ValueError("source origins require barycentric partition of unity")
        if not provenance or not provenance.get("source_origin_only"):
            raise ValueError("verified source-origin ownership provenance required")
        used = np.unique(C.indices)
        if C.shape[0]:
            # Only the support columns enter dense rank revelation, never the
            # whole-face stiffness. O(m*n_support), hundreds of source origins.
            _, R, piv = qr(C[:, used].toarray().T, mode="economic", pivoting=True)
            diag = np.abs(np.diag(R))
            rank = int(np.sum(diag > rank_rtol*max(diag.max(initial=0), 1e-30)))
            keep = np.sort(piv[:rank])
        else:
            rank, keep = 0, np.empty(0, int)
        self.C = C[keep]
        self.selected_source_rows = np.asarray(keep, int)
        self.target = self.C @ self.reference
        self.C3 = kron(self.C, eye(3), format="csr")
        self.gram = splu((self.C @ self.C.T).tocsc()) if rank else None
        self.report = dict(candidate_scalar_rows=C.shape[0], rank_scalar=rank,
            constrained_coordinate_rank=3*rank, removed_dependent_rows=C.shape[0]-rank,
            affected_carrier_nodes=len(used), rank_rtol=rank_rtol,
            constraint_model="ideal rigid source-origin bonding with retained source rest offsets",
            source=provenance)

    def projected_residual(self, gradient):
        if self.gram is None:
            return np.asarray(gradient).copy(), np.zeros_like(gradient)
        reaction = self.C.T @ self.gram.solve(self.C @ gradient)
        return gradient-reaction, reaction

    def error(self, q):
        return float(np.max(np.abs(self.C @ q-self.target), initial=0))


class ImplicitDynamics:
    def __init__(self, solid, guard, *, constraints=None, gravity=(0., -9810., 0.),
                 damping_per_second=0., jacobian_floor=.1, segment_safety=.9,
                 equilibrate_linear_system=True, guard_scope="finite_boundary"):
        self.solid, self.guard, self.constraints = solid, guard, constraints
        self.gravity = np.asarray(gravity, float)
        self.damping = float(damping_per_second)
        self.jacobian_floor, self.segment_safety = float(jacobian_floor), float(segment_safety)
        self.equilibrate_linear_system = bool(equilibrate_linear_system)
        if guard_scope not in ("finite_boundary", "coarse_physical_sites"):
            raise ValueError("Unknown explicitly selected contact resolution")
        self.guard_scope = guard_scope
        if self.gravity.shape != (3,) or not np.isfinite(self.gravity).all():
            raise ValueError("finite gravity vector required")
        if not np.isfinite(self.damping) or self.damping < 0 or not 0 <= jacobian_floor < 1 or not 0 < segment_safety < 1:
            raise ValueError("invalid damping/J/path safety")

    def require_guards(self):
        certification = getattr(self.guard, "certification", {})
        required = COARSE_SITE_GUARDS if self.guard_scope == "coarse_physical_sites" else REQUIRED_GUARDS
        missing = [x for x in required if certification.get(x) is not True]
        if getattr(self.guard, "fixture_only", False) and not self.solid.provenance.get("kind", "").startswith("synthetic"):
            raise MissingGeometryCertification("Synthetic fixture guards cannot be used on actual facial geometry")
        if missing:
            raise MissingGeometryCertification("No accepted motion: missing "+", ".join(missing))

    def path_is_valid(self, evidence):
        key = "coarse_whole_segment_checked" if self.guard_scope == "coarse_physical_sites" else "whole_segment_certified"
        return evidence.get("feasible", False) and evidence.get(key, False)

    def _project(self, g):
        return (g, np.zeros_like(g)) if self.constraints is None else self.constraints.projected_residual(g)

    def generalized_contact_detail(self, detail, q, selected_g, mixed_g, evidence,
                                   q_previous, prediction, dt):
        """Add bounded finite-branch reaction; project ONLY source supports."""
        out = dict(detail)
        mixed_g = np.asarray(mixed_g).reshape(q.shape)
        correction = mixed_g-selected_g
        projected, reaction = self._project(mixed_g)
        vmax = lambda x: float(np.max(np.linalg.norm(x, axis=1), initial=0))
        inertia = self.solid.mass[:, None]*(q-prediction)/dt**2
        damping = self.damping*self.solid.mass[:, None]*(q-q_previous)/dt
        physical = mixed_g-inertia-damping
        out.update(smooth_selected_branch_dynamic_residual_max_N=detail['dynamic_residual_max_N'],
            smooth_selected_branch_dynamic_residual_L2_N=detail['dynamic_residual_L2_N'],
            generalized_contact=evidence,
            contact_generalized_stationarity_qualified=bool(evidence['actual_comaximality_satisfied']),
            exterior_contact_stationarity='finite_exterior_energy_epigraph_v1',
            generalized_contact_gradient_correction_max_N=vmax(correction),
            dynamic_residual_max_N=vmax(projected), dynamic_residual_L2_N=float(np.linalg.norm(projected)),
            unbalanced_dynamic_gradient_max_N=vmax(mixed_g),
            unbalanced_dynamic_gradient_L2_N=float(np.linalg.norm(mixed_g)),
            support_reaction_max_N=vmax(reaction), physical_force_max_N=vmax(physical))
        if hasattr(self.guard, 'contact_accounting'):
            accounting = dict(self.guard.contact_accounting(q))
            penalty = np.asarray(accounting.pop('penalty_gradient_N'))
            barrier = np.asarray(accounting.pop('barrier_gradient_N'))
            material = selected_g-inertia-damping-penalty-barrier+self.solid.mass[:, None]*self.gravity
            penalty = penalty+correction
            accounting.update(quadratic_contact_force_net_N=(-penalty).sum(0).tolist(),
                quadratic_contact_force_representation='Valid convex mixture of actually co-maximal finite exterior features',
                finite_exterior_epigraph=evidence)
            out.update(contact_accounting=accounting, penalty_force_max_N=vmax(penalty),
                physical_force_without_barrier_max_N=vmax(physical-barrier))
            magnitudes = sum(np.linalg.norm(x, axis=1) for x in
                (material, self.solid.mass[:, None]*self.gravity, inertia, damping, penalty, barrier))
            scale_max, scale_l2 = float(magnitudes.max(initial=0)), float(np.linalg.norm(magnitudes))
            out.update(force_component_magnitude_sum_max_N=scale_max,
                force_component_magnitude_sum_L2_N=scale_l2,
                relative_dynamic_balance_max=out['dynamic_residual_max_N']/scale_max if scale_max else 0.,
                relative_dynamic_balance_L2=out['dynamic_residual_L2_N']/scale_l2 if scale_l2 else 0.)
        hot = int(np.argmax(np.linalg.norm(projected, axis=1)))
        out['dominant_dynamic_residual'] = dict(node=hot, reference_mm=self.solid.X[hot].tolist(),
            position_mm=q[hot].tolist(), projected_gradient_N=projected[hot].tolist())
        return out

    def evaluate(self, q, q_previous, prediction, dt, activation):
        J = np.linalg.det(self.solid.deformation(q))
        if not np.isfinite(J).all() or np.any(J <= self.jacobian_floor):
            return np.inf, np.full_like(q, np.nan), {"feasible": False, "reason": "positive-J guard"}
        endpoint = self.guard.endpoint(q)
        if not endpoint.get("feasible", False):
            return np.inf, np.full_like(q, np.nan), {"feasible": False, "reason": "finite endpoint guard", "endpoint": endpoint}
        e, g, _ = self.solid.energy(q, activation)
        ec, gc = self.guard.energy(q)
        physical_e = e+ec-float(np.sum(self.solid.mass[:, None]*self.gravity*q))
        physical_g = g+gc-self.solid.mass[:, None]*self.gravity
        inertia = self.solid.mass[:, None]*(q-prediction)/dt**2
        damping = self.damping*self.solid.mass[:, None]*(q-q_previous)/dt
        objective = physical_e+.5*np.sum(self.solid.mass[:, None]*(q-prediction)**2)/dt**2
        objective += .5*self.damping*np.sum(self.solid.mass[:, None]*(q-q_previous)**2)/dt
        dynamic_g = physical_g+inertia+damping
        projected, reaction = self._project(dynamic_g)
        vmax = lambda x: float(np.max(np.linalg.norm(x, axis=1), initial=0))
        detail = dict(feasible=bool(np.isfinite(objective) and np.isfinite(dynamic_g).all()),
            residual_projection_scope='source_origin_support_reactions_only',
            physical_energy_N_mm=float(physical_e), implicit_objective_N_mm=float(objective),
            physical_force_max_N=vmax(physical_g), inertial_force_max_N=vmax(inertia),
            damping_force_max_N=vmax(damping), support_reaction_max_N=vmax(reaction),
            unbalanced_dynamic_gradient_max_N=vmax(dynamic_g), dynamic_residual_max_N=vmax(projected),
            unbalanced_dynamic_gradient_L2_N=float(np.linalg.norm(dynamic_g)),
            dynamic_residual_L2_N=float(np.linalg.norm(projected)),
            material_active_force_max_N=vmax(g),
            gravity_force_max_N=vmax(self.solid.mass[:, None]*self.gravity),
            minimum_J=float(J.min()), endpoint=endpoint,
            support_constraint_error_mm=0. if self.constraints is None else self.constraints.error(q))
        if hasattr(self.guard, 'contact_accounting'):
            accounting = dict(self.guard.contact_accounting(q))
            barrier_g = np.asarray(accounting.pop('barrier_gradient_N'), float)
            penalty_g = np.asarray(accounting.pop('penalty_gradient_N'), float)
            detail.update(contact_accounting=accounting,
                penalty_force_max_N=vmax(penalty_g),
                barrier_reaction_max_N=vmax(barrier_g),
                physical_force_without_barrier_max_N=vmax(physical_g-barrier_g),
                physical_energy_without_barrier_N_mm=float(physical_e-accounting['barrier_energy_N_mm']),
                dynamic_residual_includes_barrier_reaction=True)
            magnitudes = sum(np.linalg.norm(x, axis=1) for x in
                (g, self.solid.mass[:, None]*self.gravity, inertia, damping, penalty_g, barrier_g))
            scale_max = float(magnitudes.max(initial=0))
            scale_L2 = float(np.linalg.norm(magnitudes))
            detail.update(force_component_magnitude_sum_max_N=scale_max,
                force_component_magnitude_sum_L2_N=scale_L2,
                relative_dynamic_balance_max=detail['dynamic_residual_max_N']/scale_max if scale_max else 0.,
                relative_dynamic_balance_L2=detail['dynamic_residual_L2_N']/scale_L2 if scale_L2 else 0.)
        hot = int(np.argmax(np.linalg.norm(projected, axis=1)))
        detail['dominant_dynamic_residual'] = dict(node=hot, reference_mm=self.solid.X[hot].tolist(),
            position_mm=q[hot].tolist(), projected_gradient_N=projected[hot].tolist())
        return float(objective), dynamic_g, detail

    def safe_volume_step_bound(self, q, d):
        F, dF = self.solid.deformation(q), self.solid.deformation(d)
        J = np.linalg.det(F)
        if np.any(J <= self.jacobian_floor):
            return 0.
        rho = np.linalg.norm(np.linalg.solve(F, dF), axis=(1, 2))
        moving = rho > 0
        if not np.any(moving):
            return 1.
        return min(1., float(np.min(self.segment_safety*(1-np.cbrt(self.jacobian_floor/J[moving]))/rho[moving])))

    def step(self, q_previous, velocity_previous, *, dt, activation=None,
             max_iterations=16, force_tolerance_N=1e-7, max_line_search=30,
             max_wall_seconds=None, progress_callback=None, initial_iterate=None,
             diagnostic_iterate_callback=None, defer_physical_frame_path=False,
             reuse_source_factor_search=False, allow_cap_only_gap_search=False):
        self.require_guards()
        started = time.perf_counter()
        contact_seconds_start = float(getattr(self.guard, 'query_seconds', 0.))
        contact_queries_start = int(getattr(self.guard, 'query_count', 0))
        if max_wall_seconds is not None and (not np.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
            raise ValueError("positive bounded wall-time budget required")
        over_budget = lambda: max_wall_seconds is not None and time.perf_counter()-started >= max_wall_seconds
        q_previous = np.asarray(q_previous, float)
        q = np.array(q_previous if initial_iterate is None else initial_iterate, float, copy=True)
        velocity_previous = np.asarray(velocity_previous, float)
        if (not np.isfinite(dt) or dt <= 0 or velocity_previous.shape != q.shape
                or q.shape != q_previous.shape or not np.isfinite(q).all()
                or not np.isfinite(velocity_previous).all()):
            raise ValueError("finite velocity and positive timestep required")
        if max_iterations < 0 or force_tolerance_N <= 0 or max_line_search < 1:
            raise ValueError("invalid solver budget/tolerance")
        if self.constraints is not None and self.constraints.error(q) > 1e-8:
            raise ValueError("initial coordinates violate source-origin bonding")
        # Warm starts change only the nonlinear iterate, never physical history.
        prediction = q_previous+dt*velocity_previous
        activation = self.solid.activation(activation)
        history, status, reg = [], "iteration_limit", 0.
        pending_qp = getattr(self, '_epigraph_qp_checkpoint', None)
        if pending_qp is not None:
            reg = float(pending_qp.get('numerical_relative_diagonal_regularization', 0.))
            if not np.isfinite(reg) or reg < 0:
                raise ValueError('Invalid restored search regularization')
        force_full_epigraph = bool(pending_qp and pending_qp.get('full_epigraph_required', False))
        final_generalized = None
        # Deliberately local: never saved on self, in a report, or a checkpoint.
        source_factor_cache = ModifiedNewtonSearchCache() if reuse_source_factor_search else None
        it = 0
        while it <= max_iterations:
            e, g, detail = self.evaluate(q, q_previous, prediction, dt, activation)
            if not detail["feasible"]:
                raise MissingGeometryCertification("Initial or accepted iterate is infeasible: "+str(detail))
            row = dict(iteration=it, elapsed_seconds=time.perf_counter()-started, **detail)
            history.append(row)
            if progress_callback is not None:
                progress_callback(dict(iteration=it, elapsed_seconds=time.perf_counter()-started,
                    implicit_objective_N_mm=detail['implicit_objective_N_mm'],
                    dominant_dynamic_residual=detail['dominant_dynamic_residual'],
                    dynamic_residual_max_N=detail['dynamic_residual_max_N'],
                    physical_force_max_N=detail['physical_force_max_N'], minimum_J=detail['minimum_J']))
            if detail["dynamic_residual_max_N"] <= force_tolerance_N:
                status = "converged"
                if hasattr(self.guard, 'contact_branches'):
                    # A single actual winning feature is itself a valid convex
                    # subgradient. No seam reaction is hidden in this case.
                    detail.update(contact_generalized_stationarity_qualified=True,
                        exterior_contact_stationarity='exact_selected_winner_already_stationary',
                        generalized_contact=dict(kind='exact_selected_winner_already_stationary',
                            energy_law_unchanged=True, seam_rows_are_source_support_pins=False,
                            branch_weights_nonnegative=True, group_weight_sum_error=0.,
                            actual_comaximality_satisfied=True,
                            maximum_used_branch_energy_gap_N_mm=0.,
                            max_branch_complementarity_N_mm=0., sum_branch_complementarity_N_mm=0.,
                            comaximality_tolerance_N_mm=1e-12, mixed_groups=[]))
                    final_generalized = (q.copy(), detail)
                break
            if it == max_iterations:
                break
            if over_budget():
                status = "wall_time_budget"
                break
            tangent_started = time.perf_counter()
            K = self.solid.tangent(q, activation)
            row['material_tangent_seconds'] = time.perf_counter()-tangent_started
            tangent_started = time.perf_counter()
            K += self.guard.tangent(q)
            row['contact_tangent_seconds'] = time.perf_counter()-tangent_started
            K += diags(np.repeat(self.solid.mass*(1/dt**2+self.damping/dt), 3))
            diag = np.maximum(np.abs(K.diagonal()), 1e-15)
            direction = None
            branch_started = time.perf_counter()
            try:
                branch_data = self.guard.contact_branches(q) if hasattr(self.guard, 'contact_branches') else None
            except (ValueError, AssertionError) as exc:
                row['exterior_epigraph_error'] = str(exc)
                status = 'epigraph_finite_branch_failed'
                break
            row['exterior_branch_build_seconds'] = time.perf_counter()-branch_started
            for attempt in range(8):
                if over_budget():
                    break
                fresh_source_factors = []
                numerical_K = K+diags(reg*diag)
                C = None if self.constraints is None else self.constraints.C3
                if branch_data is None:
                    result, _, linear_report = solve_kkt(numerical_K, -g.ravel(), C,
                        equilibrate=self.equilibrate_linear_system)
                    slope = float(g.ravel()@result)
                else:
                    try:
                        result, mixed_g, slope, linear_report, epigraph = epigraph_direction(
                            numerical_K, g.ravel(), C, branch_data,
                            equilibrate=self.equilibrate_linear_system,
                            progress_callback=(None if progress_callback is None else
                                lambda r: progress_callback(dict(iteration=it,
                                    elapsed_seconds=time.perf_counter()-started, **r))),
                            max_wall_seconds=None if max_wall_seconds is None else max(0., max_wall_seconds-(time.perf_counter()-started)),
                            resume_checkpoint=getattr(self, '_epigraph_qp_checkpoint', None),
                            inexact_search_after_operations=None if force_full_epigraph else 16,
                            source_factor_cache=source_factor_cache,
                            source_factor_context=(None if source_factor_cache is None else
                                search_context_identity(q_previous, velocity_previous, prediction,
                                    dt, activation, self.guard, self.constraints)),
                            numerical_regularization=reg, fresh_source_factor_callback=fresh_source_factors.append)
                        self._epigraph_qp_checkpoint = None
                    except EpigraphOperationBudget as exc:
                        exc.checkpoint['numerical_relative_diagonal_regularization'] = reg
                        exc.checkpoint['full_epigraph_required'] = force_full_epigraph
                        self._epigraph_qp_checkpoint = exc.checkpoint
                        row['epigraph_qp_checkpoint'] = exc.checkpoint
                        row['exterior_epigraph_error'] = str(exc)
                        status = 'epigraph_working_set_budget'
                        break
                    except (ValueError, AssertionError) as exc:
                        row['exterior_epigraph_error'] = str(exc)
                        status = 'epigraph_active_set_failed'
                        break
                    row['exterior_epigraph_search'] = epigraph
                    if epigraph.get('qp_solved', True) and epigraph['actual_comaximality_satisfied']:
                        generalized = self.generalized_contact_detail(detail, q, g, mixed_g,
                            epigraph, q_previous, prediction, dt)
                        row.update(generalized)
                        if progress_callback is not None:
                            progress_callback(dict(iteration=it, phase='generalized_contact_direction',
                                elapsed_seconds=time.perf_counter()-started,
                                dynamic_residual_max_N=generalized['dynamic_residual_max_N'],
                                actual_comaximality_satisfied=True,
                                active_set_factorizations=linear_report['exterior_sparse_factorizations'],
                                active_set_operations=linear_report['exterior_active_set_iterations'],
                                linear_solve_seconds=linear_report['solve_seconds'],
                                exterior_branch_build_seconds=row['exterior_branch_build_seconds']))
                        if generalized['dynamic_residual_max_N'] <= force_tolerance_N:
                            status = 'converged'
                            final_generalized = (q.copy(), generalized)
                            break
                        accuracy = accuracy_fallback(self, q, result, mixed_g, dt, generalized, epigraph, reg)
                        row['stationarity_accuracy'] = accuracy
                        if accuracy['qualified']:
                            generalized.update(stationarity_accuracy=accuracy, approximate_stationarity=True,
                                absolute_force_target_passed=False)
                            row.update(generalized)
                            status = 'converged'
                            final_generalized = (q.copy(), generalized)
                            break
                    elif progress_callback is not None:
                        progress_callback(dict(iteration=it, phase='epigraph_search_not_yet_comaximal',
                            elapsed_seconds=time.perf_counter()-started,
                            actual_comaximality_satisfied=False, qp_solved=epigraph.get('qp_solved', True),
                            search_only=epigraph.get('search_only', False),
                            active_set_factorizations=linear_report['exterior_sparse_factorizations'],
                                active_set_operations=linear_report['exterior_active_set_iterations'],
                            linear_solve_seconds=linear_report['solve_seconds'],
                            exterior_branch_build_seconds=row['exterior_branch_build_seconds']))
                d = result.reshape(q.shape)
                if np.isfinite(d).all() and slope < 0:
                    direction = d
                    break
                reg = max(.001, reg*10)
            if status == 'converged':
                row.update(linear_system=linear_report,
                    numerical_relative_diagonal_regularization=reg)
                break
            if status in ('epigraph_active_set_failed', 'epigraph_working_set_budget'):
                break
            if direction is None:
                status = "wall_time_budget" if over_budget() else "no_descent_direction"
                break
            alpha = min(self.safe_volume_step_bound(q, direction),
                float(self.guard.safe_step_bound(q, direction, safety=self.segment_safety)))
            row.update(numerical_relative_diagonal_regularization=reg, initial_safe_step=alpha,
                linear_system=linear_report)
            accepted = False
            row['line_search_trials'] = []
            slack_anchor = make_slack_anchor(self.guard, q)
            fresh_source_factor = fresh_source_factors[-1] if fresh_source_factors else None
            gap_anchor = make_gap_anchor(self.guard, q, fresh_source_factor)
            for backtrack in range(max_line_search):
                if over_budget():
                    break
                trial = q+alpha*direction
                if alpha <= 0 or np.array_equal(trial, q):
                    break
                ee, _, valid = self.evaluate(trial, q_previous, prediction, dt, activation)
                plain_trial, plain_energy, plain_valid = trial, ee, valid
                candidates = [(plain_trial, plain_energy, plain_valid, False)]
                soc_report = None
                if fresh_source_factor is not None and gap_anchor and (valid['feasible']
                        or (allow_cap_only_gap_search and cap_only_raw_endpoint(valid))):
                    try:
                        corrected, soc_report = gap_corrected_candidate(self.guard, q, plain_trial,
                            fresh_source_factor, gap_anchor, g, branch_data,
                            raw_endpoint=None if valid['feasible'] else valid)
                    except (ValueError, AssertionError, np.linalg.LinAlgError) as exc:
                        corrected, soc_report = None, dict(qualified=False, search_only=True, reason=str(exc))
                    if corrected is not None:
                        corrected_energy, _, corrected_detail = self.evaluate(corrected, q_previous, prediction, dt, activation)
                        corrected_valid = corrected_detail
                        candidates.insert(0, (corrected, corrected_energy, corrected_valid, True))
                for trial, ee, valid, used_soc in candidates:
                    armijo_slope = (soc_report['true_max_corrected_displacement_slope_N_mm']
                        if used_soc else alpha*slope)
                    constraint_ok = self.constraints is None or self.constraints.error(trial) <= 1e-8
                    armijo_ok = bool(ee <= e+1e-4*armijo_slope)
                    trial_record = dict(second_order_gap_search=soc_report, actual_search_candidate=('gap_defect_corrected' if used_soc else 'original_QP_direction'), alpha=float(alpha), endpoint_feasible=valid['feasible'],
                        armijo_pass=armijo_ok, constraint_pass=constraint_ok, reason=valid.get('reason'),
                        objective_N_mm=float(ee), endpoint=valid.get('endpoint', {}))
                    row['line_search_trials'].append(trial_record)
                    slack_check = (check_slack_retention(self.guard, trial, slack_anchor)
                        if valid["feasible"] else dict(qualified=False, reason='Infeasible endpoint'))
                    trial_record['barrier_slack_search'] = slack_check
                    if valid["feasible"] and constraint_ok and armijo_ok and slack_check['qualified']:
                        path = self.guard.path(q, trial)
                        trial_record['path_feasible'] = self.path_is_valid(path)
                        if self.path_is_valid(path):
                            if branch_data is not None and (epigraph.get('search_only', False) or used_soc):
                                site_increment = (self.guard.max_physical_site_difference(q, trial)
                                    if hasattr(self.guard, 'max_physical_site_difference') else float(np.linalg.norm(trial-q, axis=1).max()))
                                force_full_epigraph = require_full_next_search(search_only=True, backtracks=backtrack,
                                    site_increment_mm=site_increment, objective_decrease_N_mm=e-ee)
                                row['next_search_requires_full_epigraph'] = force_full_epigraph
                                row['actual_search_site_increment_mm'] = site_increment
                            elif branch_data is not None:
                                force_full_epigraph = require_full_next_search(search_only=False, backtracks=backtrack,
                                    site_increment_mm=None, objective_decrease_N_mm=e-ee)
                                row['next_search_requires_full_epigraph'] = False
                                row['full_search_requirement_consumed'] = True
                            row['actual_search_was_second_order_corrected'] = bool(used_soc)
                            q, accepted = trial, True
                            row.update(step=alpha, backtracks=backtrack, path=path)
                            break
                if accepted:
                    break
                alpha *= .5
            fresh_source_factor = None
            fresh_source_factors.clear()
            gap_anchor = None
            if not accepted:
                if (source_factor_cache is not None and branch_data is not None
                        and epigraph.get('search_candidate_kind') == 'previous_source_factor_modified_newton'
                        and not over_budget()):
                    source_factor_cache.clear()
                    force_full_epigraph = True
                    row['cached_search_failed_retry_fresh_full_qp'] = True
                    # Same q, history, dt, activation, reg, and nonlinear iteration.
                    continue
                status = "wall_time_budget" if over_budget() else "line_search_failed"
                break
            it += 1
        if source_factor_cache is not None:
            source_factor_cache.clear()
        _, _, detail = self.evaluate(q, q_previous, prediction, dt, activation)
        if final_generalized is not None and np.array_equal(q, final_generalized[0]):
            detail = final_generalized[1]
        frame_path = (dict(feasible=False, deferred_to_final_barrier_parameter_state=True,
            reason='Internal same-timestep stationarity only; no physical frame is committed here')
            if defer_physical_frame_path else self.guard.path(q_previous, q))
        accepted = status == "converged" and (defer_physical_frame_path or self.path_is_valid(frame_path))
        report = dict(status=status, accepted=bool(accepted), history=history, **detail,
            physical_frame_path_deferred=bool(defer_physical_frame_path),
            acceptance_scope='internal_same_timestep_stationarity_only' if defer_physical_frame_path else 'physical_frame_with_source_to_endpoint_path',
            warm_start_used=initial_iterate is not None,
            inertial_prediction='fixed q_previous + dt * velocity_previous throughout this solve',
            activation=activation, dt_s=float(dt), force_tolerance_N=force_tolerance_N,
            damping_model="C = damping_per_second * lumped physical mass",
            damping_per_second=self.damping, gravity_mm_s2=self.gravity.tolist(),
            mass_model="physical cut-volume barycentric row-sum mass, no inflated inertia",
            initial_state="source reference initialized with supplied velocity; transient settling, not static neutral",
            acceptance=("true incremental potential Armijo plus positive-J and finite endpoint/whole-path guards"
                if self.guard_scope == "finite_boundary" else
                "true incremental potential Armijo plus positive carrier-J, bounded occupied-site motion and coarse native-site endpoint/path checks; NOT whole-boundary collision certification"),
            guard_scope=self.guard_scope,
            exact_whole_boundary_collision_claim=self.guard_scope == "finite_boundary",
            frame_path=frame_path, fixture_only=bool(getattr(self.guard, "fixture_only", False)),
            support=None if self.constraints is None else self.constraints.report)
        report.setdefault('stationarity_accuracy', dict(policy=POLICY,route='absolute_force_target',
            qualified=bool(detail['dynamic_residual_max_N']<=force_tolerance_N),
            absolute_force_target_N=force_tolerance_N,absolute_force_target_passed=bool(detail['dynamic_residual_max_N']<=force_tolerance_N)))
        report.setdefault('approximate_stationarity',False)
        report.update(wall_seconds=time.perf_counter()-started, max_wall_seconds=max_wall_seconds,
            contact_query_seconds=float(getattr(self.guard, 'query_seconds', 0.))-contact_seconds_start,
            contact_query_count=int(getattr(self.guard, 'query_count', 0))-contact_queries_start,
            wall_budget_scope="checked between evaluations and linear solves; cannot interrupt an in-flight sparse factorization")
        # A failed nonlinear step is diagnostic, not a motion frame. Caller can
        # inspect the report but never gets the failed trial as advanced state.
        if not accepted:
            if diagnostic_iterate_callback is not None:
                diagnostic_iterate_callback(q.copy(), report)
            return np.asarray(q_previous).copy(), velocity_previous.copy(), report
        return q, (q-q_previous)/dt, report


def source_activation(time_s, *, case, ramp_s=.2):
    """Explicit test drive only; no displacement target or visual amplification."""
    ramp = float(np.clip(time_s/ramp_s, 0, 1))
    if case == "passive":
        return {}
    if case == "zygomaticus_frontalis":
        return {"zygomaticus": .3*ramp, "frontalis": .3*ramp}
    if case == "zygomaticus_corrugator":
        return {"zygomaticus": .3*ramp, "corrugator": .2*ramp}
    raise ValueError("unknown matched trajectory case")


def matched_trajectories(dynamics, *, times_s, q0=None, v0=None, **step_options):
    """Run identically initialized passive and source-labelled active cases.

    No 1e-6 static-neutral prerequisite. Caller persists only accepted frames;
    absence of complete geometry guards fails before producing any frame.
    """
    dynamics.require_guards()
    times = np.asarray(times_s, float)
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("strictly increasing times beginning at zero required")
    q0 = dynamics.solid.X.copy() if q0 is None else np.asarray(q0).copy()
    v0 = np.zeros_like(q0) if v0 is None else np.asarray(v0).copy()
    runs = {}
    for case in ("passive", "zygomaticus_frontalis", "zygomaticus_corrugator"):
        q, v, frames, reports = q0.copy(), v0.copy(), [], []
        for i in range(1, len(times)):
            q, v, report = dynamics.step(q, v, dt=times[i]-times[i-1],
                activation=source_activation(times[i], case=case), **step_options)
            reports.append(report)
            if not report["accepted"]:
                break
            frames.append(q.copy())
        runs[case] = dict(frames=frames, reports=reports, times_s=times[1:len(frames)+1].tolist())
    return runs
