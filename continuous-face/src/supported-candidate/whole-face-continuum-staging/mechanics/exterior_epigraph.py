"""Law-preserving finite exterior max-energy SQP, with eliminated epigraphs.

Seam rows are numerical contact optimality equations, never attachment pins.
Inactive finite branches remain inequalities. No contact energy is added.
"""
import hashlib
import time
import numpy as np
from scipy.linalg import qr
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import splu
from linear_equilibration import solve_kkt
from factored_seam_kkt import FactoredSeamKKT


def normalize_branches(data, ndof):
    if isinstance(data, dict):
        return data
    energies, groups, bases, rows, cols, values = [], [], [], [], [], []
    site, owner, triangle = [], [], []
    for k, bundle in enumerate(data):
        e = np.asarray(bundle['energies_N_mm'], float)
        g = np.asarray(bundle['gradients_N'], float)
        nodes = np.asarray(bundle['nodes'], int)
        offset = len(energies)
        energies.extend(e); groups.extend([k]*len(e))
        bases.append(offset+int(bundle['selected_branch_index']))
        rows.extend(np.repeat(np.arange(offset, offset+len(e)), 12))
        cols.extend(np.tile((nodes[:, None]*3+np.arange(3)).ravel(), len(e)))
        values.extend(g.ravel())
        site.extend([int(bundle.get('site', bundle.get('site_index', -1)))]*len(e))
        owner.extend([int(bundle.get('owner', bundle.get('owner_id', -1)))]*len(e))
        triangle.extend(np.asarray(bundle.get('source_faces', np.full(len(e), -1)), int))
    G = coo_matrix((values, (rows, cols)), shape=(len(energies), ndof)).tocsr()
    return dict(branch_energy_N_mm=np.asarray(energies), branch_gradient_N=G,
        branch_group=np.asarray(groups, int), group_base_branch=np.asarray(bases, int),
        branch_site=np.asarray(site), branch_owner=np.asarray(owner), branch_triangle=np.asarray(triangle))


from compact_seam_rank import independent_seams as _independent_seams


class EpigraphOperationBudget(ValueError):
    def __init__(self, checkpoint):
        super().__init__('Exterior working-set computational budget; resumable QP, not contact infeasibility')
        self.checkpoint = checkpoint


def _qp_signature(K, C, gradient, E, G, groups, original):
    h = hashlib.sha256()
    for M in (csr_matrix(K), C, G):
        for a in (np.asarray(M.shape, np.int64), M.indptr, M.indices, M.data):
            h.update(np.ascontiguousarray(a).tobytes())
    for a in (gradient, E, groups, original):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def epigraph_direction(K, gradient, C, data, *, equilibrate=True,
                       energy_tolerance_N_mm=1e-12, weight_tolerance=1e-10,
                       maximum_active_set_iterations=1024, progress_callback=None,
                       max_wall_seconds=None, resume_checkpoint=None,
                       inexact_search_after_operations=None, source_factor_cache=None,
                       source_factor_context=None, numerical_regularization=0.,
                       fresh_source_factor_callback=None):
    """Solve the convex-branch linearized epigraph QP using sparse KKT.

    The supplied K is a search Hessian. True original max energy remains the
    line-search objective; a QP mixture is NOT automatically a valid current
    physical force. Current co-maximality and complementarity are audited below.
    """
    gradient = np.asarray(gradient, float).ravel()
    n = len(gradient)
    C = csr_matrix((0, n)) if C is None else csr_matrix(C)
    data = normalize_branches(data, n)
    E = np.asarray(data['branch_energy_N_mm'], float)
    G = csr_matrix(data['branch_gradient_N'])
    groups = np.asarray(data['branch_group'], int)
    original = np.asarray(data['group_base_branch'], int)
    if G.shape != (len(E), n) or groups.shape != E.shape or not np.isfinite(E).all() or not np.isfinite(G.data).all():
        raise ValueError('Finite shape-matched exterior branch data required')
    members = [np.flatnonzero(groups == k) for k in range(len(original))]
    for k, ids in enumerate(members):
        if not len(ids) or original[k] not in ids or abs(E[original[k]]-E[ids].max()) > energy_tolerance_N_mm:
            raise ValueError('Exterior group base is not its exact current max-energy branch')
    remainder = gradient - np.asarray(G[original].sum(axis=0)).ravel()
    signature = _qp_signature(K, C, gradient, E, G, groups, original)
    active = [[int(b)] for b in original]
    trace = []
    final_linear = None
    d = np.zeros(n)
    t = np.array([E[ids].max() for ids in members])
    if resume_checkpoint is not None:
        if resume_checkpoint['qp_signature'] != signature:
            raise ValueError('QP checkpoint equations differ; never reuse changed physical/search history')
        active = [list(map(int, a)) for a in resume_checkpoint['active_branches']]
        if len(active) != len(original) or any(len(a) != len(set(a)) for a in active):
            raise ValueError('QP checkpoint active groups/duplicates invalid')
        d = np.asarray(resume_checkpoint['direction'], float)
        t = np.asarray(resume_checkpoint['epigraph_values'], float)
        trace = list(resume_checkpoint['active_set_trace'])
        if d.shape != (n,) or t.shape != (len(original),) or not np.isfinite(d).all() or not np.isfinite(t).all():
            raise ValueError('Invalid finite QP checkpoint shape')
        if np.max(abs(C@d), initial=0) > 1e-10:
            raise ValueError('QP checkpoint violates original source constraints')
        if np.max(E+G@d-t[groups], initial=0) > 10*energy_tolerance_N_mm:
            raise ValueError('QP checkpoint is not primal feasible')
        for k, a in enumerate(active):
            if not a or any(j not in members[k] for j in a) or np.max(abs(E[a]+G[a]@d-t[k])) > 10*energy_tolerance_N_mm:
                raise ValueError('QP checkpoint working branches are not tight')
    started = time.perf_counter()
    factored = None
    first_iteration = len(trace)
    seen_feasible_working_states = set()
    def model_value(direction=None, epigraph=None):
        direction = d if direction is None else direction
        epigraph = t if epigraph is None else epigraph
        return float(.5*direction@(K@direction)+remainder@direction+epigraph.sum())
    initial_model = model_value()
    zero_model = float(sum(E[ids].max() for ids in members))
    def actual_slope(direction):
        value = float(remainder@direction)
        for ids in members:
            winning = ids[E[ids].max()-E[ids] <= energy_tolerance_N_mm]
            value += float(np.max(G[winning]@direction))
        return value
    def search_only_result(candidate=None):
        lifted = candidate is not None
        if inexact_search_after_operations is None or (not lifted and len(trace) < inexact_search_after_operations):
            return None
        direction = d if candidate is None else np.asarray(candidate, float)
        # Lifting only auxiliary epigraph values preserves every affine branch
        # inequality. It does not modify the feasible working-set state d,t.
        predicted = E+G@direction
        epigraph = np.asarray([predicted[ids].max() for ids in members]) if lifted else t
        violation = float(np.max(predicted-epigraph[groups], initial=0))
        source_error = float(np.max(abs(C@direction), initial=0))
        slope = actual_slope(direction)
        model = model_value(direction, epigraph)
        if (not np.isfinite([slope, model, zero_model, violation, source_error]).all() or not np.isfinite(epigraph).all()
                or not np.isfinite(direction).all() or source_error > 1e-10
                or violation > 10*energy_tolerance_N_mm
                or not model < zero_model or not slope < 0):
            return None
        evidence = dict(kind='finite_exterior_energy_epigraph_search_only_v1',
            qp_solved=False, search_only=True, usable_for_generalized_stationarity=False,
            search_candidate_kind='lifted_equality_candidate' if lifted else 'feasible_working_iterate',
            candidate_lift_uses_all_retained_branches=lifted,
            energy_law_unchanged=True, seam_rows_are_source_support_pins=False,
            actual_comaximality_satisfied=False, unfinished_multipliers_used=False,
            branch_count=len(E), group_count=len(original), mixed_groups=[],
            source_direction_constraint_residual_max_mm=source_error,
            maximum_predicted_branch_violation_N_mm=max(0., violation),
            qp_model_decrease_N_mm=zero_model-model,
            actual_max_energy_directional_derivative_N_mm=slope,
            active_set_trace=trace)
        current_trace = trace[first_iteration:]
        linear = dict(method='Feasible decreasing full max-QP candidate used ONLY for true-energy guarded search',
            finite_solution=True, solve_seconds=sum(x['linear_solve_seconds'] for x in current_trace),
            exterior_active_set_iterations=len(trace),
            exterior_sparse_factorizations=sum(x.get('factorization_count', 1) for x in current_trace),
            resumed_prior_qp_operations=first_iteration, qp_solved=False, search_only=True,
            source_direction_constraint_residual_max_mm=source_error)
        return direction.copy(), gradient.copy(), slope, linear, evidence
    def finish(result):
        if fresh_source_factor_callback is not None and factored is not None and numerical_regularization == 0.:
            fresh_source_factor_callback(factored)
        if source_factor_cache is not None:
            source_factor_cache.remember(factored, context=source_factor_context,
                equilibrate=equilibrate, regularization=numerical_regularization)
        return result
    if source_factor_cache is not None:
        cached = source_factor_cache.take_direction(gradient, C,
            context=source_factor_context, equilibrate=equilibrate,
            regularization=numerical_regularization,
            allowed=inexact_search_after_operations is not None and resume_checkpoint is None)
        if cached is not None:
            candidate, inverse_seconds = cached
            search = search_only_result(candidate)
            if search is not None:
                search[3].update(solve_seconds=inverse_seconds,
                    cached_source_factor_inverse_calls=1)
                search[4].update(search_candidate_kind='previous_source_factor_modified_newton',
                    cached_source_factor_inverse_calls=1, current_equations_solved=False,
                    cached_multipliers_used=False)
                return search
            # The consumed old factor is gone before a fresh current factor.
    def checkpoint():
        return dict(qp_signature=signature, direction=d.tolist(), epigraph_values=t.tolist(),
            active_branches=active, active_set_trace=trace, initial_model_N_mm=initial_model,
            current_model_N_mm=model_value(), numerical_QP_only=True, usable_as_accepted_frame=False)
    # Feasible primal active set. The eliminated epigraph increment is
    # dt_group=G_base*p. Working seams are homogeneous because they already
    # meet at the current feasible QP point, not at an incompatible old offset.
    for local_iteration in range(maximum_active_set_iterations):
        iteration = first_iteration+local_iteration
        search = search_only_result()
        if search is not None:
            return finish(search)
        if factored is None:
            factored = FactoredSeamKKT(K, C, equilibrate=equilibrate)
        if max_wall_seconds is not None and time.perf_counter()-started > max_wall_seconds:
            raise EpigraphOperationBudget(checkpoint())
        if progress_callback is not None and local_iteration % 8 == 0:
            progress_callback(dict(phase='epigraph_working_set', active_set_operation=iteration,
                source_factorizations=1, cached_inverse_seam_columns=len(factored.columns),
                qp_elapsed_seconds=time.perf_counter()-started, qp_model_N_mm=model_value(),
                active_branches=sum(map(len, active))))
        state_hash = hashlib.sha256(np.ascontiguousarray(d).tobytes()+np.ascontiguousarray(t).tobytes()).digest()
        state_key = (tuple(tuple(a) for a in active), state_hash)
        if state_key in seen_feasible_working_states:
            raise ValueError('Repeated identical feasible QP point and working set: numerical cycle, not anatomical infeasibility')
        seen_feasible_working_states.add(state_key)
        bases = np.array([a[0] for a in active], int)
        extra = [(k, j) for k, a in enumerate(active) for j in a[1:]]
        A = vstack([G[j]-G[bases[k]] for k, j in extra], format='csr') if extra else csr_matrix((0, n))
        selected = _independent_seams(A, np.zeros(len(extra)), C)
        combined = vstack([C, A[selected]], format='csr')
        base_gradient = remainder+K@d+np.asarray(G[bases].sum(axis=0)).ravel()
        try:
            p, multipliers, linear = factored.solve(-base_gradient, A[selected])
        except (ValueError, np.linalg.LinAlgError) as exc:
            p, multipliers, linear = solve_kkt(K, -base_gradient, combined,
                np.zeros(combined.shape[0]), equilibrate=equilibrate)
            linear.update(schur_full_kkt_fallback=True, schur_failure=str(exc), factorization_count=1,
                original_source_constraint_residual_max_mm=float(np.max(abs(C@p), initial=0)),
                original_seam_constraint_residual_max_N_mm=float(np.max(abs(A[selected]@p), initial=0)))
        # A poorly conditioned Schur cancellation falls back to the exact full
        # KKT, never changes stiffness, drops a seam, or hides residuals.
        if (linear['original_primal_residual_max_N'] > 1e-10*max(1., np.max(abs(base_gradient), initial=0))
                or linear['original_source_constraint_residual_max_mm'] > 1e-10
                or linear['original_seam_constraint_residual_max_N_mm'] > 1e-12):
            old_linear = linear
            p, multipliers, linear = solve_kkt(K, -base_gradient, combined,
                np.zeros(combined.shape[0]), equilibrate=equilibrate)
            linear.update(schur_full_kkt_fallback=True,
                factorization_count=old_linear['factorization_count']+1,
                solve_seconds=linear['solve_seconds']+old_linear['solve_seconds'],
                schur_residual_before_fallback=old_linear)

        if not linear['finite_solution']:
            raise ValueError('Nonfinite exterior epigraph direction')
        rho = np.zeros(len(extra)); rho[selected] = multipliers[C.shape[0]:]
        weights = np.zeros(len(E)); weights[bases] = 1.
        for r, (k, j) in zip(rho, extra):
            weights[j] += r; weights[bases[k]] -= r
        trace.append(dict(iteration=iteration, active_branches=sum(map(len, active)),
            independent_seam_rows=len(selected), dependent_seam_rows=len(extra)-len(selected),
            minimum_weight=float(weights.min(initial=0)), linear_solve_seconds=linear['solve_seconds'],
            factorization_count=linear.get('factorization_count', 1),
            schur_full_kkt_fallback=linear.get('schur_full_kkt_fallback', False)))
        qp_stationary = np.max(np.abs(K@p), initial=0) <= 1e-12
        if not qp_stationary:
            # A useful equality candidate need not stop at each inactive QP
            # branch: exact max lifting makes it a feasible SEARCH candidate.
            # In full-QP mode this is disabled. Rejection leaves d,t untouched.
            search = search_only_result(d+p)
            trace[-1]['lifted_candidate_search_qualified'] = search is not None
            if search is not None:
                return finish(search)
            dt = np.asarray(G[bases]@p).ravel()
            rates = np.asarray(G@p).ravel()-dt[groups]
            slack = t[groups]-E-G@d
            inactive = np.ones(len(E), bool)
            inactive[np.concatenate([np.asarray(x, int) for x in active])] = False
            blocking = np.flatnonzero(inactive & (rates > energy_tolerance_N_mm))
            ratios = np.maximum(slack[blocking], 0.)/rates[blocking]
            blocked = bool(len(ratios) and ratios.min() < 1.)
            alpha = float(ratios.min()) if blocked else 1.
            d += alpha*p; t += alpha*dt
            trace[-1]['qp_fraction_to_inactive_branch'] = alpha
            if blocked:
                j = int(blocking[np.argmin(ratios)])
                active[int(groups[j])].append(j)
                continue
        # A full unblocked step reaches this working-set optimum. Only now
        # inspect reaction bounds; a negative weight releases that branch.
        negative = np.flatnonzero(weights < -weight_tolerance)
        if len(negative):
            j = int(negative[np.argmin(weights[negative])]); k = int(groups[j])
            active[k].remove(j)
            trace[-1]['released_branch'] = j
            continue
        predicted = E+G@d
        violation = float(np.max(predicted-t[groups], initial=0))
        if violation > 10*energy_tolerance_N_mm:
            raise ValueError('Exterior feasible active-set lost linearized epigraph feasibility')
        final_linear = linear
        final_linear['original_constraint_residual_max_mixed_units'] = final_linear.pop('original_constraint_residual_max_mm')
        final_linear['constraint_residual_scope'] = 'Mixed-unit algebraic KKT metric; source mm and seam Nmm are reported separately'
        final_linear['source_direction_constraint_residual_max_mm'] = float(np.max(np.abs(C@d), initial=0))
        final_linear['seam_search_residual_max_N_mm'] = float(np.max(np.abs(A@p), initial=0))
        break
    if final_linear is None:
        raise EpigraphOperationBudget(checkpoint())
    weights = np.maximum(weights, 0.)
    weights[weights < weight_tolerance] = 0.
    for ids in members:
        weights[ids] /= weights[ids].sum()
    mixed = remainder + G.T@weights
    gap = np.zeros(len(E))
    slope = float(remainder@d)
    mixed_groups = []
    for k, ids in enumerate(members):
        gap[ids] = E[ids].max()-E[ids]
        maxima = ids[gap[ids] <= energy_tolerance_N_mm]
        slope += float(np.max(G[maxima]@d))
        positive = ids[weights[ids] > 0]
        if len(positive) > 1:
            record = dict(group=k, branch_indices=positive.tolist(), weights=weights[positive].tolist(),
                branch_energy_gaps_N_mm=gap[positive].tolist())
            for name in ('branch_site', 'branch_owner', 'branch_triangle'):
                if name in data:
                    record[name] = np.asarray(data[name])[positive].tolist()
            mixed_groups.append(record)
    complementarity = weights*gap
    used = weights > 0
    comax = bool(np.max(gap[used], initial=0) <= energy_tolerance_N_mm)
    report = dict(kind='finite_exterior_energy_epigraph_v1', energy_law_unchanged=True,
        qp_solved=True, search_only=False, usable_for_generalized_stationarity=True,
        seam_rows_are_source_support_pins=False, branch_weights_nonnegative=True,
        roundoff_weight_truncation_threshold=weight_tolerance,
        group_weight_sum_error=float(max((abs(weights[ids].sum()-1) for ids in members), default=0.)),
        actual_comaximality_satisfied=comax,
        maximum_used_branch_energy_gap_N_mm=float(np.max(gap[used], initial=0)),
        max_branch_complementarity_N_mm=float(complementarity.max(initial=0)),
        sum_branch_complementarity_N_mm=float(complementarity.sum()),
        comaximality_tolerance_N_mm=energy_tolerance_N_mm,
        branch_count=len(E), group_count=len(original), mixed_groups=mixed_groups,
        active_set_trace=trace, maximum_predicted_branch_violation_N_mm=max(0., violation),
        numerical_search_tangent='Original material plus mass and PSD selected-contact tangent; true max energy unchanged')
    final_linear = dict(final_linear, solve_seconds=sum(t['linear_solve_seconds'] for t in trace[first_iteration:]),
        exterior_active_set_iterations=len(trace),
        exterior_sparse_factorizations=sum(t.get('factorization_count', 1) for t in trace[first_iteration:]),
        resumed_prior_qp_operations=first_iteration, qp_solved=True, search_only=False)
    return finish((d, np.asarray(mixed).ravel(), slope, final_linear, report))
