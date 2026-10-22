"""Narrow provenance bridge for an unchanged-law numerical tangent correction."""
from pathlib import Path
import copy
import math
from source_transfer import sha

EPIGRAPH_SHA256 = 'b0bee9a0be7458299d39cc69b14c42ccf4ed3359b47ba5768c7b988957e9cc1e'
REVISION = 'interior_signed_distance_curvature_v1'


def physical_provenance(provenance):
    """Exclude only measured wall-clock setup cost, not a physical parameter."""
    result = copy.deepcopy(provenance)
    for path in TIMING_PATHS:
        owner = result
        for key in path[:-1]:
            owner = owner.get(key, {})
        if path[-1] in owner:
            value = owner.pop(path[-1])
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid recorded setup duration: '+'.'.join(path))
    return result


TIMING_PATHS = [('rigid_query', 'build_seconds'),
                ('initial_site_state', 'site_query_seconds'), ('setup_seconds',)]


def validate_curvature_transition(prior, provenance, module_path, previous_guard_sha256=None,
                                  *, old_qp_import_requested=False):
    """A new Hessian may reuse q, but never the old QP or its multipliers."""
    current = sha(Path(module_path))
    if (provenance.get('interior_feature_tangent_revision') != REVISION
            or provenance.get('interior_feature_tangent_source_sha256') != current
            or provenance.get('base_epigraph_factory_sha256') != EPIGRAPH_SHA256
            or provenance.get('exterior_branch_provider_sha256') != EPIGRAPH_SHA256
            or provenance.get('unchanged_energy_gradient_guards') is not True):
        raise ValueError('Unqualified interior-feature tangent identity')
    if prior is None:
        return None
    old = prior['guard_module_sha256']
    old_provenance = physical_provenance(prior['coarse_contact_provenance'])
    new_provenance = physical_provenance(provenance)
    if old == current:
        if old_provenance != new_provenance:
            raise ValueError('Changed provenance within the same tangent factory')
        return None
    if previous_guard_sha256 != old or old != EPIGRAPH_SHA256:
        raise ValueError('Only the frozen epigraph-to-curvature transition is supported')
    if old_qp_import_requested:
        raise ValueError('A changed tangent requires q-only restart, never the old QP')
    for key, value in old_provenance.items():
        if new_provenance.get(key) != value:
            raise ValueError('Tangent transition changed inherited provenance: '+key)
    return dict(previous_guard_sha256=old, current_guard_sha256=current,
        scope='Reviewed positive interior edge/vertex signed-distance Hessian term only; unchanged energy, gradient, geometry and guards',
        source_history_dt_mu_support_mass_unchanged=True,
        old_qp_state_reused=False, numerical_q_only_restart=True,
        nonphysical_timing_fields_excluded=['.'.join(p) for p in TIMING_PATHS],
        previous_native_setup_seconds=prior['coarse_contact_provenance'].get('rigid_query', {}).get('build_seconds'),
        current_native_setup_seconds=provenance.get('rigid_query', {}).get('build_seconds'),
        interior_feature_tangent_revision=REVISION)


def checkpoint_guard_sha(report, transition):
    return transition['previous_guard_sha256'] if transition else report['guard_module_sha256']
