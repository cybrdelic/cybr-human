"""Recognize only the independently qualified, numerical-only tangent revision.

Its sources are not claimed as executed by the geometric auditor. Their exact
identities are bound through the audited mechanics report and paired controller.
This module does not import or execute any contact or dynamics implementation.
"""
REVISION='interior_signed_distance_curvature_v1'
FACTORY_SHA='0b7f1375726a6d6bdab2ade476ec9a0bacc32fdeb1750af19023d17688a24a71'
CURVATURE_SHA='2383ac35c939260e85d3e39654f92331be15db3477812b21d2bc84b31f8372e1'
BASE_FACTORY_SHA='b0bee9a0be7458299d39cc69b14c42ccf4ed3359b47ba5768c7b988957e9cc1e'
QUALIFIED_SOURCES={
    'contact/interior-feature-tangent-v1/coarse_physical_site_contact_curvature.py':FACTORY_SHA,
    'contact/interior-feature-tangent-v1/interior_feature_curvature.py':CURVATURE_SHA,
    'contact/coarse_physical_site_contact_epigraph.py':BASE_FACTORY_SHA}


def validate_epigraph_factory_identity(report,paired_cursor_snapshot=None):
    provenance=report['coarse_contact_provenance']
    revision=provenance.get('interior_feature_tangent_revision')
    if revision is None:
        if any(key in provenance for key in ('interior_feature_tangent_source_sha256',
                'interior_feature_curvature_source_sha256','base_epigraph_factory_sha256')):
            raise ValueError('Partial numerical tangent identity lacks its qualified revision')
        if provenance['exterior_branch_provider_sha256']!=report['guard_module_sha256']:
            raise ValueError('Epigraph factory hash mismatch')
        return None
    expected={'interior_feature_tangent_revision':REVISION,
        'interior_feature_tangent_source_sha256':FACTORY_SHA,
        'interior_feature_curvature_source_sha256':CURVATURE_SHA,
        'base_epigraph_factory_sha256':BASE_FACTORY_SHA,
        'exterior_branch_provider_sha256':BASE_FACTORY_SHA,
        'unchanged_energy_gradient_guards':True}
    if any(provenance.get(key)!=value for key,value in expected.items()) or \
            report.get('guard_module_sha256')!=FACTORY_SHA:
        raise ValueError('Numerical tangent does not match the frozen qualified source/physical-law identities')
    if not isinstance(paired_cursor_snapshot,dict):
        raise ValueError('Qualified tangent input requires its captured admitted controller source manifest')
    sources=paired_cursor_snapshot.get('document',{}).get('source_sha256',{})
    if not isinstance(sources,dict):
        raise ValueError('Missing captured controller source manifest')
    bindings={}
    for suffix,expected_sha in QUALIFIED_SOURCES.items():
        selected=[(path,value) for path,value in sources.items()
            if isinstance(path,str) and (path==suffix or path.endswith('/'+suffix))]
        if len(selected)!=1 or selected[0][1]!=expected_sha:
            raise ValueError('Captured controller does not bind the qualified tangent and unchanged exterior provider')
        bindings[selected[0][0]]=selected[0][1]
    return {'revision':REVISION,'qualified_factory_sha256':FACTORY_SHA,
        'qualified_curvature_helper_sha256':CURVATURE_SHA,'unchanged_exterior_provider_sha256':BASE_FACTORY_SHA,
        'controller_source_manifest_bindings':bindings,'numerical_tangent_only':True,
        'energy_gradient_guards_unchanged':True,'geometric_auditor_executes_this_tangent':False,
        'scope':'Only the numerical search tangent changed; all physical acceptance and rendering gates are retained.'}
