"""Fixed 0.02 mm chord-only allowance; unchanged physical endpoints and forces."""
from reconstruction_seed import audited_passive_endpoints

RECONSTRUCTION_FIELDS = dict(reconstruction_contract='fixed_0p02mm_interframe_allowance_v1',
    physical_endpoint_penetration_cap_mm=.05, interframe_reconstruction_allowance_mm=.02,
    certified_interframe_penetration_cap_mm=.07,
    reconstruction_allowance_scope='Fixed inter-frame geometric reconstruction allowance, not a total temporal or state error bound')
