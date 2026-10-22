"""One-use search policy; this never supplies or qualifies physical forces."""


def require_full_next_search(*, search_only, backtracks, site_increment_mm, objective_decrease_N_mm):
    # An accepted full-QP move consumes the request. It is not a permanent mode.
    return bool(search_only and (backtracks >= 4 or site_increment_mm < .001
        or objective_decrease_N_mm < 1e-8))
