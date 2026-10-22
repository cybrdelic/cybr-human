import unittest
from epigraph_search_policy import require_full_next_search


class SearchPolicyTests(unittest.TestCase):
    def test_weak_partial_requests_one_full_search(self):
        self.assertTrue(require_full_next_search(search_only=True, backtracks=0,
            site_increment_mm=.0009, objective_decrease_N_mm=.007))

    def test_successful_full_move_consumes_request(self):
        self.assertFalse(require_full_next_search(search_only=False, backtracks=5,
            site_increment_mm=None, objective_decrease_N_mm=1e-10))

    def test_meaningful_partial_stays_search_only(self):
        self.assertFalse(require_full_next_search(search_only=True, backtracks=0,
            site_increment_mm=.02, objective_decrease_N_mm=.01))


if __name__ == '__main__':
    unittest.main()
