"""Tests for Phase 5: Z3 Formal Verification."""

from atlas_mm.verification.properties import OrderBookVerifier


class TestVerification:
    def setup_method(self):
        self.verifier = OrderBookVerifier()

    def test_no_crossed_book_holds(self):
        """Test 1: verify_no_crossed_book().holds == True."""
        result = self.verifier.verify_no_crossed_book()
        assert result.holds is True
        assert result.property_name == "no_crossed_book"

    def test_as_spread_positive_holds(self):
        """Test 2: verify_as_spread_positive().holds == True."""
        result = self.verifier.verify_as_spread_positive()
        assert result.holds is True
        assert result.property_name == "as_spread_positive"

    def test_as_inventory_mean_reversion_holds(self):
        """Test 3: verify_as_inventory_mean_reversion().holds == True."""
        result = self.verifier.verify_as_inventory_mean_reversion()
        assert result.holds is True
        assert result.property_name == "as_inventory_mean_reversion"

    def test_price_time_priority_holds(self):
        """Test 4: verify_price_time_priority().holds == True."""
        result = self.verifier.verify_price_time_priority()
        assert result.holds is True
        assert result.property_name == "price_time_priority"

    def test_verify_all(self):
        """Test 5: all results in verify_all() have holds == True."""
        results = self.verifier.verify_all()
        assert len(results) == 4
        for r in results:
            assert r.holds is True, f"Property {r.property_name} failed: {r.description}"

    def test_solver_times_reasonable(self):
        """Test 6: solver times should be reasonable (< 5 seconds each)."""
        results = self.verifier.verify_all()
        for r in results:
            assert r.solver_time_ms < 5000, f"{r.property_name} took {r.solver_time_ms}ms"
