"""Run formal verification checks and print results."""

from atlas_mm.verification.properties import OrderBookVerifier


def main():
    verifier = OrderBookVerifier()
    results = verifier.verify_all()

    print("\n" + "=" * 70)
    print("FORMAL VERIFICATION RESULTS")
    print("=" * 70)

    all_pass = True
    for r in results:
        status = "PROVED" if r.holds else "FAILED"
        icon = "[PASS]" if r.holds else "[FAIL]"
        print(f"\n{icon} [{status}] {r.property_name} ({r.solver_time_ms:.1f}ms)")
        print(f"  {r.description}")
        if r.counterexample:
            print(f"  Counterexample: {r.counterexample}")
        if not r.holds:
            all_pass = False

    print("\n" + "=" * 70)
    print(f"{'ALL PROPERTIES VERIFIED' if all_pass else 'SOME PROPERTIES FAILED'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
