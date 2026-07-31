"""A tiny check-and-print harness so the demo scripts double as acceptance tests."""

from __future__ import annotations


class Report:
    def __init__(self, title: str) -> None:
        self.failures: list[str] = []
        print(f"\n{title}\n{'=' * len(title)}")

    def section(self, title: str) -> None:
        print(f"\n{title}\n{'-' * len(title)}")

    def check(self, label: str, passed: bool, detail: str = "") -> bool:
        self.failures.append(label) if not passed else None
        marker = "PASS" if passed else "FAIL"
        suffix = f"   {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        return passed

    def measure(self, label: str, detail: str) -> None:
        """Report a number that is observed, not asserted (e.g. cleanup accuracy)."""
        print(f"  [meas] {label}   {detail}")

    def note(self, text: str) -> None:
        print(f"         {text}")

    def exit_code(self) -> int:
        if self.failures:
            print(f"\n{len(self.failures)} check(s) failed:")
            for failure in self.failures:
                print(f"  - {failure}")
            return 1
        print("\nAll checks passed.")
        return 0
