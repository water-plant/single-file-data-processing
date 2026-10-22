from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    """Structured result returned by evaluator utility checks."""

    passed: bool
    check: str
    reason: str = ""
    scope: str = "global"
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        check: str,
        *,
        scope: str = "global",
        column: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> CheckResult:
        return cls(
            passed=True,
            check=check,
            scope=scope,
            column=column,
            details=details or {},
        )

    @classmethod
    def fail(
        cls,
        check: str,
        reason: str,
        *,
        scope: str = "global",
        column: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> CheckResult:
        return cls(
            passed=False,
            check=check,
            reason=reason,
            scope=scope,
            column=column,
            details=details or {},
        )

    def __bool__(self) -> bool:
        return self.passed

    def __iter__(self):
        """Allow legacy unpacking with `passed, reason = check(...)`."""
        yield self.passed
        yield self.reason

    def to_dict(self) -> dict[str, Any]:
        result = {
            "passed": self.passed,
            "check": self.check,
            "reason": self.reason,
            "scope": self.scope,
            "details": self.details,
        }

        if self.column is not None:
            result["column"] = self.column

        return result


@dataclass
class CheckResults:
    """Small collection helper for aggregating evaluator check results."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [result for result in self.results if not result.passed]

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    def extend(self, results: list[CheckResult]) -> None:
        self.results.extend(results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
            "failures": [result.to_dict() for result in self.failures],
        }
