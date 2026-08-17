"""一時検証用モジュール (F-02-2-002 の有効性実測。検証後に削除する)。"""

from __future__ import annotations

from dataclasses import dataclass

from rc_basics_lab.diagnostics.base import (
    DiagnosticContext,
    DiagnosticResult,
    resolve_context,
    validate_diagnostic_input,
)
from rc_basics_lab.types import FloatArray


@dataclass(frozen=True, slots=True)
class TmpParamDiagnostic:
    """第2形 (パラメータ化した callable) の一時検証用被検体。"""

    n_surrogates: int = 10

    def __call__(
        self,
        X: FloatArray,
        u: FloatArray | None = None,
        y: FloatArray | None = None,
        *,
        ctx: DiagnosticContext | None = None,
    ) -> DiagnosticResult:
        validate_diagnostic_input(X, u, y, ctx)
        context = resolve_context(ctx)
        return DiagnosticResult(
            name="tmp_param_diagnostic",
            scalars={"n_surrogates": float(self.n_surrogates)},
            params={"washout": str(context.washout)},
        )


tmp_param_diagnostic = TmpParamDiagnostic()
