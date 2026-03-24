"""Reference-solve policy for thesis error columns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferencePolicy:
    """How to build the proxy reference for one thesis row."""

    compare_mode: str
    method: str
    direction: str
    level: int
    init_mode: str
    epsilon: float
    maxit: int
    note: str


def get_reference_policy(case) -> ReferencePolicy | None:
    """Return the proxy-reference policy for one benchmark case."""
    table = str(case.table)
    method = str(case.method)
    direction = str(case.direction)
    level = int(case.level)
    init_mode = str(case.init_mode)
    epsilon = float(case.epsilon)

    if table in {"table_5_2", "table_5_3", "table_5_2_drn_sanity"}:
        return ReferencePolicy(
            compare_mode="same_mesh",
            method=method,
            direction=direction,
            level=level,
            init_mode=init_mode,
            epsilon=min(1.0e-8, epsilon * 1.0e-3),
            maxit=1000,
            note="same-mesh tight proxy reference for the 1D harness",
        )

    if table in {"table_5_6", "table_5_7", "table_5_8", "table_5_9", "table_5_10", "table_5_11"}:
        return ReferencePolicy(
            compare_mode="same_mesh",
            method="rmpa",
            direction="d_vh",
            level=level,
            init_mode="sine",
            epsilon=min(1.0e-8, epsilon * 1.0e-3),
            maxit=1000,
            note="same-mesh principal-branch proxy reference via tight RMPA solve",
        )

    if table == "table_5_14":
        if method == "oa2" and init_mode != "sine":
            return ReferencePolicy(
                compare_mode="same_mesh",
                method="oa2",
                direction=direction,
                level=level,
                init_mode=init_mode,
                epsilon=min(1.0e-8, epsilon * 1.0e-3),
                maxit=1000,
                note="same-mesh same-seed OA2 proxy reference for multibranch square cases",
            )
        return ReferencePolicy(
            compare_mode="same_mesh",
            method="rmpa",
            direction="d_vh",
            level=level,
            init_mode="sine",
            epsilon=min(1.0e-8, epsilon * 1.0e-3),
            maxit=1000,
            note="same-mesh principal-branch proxy reference for OA1 and OA2 sine rows",
        )

    return None
