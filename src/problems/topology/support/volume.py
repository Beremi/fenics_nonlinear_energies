"""Unambiguous material-measure and normalized-volume semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VolumeTarget:
    """A topology material target represented in both supported units."""

    normalized_fraction: float
    material_measure: float
    domain_area: float

    def __post_init__(self) -> None:
        values = (self.normalized_fraction, self.material_measure, self.domain_area)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Volume-target values must be finite.")
        if not self.domain_area > 0.0:
            raise ValueError("domain_area must be positive.")
        if not 0.0 < self.normalized_fraction < 1.0:
            raise ValueError("target_normalized_fraction must lie in (0, 1).")
        expected_measure = self.normalized_fraction * self.domain_area
        tolerance = 1e-12 * max(1.0, abs(expected_measure), abs(self.material_measure))
        if abs(self.material_measure - expected_measure) > tolerance:
            raise ValueError(
                "material_measure must equal normalized_fraction * domain_area."
            )

    @classmethod
    def from_normalized_fraction(
        cls,
        normalized_fraction: float,
        domain_area: float,
    ) -> "VolumeTarget":
        fraction = float(normalized_fraction)
        area = float(domain_area)
        return cls(fraction, fraction * area, area)

    @classmethod
    def from_material_measure(
        cls,
        material_measure: float,
        domain_area: float,
    ) -> "VolumeTarget":
        measure = float(material_measure)
        area = float(domain_area)
        if not area > 0.0:
            raise ValueError("domain_area must be positive.")
        return cls(measure / area, measure, area)


def resolve_volume_target(
    domain_area: float,
    *,
    target_normalized_fraction: float | None = None,
    target_material_measure: float | None = None,
    legacy_volume_fraction_target: float | None = None,
    default_normalized_fraction: float = 0.4,
) -> VolumeTarget:
    """Resolve one target and reject ambiguous or inconsistent specifications."""

    normalized_values = [
        float(value)
        for value in (target_normalized_fraction, legacy_volume_fraction_target)
        if value is not None
    ]
    if len(normalized_values) == 2 and not math.isclose(
        normalized_values[0], normalized_values[1], rel_tol=0.0, abs_tol=1e-14
    ):
        raise ValueError(
            "target_normalized_fraction and legacy volume_fraction_target disagree."
        )
    normalized = normalized_values[0] if normalized_values else None
    if normalized is not None and target_material_measure is not None:
        implied = float(target_material_measure) / float(domain_area)
        if not math.isclose(normalized, implied, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError(
                "Specify either a normalized fraction or a material measure, not "
                "inconsistent values of both."
            )
    if target_material_measure is not None:
        return VolumeTarget.from_material_measure(target_material_measure, domain_area)
    if normalized is None:
        normalized = float(default_normalized_fraction)
    return VolumeTarget.from_normalized_fraction(normalized, domain_area)
