"""Quadrature-independent fixed-state diagnostics for Plasticity3D."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from src.problems.slope_stability_3d.jax.jax_energy_3d import element_energy_3d
from src.problems.slope_stability_3d.support.materials import heterogenous_materials_qp
from src.problems.slope_stability_3d.support.mesh import (
    SlopeStability3DCaseData,
    TETRA_QUADRATURE_DUFFY_125POINT,
    _assemble_local_tet_ops,
    _definition_materials,
    normalize_tetra_quadrature_rule_id,
    tetra_quadrature_rule,
)
from src.problems.slope_stability_3d.support.reduction import davis_b_reduction_qp


jax.config.update("jax_enable_x64", True)


BRANCH_NAMES = ("elastic", "shear", "left_edge", "right_edge", "apex")
BRANCH_MARGIN_GATE = 1.0e-8
DETERMINISTIC_DIRECTION_SEED = 20260710


def prescribed_analytic_displacement(
    coords_ref: np.ndarray,
    *,
    amplitude: float,
) -> np.ndarray:
    """Return the route-study displacement field in canonical nodal order.

    The field is defined on coordinates normalized to the axis-aligned bounding
    box.  It is therefore independent of the polynomial degree, element
    numbering, MPI partition, and quadrature rule.  Constraint elimination is
    deliberately left to the caller because the same field is used both by
    distributed free-vector runners and by full-state export utilities.
    """

    coords = np.asarray(coords_ref, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError("coords_ref must be a nonempty (n, 3) array")
    amplitude = float(amplitude)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude must be finite and positive")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords_ref must contain only finite coordinates")

    lower = np.min(coords, axis=0)
    span = np.max(coords, axis=0) - lower
    if np.any(span <= np.finfo(np.float64).eps):
        raise ValueError("coords_ref must span all three coordinate axes")
    x, y, z = ((coords - lower) / span).T
    displacement = np.empty_like(coords)
    displacement[:, 0] = (
        amplitude * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    )
    displacement[:, 1] = (
        amplitude * np.cos(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    )
    displacement[:, 2] = (
        amplitude * np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
    )
    if not np.all(np.isfinite(displacement)):
        raise FloatingPointError("prescribed displacement contains nonfinite values")
    return displacement


def _element_value_residual_hvp(
    u_elem: jnp.ndarray,
    direction_elem: jnp.ndarray,
    dphix_e: jnp.ndarray,
    dphiy_e: jnp.ndarray,
    dphiz_e: jnp.ndarray,
    quad_weight_e: jnp.ndarray,
    c_bar_e: jnp.ndarray,
    sin_phi_e: jnp.ndarray,
    shear_e: jnp.ndarray,
    bulk_e: jnp.ndarray,
    lame_e: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Differentiate the production scalar element energy at one fixed state."""

    energy, residual = jax.value_and_grad(element_energy_3d, argnums=0)(
        u_elem,
        dphix_e,
        dphiy_e,
        dphiz_e,
        quad_weight_e,
        c_bar_e,
        sin_phi_e,
        shear_e,
        bulk_e,
        lame_e,
    )
    _, action = jax.jvp(
        lambda trial: jax.grad(element_energy_3d, argnums=0)(
            trial,
            dphix_e,
            dphiy_e,
            dphiz_e,
            quad_weight_e,
            c_bar_e,
            sin_phi_e,
            shear_e,
            bulk_e,
            lame_e,
        ),
        (u_elem,),
        (direction_elem,),
    )
    return energy, residual, action


_ELEMENT_VALUE_RESIDUAL_HVP_BATCH = jax.jit(
    jax.vmap(
        _element_value_residual_hvp,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
)


@dataclass(frozen=True)
class FixedStateQuadratureDiagnostics:
    """Scalar summary plus global vectors needed for cross-rule comparisons."""

    summary: dict[str, object]
    full_residual: np.ndarray
    hessian_action: np.ndarray
    deterministic_direction: np.ndarray
    freedofs: np.ndarray
    branch_labels: np.ndarray


def _case_field(
    case_data: SlopeStability3DCaseData | Mapping[str, object],
    key: str,
) -> object:
    if isinstance(case_data, Mapping):
        return case_data[key]
    return getattr(case_data, key)


def _optional_case_field(
    case_data: SlopeStability3DCaseData | Mapping[str, object],
    key: str,
    default: object,
) -> object:
    if isinstance(case_data, Mapping):
        return case_data.get(key, default)
    return getattr(case_data, key, default)


def _validate_freedofs(raw_freedofs: object, *, number_of_dofs: int) -> np.ndarray:
    freedofs = np.asarray(raw_freedofs, dtype=np.int64).reshape(-1)
    if freedofs.size == 0:
        raise ValueError("fixed-state diagnostics require at least one free degree of freedom")
    if np.any(freedofs < 0) or np.any(freedofs >= int(number_of_dofs)):
        raise ValueError("freedofs contains an out-of-range degree of freedom")
    if np.unique(freedofs).size != freedofs.size:
        raise ValueError("freedofs must not contain duplicates")
    return np.sort(freedofs)


def _deterministic_free_direction(
    number_of_dofs: int,
    freedofs: np.ndarray,
) -> np.ndarray:
    """Return a reproducible coefficient-space unit vector on the free DOFs."""

    rng = np.random.Generator(np.random.PCG64(DETERMINISTIC_DIRECTION_SEED))
    direction = np.zeros(int(number_of_dofs), dtype=np.float64)
    direction[freedofs] = rng.standard_normal(int(freedofs.size))
    norm = float(np.linalg.norm(direction[freedofs]))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("failed to construct a finite deterministic direction")
    direction[freedofs] /= norm
    return direction


def _strain6_numpy(
    u_elem: np.ndarray,
    dphix: np.ndarray,
    dphiy: np.ndarray,
    dphiz: np.ndarray,
) -> np.ndarray:
    """Rebuild strains in NumPy without calling the production JAX strain kernel."""

    ux = np.asarray(u_elem[:, 0::3], dtype=np.float64)
    uy = np.asarray(u_elem[:, 1::3], dtype=np.float64)
    uz = np.asarray(u_elem[:, 2::3], dtype=np.float64)
    eps = np.empty((u_elem.shape[0], dphix.shape[1], 6), dtype=np.float64)
    eps[:, :, 0] = np.einsum("ea,eqa->eq", ux, dphix, optimize=True)
    eps[:, :, 1] = np.einsum("ea,eqa->eq", uy, dphiy, optimize=True)
    eps[:, :, 2] = np.einsum("ea,eqa->eq", uz, dphiz, optimize=True)
    eps[:, :, 3] = np.einsum("ea,eqa->eq", ux, dphiy, optimize=True) + np.einsum(
        "ea,eqa->eq", uy, dphix, optimize=True
    )
    eps[:, :, 4] = np.einsum("ea,eqa->eq", uy, dphiz, optimize=True) + np.einsum(
        "ea,eqa->eq", uz, dphiy, optimize=True
    )
    eps[:, :, 5] = np.einsum("ea,eqa->eq", ux, dphiz, optimize=True) + np.einsum(
        "ea,eqa->eq", uz, dphix, optimize=True
    )
    return eps


def _branch_diagnostics_numpy(
    eps6: np.ndarray,
    c_bar: np.ndarray,
    sin_phi: np.ndarray,
    shear: np.ndarray,
    bulk: np.ndarray,
    lame: np.ndarray,
    quad_weight: np.ndarray,
    *,
    tiny: float = 1.0e-15,
    include_internal_arrays: bool = False,
) -> dict[str, object]:
    """Classify all quadrature points with a vectorized NumPy transcription.

    The geometry, strain evaluation, and implementation are independent of the
    production JAX kernel.  The branch predicates necessarily share the same
    Mohr--Coulomb algebra, so these diagnostics are not an independent
    constitutive model or a generalized-differentiability certificate.
    """

    eps = np.asarray(eps6, dtype=np.float64)
    shape = eps.shape[:2]
    tensor = np.empty(shape + (3, 3), dtype=np.float64)
    tensor[:, :, 0, 0] = eps[:, :, 0]
    tensor[:, :, 1, 1] = eps[:, :, 1]
    tensor[:, :, 2, 2] = eps[:, :, 2]
    tensor[:, :, 0, 1] = tensor[:, :, 1, 0] = 0.5 * eps[:, :, 3]
    tensor[:, :, 1, 2] = tensor[:, :, 2, 1] = 0.5 * eps[:, :, 4]
    tensor[:, :, 0, 2] = tensor[:, :, 2, 0] = 0.5 * eps[:, :, 5]

    raw_eigvals = np.linalg.eigvalsh(tensor)
    tie_break = float(tiny) * np.diag(np.array([0.0, 1.0, 2.0]))
    eigvals = np.linalg.eigvalsh(tensor + tie_break)
    eig_3 = eigvals[:, :, 0]
    eig_2 = eigvals[:, :, 1]
    eig_1 = eigvals[:, :, 2]
    invariant_1 = np.trace(tensor, axis1=2, axis2=3)

    c_bar = np.asarray(c_bar, dtype=np.float64)
    sin_phi = np.asarray(sin_phi, dtype=np.float64)
    shear = np.asarray(shear, dtype=np.float64)
    bulk = np.asarray(bulk, dtype=np.float64)
    lame = np.asarray(lame, dtype=np.float64)
    f_tr = (
        2.0 * shear * ((1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * eig_3)
        + 2.0 * lame * sin_phi * invariant_1
        - c_bar
    )
    gamma_sl = (eig_1 - eig_2) / np.maximum(tiny, 1.0 + sin_phi)
    gamma_sr = (eig_2 - eig_3) / np.maximum(tiny, 1.0 - sin_phi)
    gamma_la = (eig_1 + eig_2 - 2.0 * eig_3) / np.maximum(tiny, 3.0 - sin_phi)
    gamma_ra = (2.0 * eig_1 - eig_2 - eig_3) / np.maximum(tiny, 3.0 + sin_phi)

    denom_s = 4.0 * lame * sin_phi**2 + 4.0 * shear * (1.0 + sin_phi**2)
    denom_l = (
        4.0 * lame * sin_phi**2
        + shear * (1.0 + sin_phi) ** 2
        + 2.0 * shear * (1.0 - sin_phi) ** 2
    )
    denom_r = (
        4.0 * lame * sin_phi**2
        + 2.0 * shear * (1.0 + sin_phi) ** 2
        + shear * (1.0 - sin_phi) ** 2
    )
    denom_a = 4.0 * bulk * sin_phi**2

    def safe_signed(values: np.ndarray) -> np.ndarray:
        signs = np.where(values >= 0.0, 1.0, -1.0)
        return np.where(np.abs(values) < tiny, signs * tiny, values)

    lambda_s = f_tr / safe_signed(denom_s)
    lambda_l = (
        shear * ((1.0 + sin_phi) * (eig_1 + eig_2) - 2.0 * (1.0 - sin_phi) * eig_3)
        + 2.0 * lame * sin_phi * invariant_1
        - c_bar
    ) / safe_signed(denom_l)
    lambda_r = (
        shear * (2.0 * (1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * (eig_2 + eig_3))
        + 2.0 * lame * sin_phi * invariant_1
        - c_bar
    ) / safe_signed(denom_r)
    lambda_a = (2.0 * bulk * sin_phi * invariant_1 - c_bar) / safe_signed(denom_a)

    elastic = f_tr <= 0.0
    shear_branch = (~elastic) & (lambda_s <= np.minimum(gamma_sl, gamma_sr))
    left = (
        (~(elastic | shear_branch))
        & (gamma_sl < gamma_sr)
        & (lambda_l >= gamma_sl)
        & (lambda_l <= gamma_la)
    )
    right = (
        (~(elastic | shear_branch | left))
        & (gamma_sl > gamma_sr)
        & (lambda_r >= gamma_sr)
        & (lambda_r <= gamma_ra)
    )
    apex = ~(elastic | shear_branch | left | right)
    masks = (elastic, shear_branch, left, right, apex)

    stress_scale = np.maximum.reduce(
        (
            np.abs(c_bar),
            np.abs(f_tr),
            2.0 * np.abs(shear) * np.maximum(np.abs(eig_1), np.abs(eig_3)),
            2.0 * np.abs(lame * sin_phi * invariant_1),
            np.full(shape, np.finfo(float).tiny),
        )
    )
    modulus_scale = np.maximum.reduce(
        (
            np.abs(shear),
            np.abs(bulk),
            np.abs(lame),
            np.full(shape, np.finfo(float).tiny),
        )
    )
    strain_scale = np.maximum.reduce(
        tuple(
            np.abs(value)
            for value in (
                eig_1,
                eig_2,
                eig_3,
                lambda_s,
                lambda_l,
                lambda_r,
                lambda_a,
                gamma_sl,
                gamma_sr,
                gamma_la,
                gamma_ra,
                c_bar / modulus_scale,
            )
        )
        + (np.full(shape, np.finfo(float).tiny),)
    )
    normalized_yield = f_tr / stress_scale
    shear_exclusion = (lambda_s - np.minimum(gamma_sl, gamma_sr)) / strain_scale
    edge_order = (gamma_sl - gamma_sr) / strain_scale
    left_lower = (lambda_l - gamma_sl) / strain_scale
    left_upper = (gamma_la - lambda_l) / strain_scale
    right_lower = (lambda_r - gamma_sr) / strain_scale
    right_upper = (gamma_ra - lambda_r) / strain_scale
    left_apex = (lambda_l - gamma_la) / strain_scale
    right_apex = (lambda_r - gamma_ra) / strain_scale

    active_margin = np.empty(shape, dtype=np.float64)
    active_margin[elastic] = -normalized_yield[elastic]
    active_margin[shear_branch] = np.minimum(
        normalized_yield[shear_branch], -shear_exclusion[shear_branch]
    )
    active_margin[left] = np.minimum.reduce(
        (
            normalized_yield[left],
            shear_exclusion[left],
            -edge_order[left],
            left_lower[left],
            left_upper[left],
        )
    )
    active_margin[right] = np.minimum.reduce(
        (
            normalized_yield[right],
            shear_exclusion[right],
            edge_order[right],
            right_lower[right],
            right_upper[right],
        )
    )
    apex_left = apex & (edge_order < 0.0)
    apex_right = apex & (edge_order > 0.0)
    apex_tie = apex & ~(apex_left | apex_right)
    active_margin[apex_left] = np.minimum.reduce(
        (
            normalized_yield[apex_left],
            shear_exclusion[apex_left],
            -edge_order[apex_left],
            left_apex[apex_left],
        )
    )
    active_margin[apex_right] = np.minimum.reduce(
        (
            normalized_yield[apex_right],
            shear_exclusion[apex_right],
            edge_order[apex_right],
            right_apex[apex_right],
        )
    )
    active_margin[apex_tie] = 0.0

    raw_gaps = np.diff(raw_eigvals, axis=2)
    eigen_scale = np.maximum(np.max(np.abs(raw_eigvals), axis=2), tiny)
    normalized_gap = np.min(raw_gaps, axis=2) / eigen_scale
    normalized_denominator = np.minimum.reduce(
        (np.abs(denom_s), np.abs(denom_l), np.abs(denom_r), np.abs(denom_a))
    ) / modulus_scale
    weights = np.asarray(quad_weight, dtype=np.float64)
    counts = {name: int(np.count_nonzero(mask)) for name, mask in zip(BRANCH_NAMES, masks)}
    total_points = int(np.prod(shape))
    absolute_weight_total = float(np.sum(np.abs(weights)))
    absolute_weights = {
        name: float(np.sum(np.abs(weights[mask]))) for name, mask in zip(BRANCH_NAMES, masks)
    }
    result: dict[str, object] = {
        "branch_point_counts": counts,
        "branch_point_fractions": {
            name: float(counts[name] / total_points) for name in BRANCH_NAMES
        },
        "branch_absolute_quadrature_weight_fractions": {
            name: float(absolute_weights[name] / absolute_weight_total)
            for name in BRANCH_NAMES
        },
        "quadrature_points": total_points,
        "quadrature_points_at_or_below_margin_gate": int(
            np.count_nonzero(active_margin <= BRANCH_MARGIN_GATE)
        ),
        "minimum_normalized_active_branch_margin": float(np.min(active_margin)),
        "minimum_raw_principal_value_gap": float(np.min(raw_gaps)),
        "minimum_normalized_principal_value_gap": float(np.min(normalized_gap)),
        "minimum_normalized_constitutive_denominator": float(
            np.min(normalized_denominator)
        ),
        "signed_quadrature_weight": float(np.sum(weights)),
        "absolute_quadrature_weight": absolute_weight_total,
        "quadrature_weights_are_strictly_positive": bool(np.all(weights > 0.0)),
    }
    if include_internal_arrays:
        branch_labels = np.empty(shape, dtype=np.int8)
        for code, mask in enumerate(masks):
            branch_labels[mask] = code
        result["_branch_labels"] = branch_labels
        result["_active_margin"] = np.asarray(active_margin, dtype=np.float64)
    return result


def evaluate_fixed_state_quadrature_diagnostics(
    case_data: SlopeStability3DCaseData | Mapping[str, object],
    displacement: np.ndarray,
    *,
    lambda_target: float,
    quadrature_rule_id: str = TETRA_QUADRATURE_DUFFY_125POINT,
    element_chunk_size: int = 256,
) -> FixedStateQuadratureDiagnostics:
    """Re-evaluate energy and first/second derivatives with one named rule.

    Geometry gradients, material arrays, gravity work, and global scatter-adds
    are rebuilt from the mesh and named rule; solve-time operator arrays are not
    reused.  Each element chunk invokes automatic differentiation of the
    production scalar energy.  NumPy independently rebuilds strains and replays
    the branch predicates, but this replay shares the published constitutive
    algebra and is not an independent material model.
    """

    degree = int(_case_field(case_data, "degree"))
    rule_id = normalize_tetra_quadrature_rule_id(
        quadrature_rule_id,
        element_degree=degree,
    )
    chunk_size = int(element_chunk_size)
    if chunk_size <= 0:
        raise ValueError("element_chunk_size must be positive")

    u_full = np.asarray(displacement, dtype=np.float64).reshape(-1)
    nodes = np.asarray(_case_field(case_data, "nodes"), dtype=np.float64)
    expected_dofs = 3 * int(nodes.shape[0])
    if u_full.size != expected_dofs:
        raise ValueError(
            f"displacement has {u_full.size} entries; expected {expected_dofs}"
        )
    if not np.all(np.isfinite(u_full)):
        raise ValueError("displacement must contain only finite values")
    default_freedofs = np.arange(expected_dofs, dtype=np.int64)
    freedofs = _validate_freedofs(
        _optional_case_field(case_data, "freedofs", default_freedofs),
        number_of_dofs=expected_dofs,
    )
    direction = _deterministic_free_direction(expected_dofs, freedofs)

    elems_scalar = np.asarray(_case_field(case_data, "elems_scalar"), dtype=np.int64)
    elems = np.asarray(_case_field(case_data, "elems"), dtype=np.int64)
    material_id = np.asarray(_case_field(case_data, "material_id"), dtype=np.int64)
    n_elements = int(elems_scalar.shape[0])
    if n_elements <= 0:
        raise ValueError("fixed-state diagnostics require at least one element")
    materials = _definition_materials()
    xi, reference_weights = tetra_quadrature_rule(rule_id)
    n_q = int(xi.shape[1])

    internal_energy = 0.0
    external_work = 0.0
    full_internal_residual = np.zeros(expected_dofs, dtype=np.float64)
    full_force = np.zeros(expected_dofs, dtype=np.float64)
    full_hessian_action = np.zeros(expected_dofs, dtype=np.float64)
    branch_counts = {name: 0 for name in BRANCH_NAMES}
    branch_absolute_weights = {name: 0.0 for name in BRANCH_NAMES}
    total_branch_points = 0
    points_near_branch_boundary = 0
    minimum_active_margin = np.inf
    minimum_raw_gap = np.inf
    minimum_normalized_gap = np.inf
    minimum_denominator = np.inf
    signed_weight = 0.0
    absolute_weight = 0.0
    all_weights_positive = True
    branch_label_chunks: list[np.ndarray] = []

    for start in range(0, n_elements, chunk_size):
        stop = min(start + chunk_size, n_elements)
        elem_slice = slice(start, stop)
        scalar_chunk = elems_scalar[elem_slice]
        elem_dofs = elems[elem_slice]
        dphix, dphiy, dphiz, quad_weight, hatp = _assemble_local_tet_ops(
            nodes,
            scalar_chunk,
            degree=degree,
            quadrature_rule_id=rule_id,
        )
        c0_q, phi_q, psi_q, shear_q, bulk_q, lame_q, gamma_q = (
            heterogenous_materials_qp(
                material_id[elem_slice],
                n_q=n_q,
                materials=materials,
            )
        )
        c_bar_q, sin_phi_q = davis_b_reduction_qp(
            c0_q,
            phi_q,
            psi_q,
            float(lambda_target),
        )
        u_elem = u_full[elem_dofs]
        direction_elem = direction[elem_dofs]
        values, residuals, actions = _ELEMENT_VALUE_RESIDUAL_HVP_BATCH(
            jnp.asarray(u_elem, dtype=jnp.float64),
            jnp.asarray(direction_elem, dtype=jnp.float64),
            jnp.asarray(dphix, dtype=jnp.float64),
            jnp.asarray(dphiy, dtype=jnp.float64),
            jnp.asarray(dphiz, dtype=jnp.float64),
            jnp.asarray(quad_weight, dtype=jnp.float64),
            jnp.asarray(c_bar_q, dtype=jnp.float64),
            jnp.asarray(sin_phi_q, dtype=jnp.float64),
            jnp.asarray(shear_q, dtype=jnp.float64),
            jnp.asarray(bulk_q, dtype=jnp.float64),
            jnp.asarray(lame_q, dtype=jnp.float64),
        )
        values_np = np.asarray(values.block_until_ready(), dtype=np.float64)
        residuals_np = np.asarray(residuals.block_until_ready(), dtype=np.float64)
        actions_np = np.asarray(actions.block_until_ready(), dtype=np.float64)
        if not (
            np.all(np.isfinite(values_np))
            and np.all(np.isfinite(residuals_np))
            and np.all(np.isfinite(actions_np))
        ):
            raise FloatingPointError(
                f"nonfinite fixed-state derivative diagnostic in elements {start}:{stop}"
            )
        internal_energy += float(np.sum(values_np))
        np.add.at(full_internal_residual, elem_dofs.ravel(), residuals_np.ravel())
        np.add.at(full_hessian_action, elem_dofs.ravel(), actions_np.ravel())

        local_y = -np.einsum(
            "eq,aq,eq->ea",
            quad_weight,
            hatp,
            gamma_q,
            optimize=True,
        )
        dofs_y = 3 * scalar_chunk + 1
        np.add.at(full_force, dofs_y.ravel(), local_y.ravel())
        external_work += float(np.sum(local_y * u_full[dofs_y]))

        strains = _strain6_numpy(u_elem, dphix, dphiy, dphiz)
        branch = _branch_diagnostics_numpy(
            strains,
            c_bar_q,
            sin_phi_q,
            shear_q,
            bulk_q,
            lame_q,
            quad_weight,
            include_internal_arrays=True,
        )
        branch_label_chunks.append(
            np.asarray(branch.pop("_branch_labels"), dtype=np.int8).reshape(-1)
        )
        branch.pop("_active_margin")
        for name in BRANCH_NAMES:
            branch_counts[name] += int(branch["branch_point_counts"][name])
            branch_absolute_weights[name] += float(
                branch["branch_absolute_quadrature_weight_fractions"][name]
            ) * float(branch["absolute_quadrature_weight"])
        total_branch_points += int(branch["quadrature_points"])
        points_near_branch_boundary += int(
            branch["quadrature_points_at_or_below_margin_gate"]
        )
        minimum_active_margin = min(
            minimum_active_margin,
            float(branch["minimum_normalized_active_branch_margin"]),
        )
        minimum_raw_gap = min(
            minimum_raw_gap,
            float(branch["minimum_raw_principal_value_gap"]),
        )
        minimum_normalized_gap = min(
            minimum_normalized_gap,
            float(branch["minimum_normalized_principal_value_gap"]),
        )
        minimum_denominator = min(
            minimum_denominator,
            float(branch["minimum_normalized_constitutive_denominator"]),
        )
        signed_weight += float(branch["signed_quadrature_weight"])
        absolute_weight += float(branch["absolute_quadrature_weight"])
        all_weights_positive = bool(
            all_weights_positive and branch["quadrature_weights_are_strictly_positive"]
        )

    full_residual = full_internal_residual - full_force
    free_residual = full_residual[freedofs]
    free_action = full_hessian_action[freedofs]
    if not (
        np.all(np.isfinite(full_residual))
        and np.all(np.isfinite(full_hessian_action))
    ):
        raise FloatingPointError("assembled fixed-state vectors contain nonfinite entries")

    displacement_nodes = u_full.reshape((-1, 3))
    branch_labels = np.concatenate(branch_label_chunks)
    if branch_labels.size != total_branch_points:
        raise RuntimeError("assembled branch-label map has the wrong size")
    summary: dict[str, object] = {
        "quadrature_rule_id": str(rule_id),
        "quadrature_points": int(n_q),
        "quadrature_points_per_element": int(n_q),
        "element_degree": int(degree),
        "elements": int(n_elements),
        "degrees_of_freedom": int(expected_dofs),
        "free_degrees_of_freedom": int(freedofs.size),
        "lambda_target": float(lambda_target),
        "internal_energy": float(internal_energy),
        "external_work": float(external_work),
        "total_potential_energy": float(internal_energy - external_work),
        "u_max": float(np.max(np.linalg.norm(displacement_nodes, axis=1))),
        "full_residual_l2_norm": float(np.linalg.norm(full_residual)),
        "full_residual_linf_norm": float(np.linalg.norm(full_residual, ord=np.inf)),
        "free_residual_l2_norm": float(np.linalg.norm(free_residual)),
        "free_residual_linf_norm": float(np.linalg.norm(free_residual, ord=np.inf)),
        "full_hessian_action_l2_norm": float(np.linalg.norm(full_hessian_action)),
        "full_hessian_action_linf_norm": float(
            np.linalg.norm(full_hessian_action, ord=np.inf)
        ),
        "free_hessian_action_l2_norm": float(np.linalg.norm(free_action)),
        "free_hessian_action_linf_norm": float(np.linalg.norm(free_action, ord=np.inf)),
        "deterministic_direction": {
            "generator": "numpy.random.PCG64 standard_normal",
            "seed": int(DETERMINISTIC_DIRECTION_SEED),
            "support": "free degrees of freedom; constrained entries are zero",
            "normalization": "unit coefficient-space l2 norm on free degrees of freedom",
            "free_l2_norm": float(np.linalg.norm(direction[freedofs])),
        },
        "branch_point_counts": dict(branch_counts),
        "branch_point_fractions": {
            name: float(branch_counts[name] / total_branch_points) for name in BRANCH_NAMES
        },
        "branch_absolute_quadrature_weight_fractions": {
            name: float(branch_absolute_weights[name] / absolute_weight)
            for name in BRANCH_NAMES
        },
        "branch_sample_points": int(total_branch_points),
        "branch_margin_gate": float(BRANCH_MARGIN_GATE),
        "quadrature_points_at_or_below_margin_gate": int(
            points_near_branch_boundary
        ),
        "quadrature_point_fraction_at_or_below_margin_gate": float(
            points_near_branch_boundary / total_branch_points
        ),
        "minimum_normalized_active_branch_margin": float(minimum_active_margin),
        "minimum_raw_principal_value_gap": float(minimum_raw_gap),
        "minimum_normalized_principal_value_gap": float(minimum_normalized_gap),
        "minimum_normalized_constitutive_denominator": float(minimum_denominator),
        "signed_quadrature_weight": float(signed_weight),
        "absolute_quadrature_weight": float(absolute_weight),
        "quadrature_weights_are_strictly_positive": bool(all_weights_positive),
        "reference_rule_weights_sum": float(np.sum(reference_weights)),
        "diagnostic_scope": {
            "geometry_material_load": (
                "rebuilt from mesh coordinates, connectivity, material identifiers, "
                "Davis-B reduction, and the named rule; solve-time operator arrays are unused"
            ),
            "energy_residual_hessian_action": (
                "production scalar element energy differentiated by JAX independently "
                "for each streamed element chunk"
            ),
            "global_assembly": "independent NumPy scatter-add of element vectors and gravity load",
            "branch_classification": (
                "independent vectorized NumPy implementation sharing the production "
                "Mohr-Coulomb branch-predicate algebra; not an independent constitutive model"
            ),
            "interpretation": (
                "fixed-state quadrature sensitivity only; branch fractions sampled by "
                "different rules are not a mesh- or solved-state convergence result"
            ),
        },
    }
    return FixedStateQuadratureDiagnostics(
        summary=summary,
        full_residual=np.asarray(full_residual, dtype=np.float64),
        hessian_action=np.asarray(full_hessian_action, dtype=np.float64),
        deterministic_direction=np.asarray(direction, dtype=np.float64),
        freedofs=np.asarray(freedofs, dtype=np.int64),
        branch_labels=np.asarray(branch_labels, dtype=np.int8),
    )


def evaluate_fixed_state_with_quadrature(
    case_data: SlopeStability3DCaseData | Mapping[str, object],
    displacement: np.ndarray,
    *,
    lambda_target: float,
    quadrature_rule_id: str = TETRA_QUADRATURE_DUFFY_125POINT,
    element_chunk_size: int = 256,
) -> dict[str, object]:
    """Return the JSON-ready summary for one fixed-state rule evaluation."""

    return evaluate_fixed_state_quadrature_diagnostics(
        case_data,
        displacement,
        lambda_target=float(lambda_target),
        quadrature_rule_id=quadrature_rule_id,
        element_chunk_size=int(element_chunk_size),
    ).summary


def evaluate_fixed_state_overintegrated(
    case_data: SlopeStability3DCaseData | Mapping[str, object],
    displacement: np.ndarray,
    *,
    lambda_target: float,
    element_chunk_size: int = 256,
) -> dict[str, object]:
    """Evaluate a fixed state with the positive 125-point reference rule."""

    return evaluate_fixed_state_with_quadrature(
        case_data,
        displacement,
        lambda_target=float(lambda_target),
        quadrature_rule_id=TETRA_QUADRATURE_DUFFY_125POINT,
        element_chunk_size=int(element_chunk_size),
    )
