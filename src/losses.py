"""
Loss functions for semi-supervised DeepSDF training.

L_total = L_sdf + lambda_eik(t) * L_eik + lambda_z * L_z [+ lambda_2nd * L_2nd]

- L_sdf: L1 loss on supervised points only
- L_eik: Eikonal regularization on ALL points (supervised + unsupervised)
- L_z:   Latent code regularization
- L_2nd: Optional second-order divergence regularization
"""

import torch


def sdf_loss(pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> torch.Tensor:
    """L1 loss between predicted and ground-truth SDF values.

    Applied only to supervised points.

    Args:
        pred_sdf: (N, 1) predicted SDF
        gt_sdf: (N,) or (N, 1) ground truth SDF

    Returns:
        Scalar L1 loss
    """
    gt_sdf = gt_sdf.view_as(pred_sdf)
    return torch.mean(torch.abs(pred_sdf - gt_sdf))


def eikonal_loss(gradients: torch.Tensor) -> torch.Tensor:
    """Eikonal regularization: enforce ||grad_x f|| = 1.

    Applied to ALL points (supervised + unsupervised).

    Args:
        gradients: (N, 3) spatial gradients of SDF w.r.t. input coordinates

    Returns:
        Scalar mean((||grad|| - 1)^2)
    """
    grad_norm = torch.norm(gradients, dim=-1)  # (N,)
    return torch.mean((grad_norm - 1.0) ** 2)


def latent_reg_loss(latent_codes: torch.Tensor) -> torch.Tensor:
    """Latent code regularization: mean(||z||^2).

    Encourages latent codes to stay near the origin.

    Args:
        latent_codes: (N, latent_dim) latent vectors used in this batch

    Returns:
        Scalar mean squared norm
    """
    return torch.mean(latent_codes ** 2)


def second_order_loss(
    gradients: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Second-order regularization: |div(grad f)|.

    Penalizes the divergence of the gradient field, encouraging
    smoother SDF surfaces. Requires create_graph=True in the
    first gradient computation.

    Args:
        gradients: (N, 3) spatial gradients (must have grad_fn)
        coords: (N, 3) input coordinates (must have requires_grad=True)

    Returns:
        Scalar mean |divergence|
    """
    divergence = 0.0
    for i in range(3):
        d2f = torch.autograd.grad(
            outputs=gradients[:, i],
            inputs=coords,
            grad_outputs=torch.ones_like(gradients[:, i]),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if d2f is None:
            continue
        divergence = divergence + d2f[:, i]

    return torch.mean(torch.abs(divergence))


def compute_spatial_gradients(
    sdf_pred: torch.Tensor,
    coords: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Compute spatial gradients of SDF predictions w.r.t. input coordinates.

    Args:
        sdf_pred: (N, 1) predicted SDF values
        coords: (N, 3) input coordinates with requires_grad=True
        create_graph: whether to create graph for higher-order gradients

    Returns:
        gradients: (N, 3) spatial gradients
    """
    gradients = torch.autograd.grad(
        outputs=sdf_pred,
        inputs=coords,
        grad_outputs=torch.ones_like(sdf_pred),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return gradients


def get_warmup_epochs(supervision_ratio: float) -> int:
    """Get Eikonal warmup epochs based on supervision ratio.

    Lower supervision ratios need longer warmup to prevent L_eik
    from dominating before the network learns basic geometry from L_sdf.

    Args:
        supervision_ratio: fraction of supervised points (1.0, 0.5, 0.1, 0.05)

    Returns:
        Number of warmup epochs
    """
    if supervision_ratio >= 0.5:
        return 100
    elif supervision_ratio >= 0.1:
        return 150
    else:
        return 200


def compute_eikonal_weight(
    epoch: int,
    warmup_epochs: int,
    lambda_eik: float,
) -> float:
    """Compute current Eikonal loss weight with linear warmup.

    lambda_eik(t) = lambda_eik * min(1.0, t / warmup_epochs)

    Args:
        epoch: current training epoch (1-indexed)
        warmup_epochs: number of warmup epochs
        lambda_eik: target Eikonal weight

    Returns:
        Current effective Eikonal weight
    """
    if warmup_epochs <= 0:
        return lambda_eik
    ramp = min(1.0, epoch / warmup_epochs)
    return lambda_eik * ramp
