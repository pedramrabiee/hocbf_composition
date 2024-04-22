import torch
import numpy as np
from torch.autograd import grad
import math


def make_circle_barrier_functional(center, radius):
    center = vectorize_tensors(center).to(torch.float64)
    return lambda x: torch.norm((vectorize_tensors(x)[..., :2] - center), p=2, dim=-1) / radius - 1


def make_rectangular_barrier_functional(center, rotation, size, p=20):
    size, center = vectorize_tensors(size).to(torch.float64), vectorize_tensors(center).to(torch.float64)
    return lambda x: torch.norm(
        (rotate_tensors(points=vectorize_tensors(x), center=center, angle_rad=-rotation) - center) / size, p=p,
        dim=-1) - 1


def make_rectangular_boundary_functional(center, rotation, size, p=20):
    return lambda x: -make_rectangular_barrier_functional(center, rotation, size, p)(x)


def make_box_barrier_functionals(bounds, idx):
    lb, ub = bounds
    # TODO: test dimensions
    return [lambda x: vectorize_tensors(x)[..., idx] - lb, lambda x: ub - vectorize_tensors(x)[..., idx]]


def make_linear_alpha_function_form_list_of_coef(coef_list):
    return [(lambda x, c=c: c * x) for c in coef_list]


def vectorize_tensors(arr):
    if isinstance(arr, torch.Tensor):
        return arr.unsqueeze_(0) if arr.ndim == 1 else arr
    return tensify(arr)


def tensify(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, (list, tuple)):
        return torch.tensor(obj)
    raise ValueError("Unsupported object type for conversion to tensor")


def apply_and_batchize(func, x):
    res = func(vectorize_tensors(x))
    if x.ndim == 2 and res.ndim == 1:
        return res.view(-1, 1)
    return res


def softmin(x, rho, conservative=False, dim=0):
    return softmax(x=x, rho=-rho, conservative=conservative, dim=dim)


def softmax(x, rho, conservative=True, dim=0):
    res = 1 / rho * torch.logsumexp(rho * x, dim=dim)
    return res - np.log(x.size(dim)) / rho if conservative else res


def lie_deriv(x, func, field):
    x = vectorize_tensors(x)
    grad_req = x.requires_grad
    x.requires_grad_()
    func_val = func(x).sum(0)
    func_deriv = [grad(fval, x, retain_graph=True)[0] for fval in func_val]
    x.requires_grad_(requires_grad=grad_req)
    field_val = field(x)

    assert field_val.ndim in {2, 3}, 'Field dimension is not accepted'

    if field_val.ndim == 2:
        return torch.cat([torch.einsum('bn,bn->b', fderiv, field_val).unsqueeze(1) for fderiv in func_deriv], dim=1)

    if field_val.ndim == 3:
        return torch.einsum('rbn,bnm->brm', torch.stack(func_deriv), field_val).squeeze(1)


def make_higher_order_lie_deriv_series(func, field, deg):
    """
          Generate a series of higher-order lie derivative..

          Parameters:

      """
    ans = [func]
    for i in range(deg):
        holie_i = lambda x, holie=ans[i], f=field: lie_deriv(x, holie, f)
        ans.append(holie_i)
    return ans


def rotate_tensors(points, center, angle_rad):
    # Perform rotation
    rotation_matrix = torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad)],
                                    [math.sin(angle_rad), math.cos(angle_rad)]], dtype=torch.float64)
    rotated_points = torch.matmul(points[..., :2] - center, rotation_matrix.t()) + center
    return rotated_points


def apply_and_match_dim(func, x):
    # Apply the function to the input tensor
    res = func(x)

    if x.ndim == 1 and res.ndim == 1:
        return res
    # Check if dimensions need to be matched
    if x.ndim == 1 and res.ndim == 2:
        return res.squeeze_(0)
    if x.ndim == 2 and res.ndim == 1:
        return res.view(-1, 1)

    return res
