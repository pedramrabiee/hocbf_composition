import torch
import numpy as np
from torch.autograd import grad
import math
from functools import partial
from torchdiffeq import odeint


def make_circle_barrier_functional(center, radius):
    center = vectorize_tensors(center).to(torch.float64)
    return lambda x: torch.norm((vectorize_tensors(x)[..., :2] - center.to(x.device)), p=2,
                                dim=-1) / radius - 1


def make_norm_rectangular_barrier_functional(center, size, rotation=0.0, p=20):
    size, center = vectorize_tensors(size).to(torch.float64), vectorize_tensors(center).to(torch.float64)
    return lambda x: torch.norm(
        (rotate_tensors(points=vectorize_tensors(x), center=center.to(x.device),
                        angle_rad=-rotation) - center.to(x.device)) / size.to(x.device),
        p=p,
        dim=-1) - 1


def make_affine_rectangular_barrier_functional(center, size, rotation=0.0, smooth=False, softmin_rho=40):
    size, center = vectorize_tensors(size).to(torch.float64), vectorize_tensors(center).to(torch.float64)

    # Define the normals for the axis-aligned rectangle (in local coordinate system)
    A = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float64)
    # Rotate the normals
    b = torch.tensor([
        center[0] + size[0],
        -center[0] + size[0],
        center[1] + size[1],
        -center[1] + size[1]
    ], dtype=torch.float64).unsqueeze(0)

    def affine_rectangle(x):
        rotate_x = rotate_tensors(points=vectorize_tensors(x[..., :2]), center=center.to(x.device), angle_rad=-rotation)
        ans = torch.einsum('mn,bn->bm', A, rotate_x) - b
        if smooth:
            return softmax(x=ans, rho=softmin_rho, dim=-1)
        return torch.max(ans , dim=-1).values


    # Return the affine constraint function
    return affine_rectangle

def make_norm_rectangular_boundary_functional(center, size, rotation=0.0, p=20):
    return lambda x: -make_norm_rectangular_barrier_functional(center, rotation, size, p)(x)

def make_affine_rectangular_boundary_functional(center, size, rotation=0.0, smooth=False, softmin_rho=40):
    return lambda x: -make_affine_rectangular_barrier_functional(center, rotation, size, smooth, softmin_rho)(x)


def make_box_barrier_functionals(bounds, idx):
    lb, ub = bounds
    # TODO: test dimensions
    return [lambda x: vectorize_tensors(x)[..., idx] - lb, lambda x: ub - vectorize_tensors(x)[..., idx]]


def make_linear_alpha_function_form_list_of_coef(coef_list):
    return [(lambda x, c=c: c * x) for c in coef_list]


def make_cubic_alpha_function_form_list_of_coef(coef_list):
    return [(lambda x, c=c: c * x ** 3) for c in coef_list]


def make_tanh_alpha_function_form_list_of_coef(coef_list):
    return [(lambda x, c=c: c * torch.tanh(x)) for c in coef_list]


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
    func_deriv = get_func_deriv(x, func)
    field_val = field(x)
    return lie_deriv_from_values(func_deriv, field_val)


def get_func_deriv(x, func):
    x = vectorize_tensors(x)
    grad_req = x.requires_grad
    x.requires_grad_()
    func_val = func(x).sum(0)
    func_deriv = get_func_deriv_from_func_vals(x, func_val)
    x.requires_grad_(requires_grad=grad_req)
    return func_deriv


def get_func_deriv_from_func_vals(x, func_val):
    func_deriv = [grad(fval, x, retain_graph=True)[0] for fval in func_val]
    return func_deriv


def lie_deriv_from_values(func_deriv, field_val):
    assert field_val.ndim in {2, 3}, 'Field dimension is not accepted'

    if field_val.ndim == 2:
        return torch.cat([torch.einsum('bn,bn->b', fderiv, field_val).unsqueeze(1) for fderiv in func_deriv], dim=1)

    if field_val.ndim == 3:
        return torch.einsum('rbn,bnm->brm', torch.stack(func_deriv), field_val).squeeze(1)


def make_higher_order_lie_deriv_series(func, field, deg):
    """
          Generate a series of higher-order lie derivative.

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
                                    [math.sin(angle_rad), math.cos(angle_rad)]], dtype=torch.float64).to(points.device)
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


def get_trajs_from_action_func(x0, dynamics, action_func, timestep, sim_time, method='euler'):
    return odeint(func=lambda t, y: dynamics.rhs(y, action_func(y)),
                  y0=x0,
                  t=torch.linspace(0.0, sim_time, int(sim_time / timestep) + 1),
                  method=method).detach()


def get_trajs_from_action_func_zoh(x0, dynamics, action_func, timestep, sim_time, method='euler'):
    # dynamics = dynamics.reset_zoh_time()
    return odeint(func=lambda t, y: partial(dynamics.rhs_zoh, action_func=action_func, timestep=timestep)(t, y),
                  y0=x0,
                  t=torch.linspace(0.0, sim_time, int(sim_time / timestep) + 1),
                  method=method).detach()


def update_dict_no_overwrite(original_dict, new_dict):
    for key, value in new_dict.items():
        if key not in original_dict:
            original_dict[key] = value