import torch

def gauge_transform(data, in_fields):
    """Gauge transformation which reflects *all* reference frames

    Args:
        data (torch.Tensor): Feature field coefficient tensor.
        in_fields (Tuple[int, int, int]): Multiplicities of the input feature fields, transforming according to
            trivial, signflip and regular representations, respectively.

    Returns:
        torch.Tensor: Gauge transformed feature field coefficient tensor.

    """
    bound_1 = in_fields[0]
    bound_2 = in_fields[0] + in_fields[1]
    data_flip = torch.zeros_like(data)
    if in_fields[0]>0:
        data_flip[:,:bound_1] = data[:,:bound_1]
    if in_fields[1]>0:
        data_flip[:,bound_1:bound_2] = -data[:,bound_1:bound_2]
    if in_fields[2]>0:
        data_flip[:,bound_2::2] = data[:,bound_2+1::2]
        data_flip[:,bound_2+1::2] = data[:,bound_2::2]
    return data_flip


def isom_action(data, shift, in_fields):
    """Isometry action on feature fields on the Mobius strip.

    Shifts feature fields along the base space S^1 of the Mobius strip.
    Due to its twist, the image re-enters the strip upside down (see Fig. 28 in the paper).

    Args:
        data (torch.Tensor): Feature field coefficient tensor.
        shift (int): Number of pixels by which the input field is shifted.
        in_fields (Tuple[int, int, int]): Multiplicities of the input feature fields, transforming according to
            trivial, signflip and regular representations, respectively.

    Returns:
        torch.Tensor: Isometry transformed feature field coefficient tensor.

    """
    assert data.ndimension() == 4
    length = data.shape[-1]
    has_triv = in_fields[0]>0
    has_sign = in_fields[1]>0
    has_reg = in_fields[2]>0
    bound_1 = in_fields[0]
    bound_2 = in_fields[0] + in_fields[1]
    section = (shift//length)%2
    shift = shift%length
    shifted = torch.zeros_like(data)
    if section==0 and shift==0:
        shifted = data 
    elif section==1 and shift==0:
        if has_triv:
            shifted[:,:bound_1] = data[:,:bound_1].flip(2)
        if has_sign:
            shifted[:,bound_1:bound_2] = -data[:,bound_1:bound_2].flip(2)
        if has_reg:
            shifted[:,bound_2::2] = data[:,bound_2+1::2].flip(2)
            shifted[:,bound_2+1::2] = data[:,bound_2::2].flip(2)
    elif section==0:
        if has_triv:
            shifted[:, :bound_1, :, shift:] = data[:, :bound_1, :, :-shift]
            shifted[:, :bound_1, :, :shift] = data[:, :bound_1, :, -shift:].flip(2)
        if has_sign:
            shifted[:, bound_1:bound_2, :, shift:] = data[:, bound_1:bound_2, :, :-shift]
            shifted[:, bound_1:bound_2, :, :shift] = -data[:, bound_1:bound_2, :, -shift:].flip(2)
        if has_reg:
            shifted[:, bound_2::2, :, shift:] = data[:, bound_2::2, :, :-shift]
            shifted[:, bound_2+1::2, :, shift:] = data[:, bound_2+1::2, :, :-shift]
            shifted[:, bound_2::2, :, :shift] = data[:, bound_2+1::2, :, -shift:].flip(2)
            shifted[:, bound_2+1::2, :, :shift] = data[:, bound_2::2, :, -shift:].flip(2)
    else:
        if has_triv:
            shifted[:, :bound_1, :, shift:] = data[:, :bound_1, :, :-shift].flip(2)
            shifted[:, :bound_1, :, :shift] = data[:, :bound_1, :, -shift:]
        if has_sign:
            shifted[:, bound_1:bound_2, :, shift:] = -data[:, bound_1:bound_2, :, :-shift].flip(2)
            shifted[:, bound_1:bound_2, :, :shift] = data[:, bound_1:bound_2, :, -shift:]
        if has_reg:
            shifted[:, bound_2::2, :, shift:] = data[:, bound_2+1::2, :, :-shift].flip(2)
            shifted[:, bound_2+1::2, :, shift:] = data[:, bound_2::2, :, :-shift].flip(2)
            shifted[:, bound_2::2, :, :shift] = data[:, bound_2::2, :, -shift:]
            shifted[:, bound_2+1::2, :, :shift] = data[:, bound_2+1::2, :, -shift:]
    return shifted


def isom_action_numpy(data, shift, in_fields):
    """Isometry action wrapper for fields that are passed as numpy arrays"""
    data = torch.from_numpy(data)
    dim = data.dim()
    if dim == 3:
        data = data[None] # optionally add new batch axis, as assumed by isom_action
    data = isom_action(data, shift, in_fields)
    if dim == 3:
        data = data[0]
    return data.detach().numpy()
