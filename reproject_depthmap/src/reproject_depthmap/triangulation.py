import torch


def triangulate_grid(h, w, mask=None):
    r"""
    Parameters
    ----------
    h : int
    w : int
    mask : torch.BoolTensor
        of shape [height, width].

    Returns
    -------
    triangles : torch.Tensor
        of shape [triangles_n, 3], oriented "to the camera"
    """
    if mask is None:
        ids = torch.arange(h * w).view(h, w)
    else:
        ids = torch.full([h, w], -1, dtype=torch.long)
        ids.masked_scatter_(mask, torch.arange(mask.sum()))
    tris = torch.stack([
        ids[:-1, :-1], ids[1:, :-1], ids[:-1, 1:],
        ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:],
    ], -1).view(-1, 3)
    del ids

    if mask is not None:
        tri_is_in_mask = (tris != -1).all(1)
        tris = tris[tri_is_in_mask]
    return tris
