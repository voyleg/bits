import torch

from .camera_model import CameraModel
from .utils import ignore_warnings


class PinholeCameraModel(CameraModel):
    r"""Represents a pinhole camera model.

    Parameters
    ----------
    focal : array-like
        (fx, fy)
    principal : array-like
        (cx, cy)
    size_wh : array-like
        (w, h)
    """
    @ignore_warnings(['To copy construct from a tensor, it is recommended to use'])
    def __init__(self, focal, principal, size_wh):
        super().__init__(size_wh)
        self.focal = torch.nn.Parameter(torch.tensor(focal), requires_grad=False)
        self.principal = torch.nn.Parameter(torch.tensor(principal), requires_grad=False)

    def __repr__(self):
        return (f'PinholeCameraModel(size_wh={tuple(self.size_wh.tolist())}, '
                f'focal={tuple(self.focal.half().tolist())}, '
                f'principal={tuple(self.principal.half().tolist())})')

    def unproject(self, uv):
        r"""For points in the image space calculates the respective 3D directions in the camera space.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [2, n], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).

        Returns
        -------
        direction : torch.Tensor
            of shape [3, n], directions in camera space, X to the right, Y down, Z from the camera.
        """
        n = uv.shape[1]
        direction = (uv - self.principal.unsqueeze(1)).div_(self.focal.unsqueeze(1))
        direction = torch.cat([direction, direction.new_ones(1, n)], 0)
        direction = torch.nn.functional.normalize(direction, dim=0)
        return direction

    # Dtype, device
    # -------------
    @property
    def dtype(self):
        return self.focal.dtype
