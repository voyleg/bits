from abc import ABC, abstractmethod

import torch

from .utils import ignore_warnings


class CameraModel(ABC, torch.nn.Module):
    @ignore_warnings(['To copy construct from a tensor, it is recommended to use'])
    def __init__(self, size_wh):
        super().__init__()
        size_wh = torch.tensor(size_wh)
        if torch.is_floating_point(size_wh):
            raise ValueError(f'Expected integer size_wh, got {size_wh}')
        self.register_buffer('size_wh', size_wh)

    @property
    def device(self):
        return self.size_wh.device

    @property
    @abstractmethod
    def dtype(self): ...

    @abstractmethod
    def unproject(self, uv, **kwargs):
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
        ...

    def get_pixel_rays(self, uv_shift=(0, 0)):
        r"""For each pixel center calculates the respective 3D directions in the camera space.

        Returns
        -------
        direction : torch.Tensor
            of shape [3, h, w], directions in camera space, X to the right, Y down, Z from the camera.
            For pixels in the non-calibrated area the values are nan.
        """
        w, h = self.size_wh.cpu().tolist()
        uv = self.get_pix_uvs(uv_shift)
        directions = self.unproject(uv.view(2, -1)); del uv
        directions = directions.view(3, h, w)
        return directions

    def get_pix_uvs(self, uv_shift=(0, 0)):
        r"""For each pixel center calculates the respective UV coordinates.

        Returns
        -------
        uv : torch.Tensor
            of shape [2, h, w], image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).
        """
        w, h = self.size_wh.cpu().tolist()
        u_shift, v_shift = uv_shift
        v, u = torch.meshgrid([torch.linspace(.5 + v_shift, h - .5 + v_shift, h, device=self.device, dtype=self.dtype),
                               torch.linspace(.5 + u_shift, w - .5 + u_shift, w, device=self.device, dtype=self.dtype)])
        uv = torch.stack([u, v]); del u, v
        return uv
