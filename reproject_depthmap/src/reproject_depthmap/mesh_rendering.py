import numpy as np
import open3d as o3d
import torch

from .camera_model import CameraModel


class MeshRenderer:
    r"""
    Parameters
    ----------
    scan : o3d.geometry.TriangleMesh
    """
    def __init__(self, scan):
        self.mesh = o3d.t.geometry.TriangleMesh.from_legacy(scan)
        self.raycasting = o3d.t.geometry.RaycastingScene()
        self.raycasting.add_triangles(self.mesh)

        self.device = 'cpu'
        self.dtype = torch.float32

        self.rays = None
        self.valid_ray_ids = None
        self.w = None
        self.h = None

    def set_rays_from_camera(self, camera_model):
        r"""
        Parameters
        ----------
        camera_model : CameraModel
        """
        rays = camera_model.get_pixel_rays()
        rays = rays.permute(1, 2, 0).to(self.device, self.dtype, memory_format=torch.contiguous_format).view(-1, 3)
        valid_ray_ids = rays.isfinite().all(1).nonzero().squeeze(1)
        if len(valid_ray_ids) == len(rays):
            self.valid_ray_ids = None
            self.rays = rays
        else:
            self.valid_ray_ids = valid_ray_ids
            self.rays = rays[valid_ray_ids].contiguous()
        self.w, self.h = camera_model.size_wh.tolist()

    def render_to_camera(self, cam_to_mesh_t, cam_to_mesh_rot, outputs, cull_back_faces=False):
        r"""
        Parameters
        ----------
        cam_to_mesh_t : torch.Tensor
            of shape [3]
        cam_to_mesh_rot : torch.Tensor
            of shape [:3, :3]
        outputs : list of str
            from {'ray_depth', 'z_depth', 'tri_ids', 'uv'}
        cull_back_faces : bool
            If True, invalidate the results for the rays hitting the back of the face.

        Returns
        -------
        render : dict
            ray_depth, z_depth: torch.Tensor
                of shape [height, width], inf for rays that didn't hit any face.
            tri_ids: torch.LongTensor
                of shape [height, width], o3d.t.geometry.RaycastingScene.INVALID_ID for rays that didn't hit any face.
            uv: torch.Tensor
                of shape [height, width, 2], all 0 for rays that didn't hit any face.
        """
        # Make rays
        cam_to_mesh_rot = cam_to_mesh_rot.to(self.rays)
        casted_rays = self.make_rays_from_cam(cam_to_mesh_t, cam_to_mesh_rot)

        # Trace rays
        raw_render = self.render_rays(casted_rays, cull_back_faces=cull_back_faces,
                                      get_tri_ids=('tri_ids' in outputs), get_bar_uvs=('uv' in outputs))

        # Collect outputs
        render = dict()
        for key in outputs:
            if key in {'ray_depth', 'z_depth'}:
                render[key] = self.get_depth(raw_render, key)
            if key == 'tri_ids':
                render[key] = self.get_tri_ids(raw_render)
            if key == 'uv':
                render[key] = self.get_uv(raw_render)
        return render

    def make_rays_from_cam(self, cam_to_mesh_t, cam_to_mesh_rot, rays=None):
        r"""Calculates the optical rays passing through centers of the pixels of the destination camera in the world space.

        Parameters
        ----------
        cam_to_mesh_t : torch.Tensor
            of shape [3]
        cam_to_mesh_rot : torch.Tensor
            of shape [:3, :3]
        rays : torch.Tensor
            of shape [rays_n, 3]

        Returns
        -------
        casted_rays : torch.Tensor
            of shape [rays_n, 6].
        """
        if rays is None:
            rays = self.rays
        casted_rays = rays.new_empty([len(rays), 6])
        ray_origins, ray_dirs = casted_rays[:, :3], casted_rays[:, 3:6]
        ray_origins.copy_(cam_to_mesh_t); del cam_to_mesh_t
        cam_to_mesh_rot = cam_to_mesh_rot.to(rays)
        torch.mm(rays, cam_to_mesh_rot.T, out=ray_dirs)
        return casted_rays

    def get_depth(self, raw_render, var='ray_depth'):
        r"""
        Parameters
        ----------
        raw_render
            ray_hit_depth : torch.Tensor
                of shape [rays_n].
        var : {'ray_depth', 'z_depth'}

        Returns
        -------
        depth : torch.Tensor
            of shape [height, width], inf for rays that didn't hit any face.
        """
        ray_depth = raw_render['ray_hit_depth']

        # Optionally, transform
        if var == 'ray_depth':
            depth = ray_depth
        elif var == 'z_depth':
            depth = ray_depth * self.rays[:, 2]
        del ray_depth

        # Set depth for invalid hits to inf
        if self.valid_ray_ids is not None:
            depth = self.scatter_ray_data(depth.unsqueeze(1), self.valid_ray_ids, float('inf'))
        depth = depth.view(self.h, self.w)
        return depth

    def get_tri_ids(self, raw_render):
        r"""
        Parameters
        ----------
        raw_render
            tri_ids : torch.Tensor
                of shape [rays_n].

        Returns
        -------
        tri_ids: torch.LongTensor
            of shape [height, width], o3d.t.geometry.RaycastingScene.INVALID_ID for rays that didn't hit any face.
        """
        tri_ids = raw_render['tri_ids']
        if self.valid_ray_ids is not None:
            tri_ids = self.scatter_ray_data(tri_ids.unsqueeze(1), self.valid_ray_ids,
                                            o3d.t.geometry.RaycastingScene.INVALID_ID)
        tri_ids = tri_ids.view(self.h, self.w)
        return tri_ids

    def get_uv(self, raw_render):
        r"""
        Parameters
        ----------
        raw_render
            bar_uvs : torch.Tensor
                of shape [rays_n, 2].

        Returns
        -------
        uv: torch.Tensor
            of shape [height, width, 2], all 0 for rays that didn't hit any face.
        """
        uv = raw_render['bar_uvs']
        if self.valid_ray_ids is not None:
            uv = self.scatter_ray_data(uv, self.valid_ray_ids, 0)
        uv = uv.view(self.h, self.w, 2)
        return uv

    def scatter_ray_data(self, ray_data, ray_ids, default_val=float('inf')):
        r"""Scatters ray data into image pixels.

        Parameters
        ----------
        ray_data : torch.Tensor
            of shape [valid_rays_n, channels_n].
        ray_ids : torch.LongTensor
            of shape [valid_rays_n].
        default_val : scalar

        Returns
        -------
        img_data : torch.Tensor
            of shape [height, width, channels_n].
        """
        channels_n = ray_data.shape[1]
        img_data = ray_data.new_full([self.h, self.w, channels_n], default_val, dtype=ray_data.dtype)
        img_data.view(-1, channels_n).index_copy_(0, ray_ids, ray_data); del ray_ids, ray_data
        return img_data

    def render_rays(
            self, casted_rays, cull_back_faces=False, backface_val=float('inf'), get_tri_ids=False, get_bar_uvs=False,
    ):
        r"""
        Parameters
        ----------
        casted_rays : torch.Tensor
            of shape [rays_n, 6].
        cull_back_faces : bool
            If True, set the depth value for the rays hitting the back of the face to `backface_val`.
        backface_val : float
        get_tri_ids : bool
        get_bar_uvs : bool

        Returns
        -------
        render : dict
            ray_hit_depth : torch.Tensor
                of shape [rays_n], inf for rays that didn't hit any face.
            normals : torch.Tensor
                of shape [rays_n, 3], all 0 for rays that didn't hit any face.
            tri_ids : torch.LongTensor
                of shape [rays_n], o3d.t.geometry.RaycastingScene.INVALID_ID for rays that didn't hit any face.
            bar_uvs : torch.Tensor
                of shape [rays_n, 2], all 0 for rays that didn't hit any face.
        """
        casted_rays_t = o3d.core.Tensor.from_numpy(casted_rays.numpy())
        result = self.raycasting.cast_rays(casted_rays_t); del casted_rays_t
        ray_hit_depth = torch.from_numpy(result['t_hit'].numpy())
        normals = torch.from_numpy(result['primitive_normals'].numpy())

        if cull_back_faces:
            ray_dirs = casted_rays[:, 3:6]
            hit_front_facing = (normals.unsqueeze(1) @ ray_dirs.unsqueeze(2)).squeeze(2).squeeze(1) < 0; del ray_dirs
            ray_hit_depth = ray_hit_depth.where(hit_front_facing, ray_hit_depth.new_tensor(backface_val))
            del hit_front_facing
        del casted_rays

        render = dict(ray_hit_depth=ray_hit_depth, normals=normals)
        if get_tri_ids:
            render['tri_ids'] = torch.from_numpy(result['primitive_ids'].numpy().astype(np.int64))
        if get_bar_uvs:
            render['bar_uvs'] = torch.from_numpy(result['primitive_uvs'].numpy())
        return render

    def interpolate_to_camera(self, attrs, render, default_val=float('nan')):
        r"""Renders vertex attributes.

        Parameters
        ----------
        attrs : torch.Tensor
            of shape [verts_n, attrs_n].
        render : dict
            tri_ids: torch.LongTensor
                of shape [height, width], o3d.t.geometry.RaycastingScene.INVALID_ID for rays that didn't hit any face.
            uv: torch.Tensor
                of shape [height, width, 2], all 0 for rays that didn't hit any face.
        default_val : scalar

        Returns
        -------
        samples : torch.Tensor
            of shape [height, width, attrs_n], with `default_val` for rays that didn't hit any face.
        """
        valid_tri_ids = (render['tri_ids'] != o3d.t.geometry.RaycastingScene.INVALID_ID).ravel().nonzero().squeeze(1)

        ray_render = dict()
        ray_render['tri_ids'] = render['tri_ids'].view(-1)[valid_tri_ids]
        ray_render['bar_uvs'] = render['uv'].view(-1, 2)[valid_tri_ids]
        ray_samples = self.interpolate(attrs, ray_render); del attrs, ray_render

        samples = self.scatter_ray_data(ray_samples, valid_tri_ids, default_val); del ray_samples, valid_tri_ids
        return samples

    def interpolate(self, attrs, render):
        r"""Renders vertex attributes.

        Parameters
        ----------
        attrs : torch.Tensor
            of shape [verts_n, attrs_n].
        render : dict
            tri_ids: torch.LongTensor
                of shape [rays_n].
            bar_uvs: torch.Tensor
                of shape [rays_n, 2].

        Returns
        -------
        samples : torch.Tensor
            of shape [rays_n, attrs_n].
        """
        tri_ids, bar_uvs = render['tri_ids'], render['bar_uvs']; del render
        attrs_n = attrs.shape[1]
        rays_n = tri_ids.shape[0]

        vert_weights = bar_uvs.new_empty(rays_n, 3)
        vert_weights[:, 1:3] = bar_uvs
        vert_weights[:, 0] = 1 - bar_uvs.sum(1); del bar_uvs

        tri_vert_ids = torch.from_numpy(self.mesh.triangle['indices'].numpy())
        vert_ids = tri_vert_ids[tri_ids]; del tri_vert_ids
        tri_attrs = attrs[vert_ids.view(-1)].view(rays_n, 3, attrs_n); del vert_ids

        samples = (vert_weights.unsqueeze(1) @ tri_attrs).squeeze(1); del vert_weights, tri_attrs
        return samples
