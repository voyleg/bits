import open3d as o3d

from .camera_model import CameraModel
from .mesh_rendering import MeshRenderer
from .triangulation import triangulate_grid


class DepthmapRenderer(MeshRenderer):
    r"""Represents a renderer of a mesh made from a depth map.

    Parameters
    ----------
    depthmap : torch.Tensor
        of shape [h, w].
    depth_is_valid : torch.BoolTensor
        of shape [h, w]. False for pixels of the depthmap with missing value.
    cam_model : CameraModel
    cam_to_world : torch.Tensor
        of shape [4, 4].
    max_rel_edge_len : float
        Triangles with edges longer than the distance to the triangle from the depthmap source camera multiplied by
        `max_rel_edge_len` are considered false boundaries.
    attrs : torch.Tensor
        of shape [h, w, attrs_n], mesh attributes.
    """
    def __init__(self, depthmap, depth_is_valid, cam_model, cam_to_world=None, max_rel_edge_len=None, attrs=None):
        mesh, tri_is_valid, vert_attrs = mesh_from_depthmap(depthmap, depth_is_valid, cam_model,
                                                            cam_to_world, max_rel_edge_len, attrs)
        super().__init__(mesh)
        self.tri_is_valid = tri_is_valid
        self.vert_attrs = vert_attrs

    def render_rays(
            self, casted_rays, cull_back_faces=False, backface_val=float('inf'), get_tri_ids=False, get_bar_uvs=False
    ):
        r"""
        Parameters
        ----------
        casted_rays : torch.Tensor
            of shape [rays_n, 6]
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
            tri_ids : torch.Tensor
                of shape [rays_n], o3d.t.geometry.RaycastingScene.INVALID_ID for rays that didn't hit any face.
            bar_uvs : torch.Tensor
                of shape [rays_n, 2], all 0 for rays that didn't hit any face.
        """
        render = super().render_rays(casted_rays, cull_back_faces, backface_val,
                                     get_tri_ids=True, get_bar_uvs=get_bar_uvs)

        tri_is_valid = self.tri_is_valid[render['tri_ids'].clamp(max=len(self.tri_is_valid) - 1)]
        render['ray_hit_depth'] = render['ray_hit_depth'].where(
            tri_is_valid, render['ray_hit_depth'].new_tensor(float('inf')))
        render['normals'] = render['normals'].where(tri_is_valid.unsqueeze(1), render['normals'].new_tensor(0))
        if get_tri_ids:
            render['tri_ids'] = render['tri_ids'].where(
                tri_is_valid, render['tri_ids'].new_tensor(o3d.t.geometry.RaycastingScene.INVALID_ID))
        else:
            del render['tri_ids']
        if get_bar_uvs:
            render['bar_uvs'] = render['bar_uvs'].where(tri_is_valid.unsqueeze(-1), render['bar_uvs'].new_tensor(0))
        return render


def mesh_from_depthmap(depthmap, depth_is_valid, cam_model, cam_to_world=None, max_rel_edge_len=None, attrs=None):
    r"""Makes a mesh from the depthmap for further reprojection to different points of view.

    Parameters
    ----------
    depthmap : torch.Tensor
        of shape [h, w].
    depth_is_valid : torch.BoolTensor
        of shape [h, w], False for pixels of the depthmap with missing value.
    cam_model : CameraModel
    cam_to_world : torch.Tensor
        of shape [4, 4].
    max_rel_edge_len : float
        Triangles with edges longer than the distance to the triangle from the depthmap source camera multiplied by
        `max_rel_edge_len` are considered false boundaries.
    attrs : torch.Tensor
        of shape [h, w, attrs_n], mesh attributes.

    Returns
    -------
    mesh : o3d.geometry.TriangleMesh
    tri_is_valid : torch.BoolTensor
        of shape [tris_n], False for triangles at false boundaries.
    vert_attrs : torch.Tensor
        of shape [verts_n, attrs_n], vertex attributes.
    """
    depthmap = depthmap.where(depth_is_valid, depthmap[depth_is_valid].max())

    rays = cam_model.get_pixel_rays(); del cam_model
    uvo_n = rays / rays[2]; del rays
    xyz = uvo_n * depthmap; del uvo_n, depthmap

    xyz = xyz.permute(1, 2, 0)
    mask = xyz.isfinite().all(2)
    tris = triangulate_grid(*mask.shape, mask)
    xyz = xyz[mask]
    depth_is_valid = depth_is_valid[mask]
    if attrs is not None:
        vert_attrs = attrs[mask]
    else:
        vert_attrs = None
    del mask

    tri_is_valid = depth_is_valid[tris.view(-1)].view_as(tris).all(1); del depth_is_valid
    if max_rel_edge_len is not None:
        tri_verts = xyz[tris.view(-1)].view(*tris.shape, 3)
        min_vert_dists = tri_verts.norm(dim=2).min(1)[0]
        edges = tri_verts - tri_verts.roll(1, 1); del tri_verts
        max_edge_lens = edges.norm(dim=2).max(1)[0]; del edges
        rel_edge_lens = max_edge_lens.div_(min_vert_dists); del max_edge_lens, min_vert_dists
        tri_is_small = rel_edge_lens <= max_rel_edge_len; del rel_edge_lens
        tri_is_valid = tri_is_valid.logical_and_(tri_is_small); del tri_is_small

    if cam_to_world is not None:
        xyz = xyz @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]; del cam_to_world

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices.extend(xyz.double().numpy()); del xyz
    mesh.triangles.extend(tris.long().numpy())
    return mesh, tri_is_valid, vert_attrs
