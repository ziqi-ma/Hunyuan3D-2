import torch
import numpy as np
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer.mesh.textures import TexturesUV
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from PIL import Image
import itertools
import os
import imageio

def get_cameras_round(num_views, dist, device = None):
    # up and down alternating
    elev = torch.tile(torch.tensor([30,-20]), (num_views //2,))
    azim = torch.tile(torch.tensor(np.linspace(-180, 180, num=num_views//1, endpoint=False)).float(), (1,))
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras

def get_cameras_circle(num_views, dist, device = None):
    # up and down alternating
    elev = 20*torch.sin(torch.linspace(0, 2 * 3.1415, num_views))
    azim = torch.tensor(np.linspace(0, 360, num=num_views, endpoint=False)).float()
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras

def get_cameras_combinatoric(dists, elevs, azims, device=None):
    combinations = list(itertools.product(dists, elevs, azims))
    dists_repeat, elevs_repeat, azims_repeat = zip(*combinations)
    dists_repeat = torch.tensor(dists_repeat)
    elevs_repeat = torch.tensor(elevs_repeat)
    azims_repeat = torch.tensor(azims_repeat)
    R,T = look_at_view_transform(dists_repeat, elevs_repeat, azims_repeat)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras, dists_repeat, elevs_repeat, azims_repeat


def glb_to_py3d(path):
    import trimesh
    trimesh_scene = trimesh.load(path)
    meshes_list = []
    for node in list(trimesh_scene.geometry.keys()):
        geometry = trimesh_scene.geometry[node]
        # skip if it doesn't have faces (e.g. Path3D is curves in 3D)
        if not hasattr(geometry, 'faces'):
            continue
        verts = torch.tensor(geometry.vertices).float()
        try:
            transform_tuple = trimesh_scene.graph.get(node)
            assert transform_tuple[1] == node
            transform_mat = torch.tensor(transform_tuple[0]).float()
            verts_transformed = (transform_mat @ torch.cat([verts, torch.ones(verts.shape[0],1)], axis=1).T)[:3,:].T   
        except Exception:
            verts_transformed = verts
        faces = torch.tensor(geometry.faces)


        if geometry.visual.material.baseColorTexture and len(np.asarray(geometry.visual.material.baseColorTexture).shape)>=3:
            # convert to rgb if greyscale
            if geometry.visual.material.baseColorTexture.mode in ["LA", "L"]:
                geometry.visual.material.baseColorTexture = geometry.visual.material.baseColorTexture.convert("RGB")
            texture_map = (torch.tensor(np.asarray(geometry.visual.material.baseColorTexture))/255)[:,:,:3]
            verts_uv = torch.tensor(geometry.visual.uv).float()
            # some uv needs to be wrapped around
            if verts_uv.min() < 0:
                verts_uv = (verts_uv - verts_uv.min(0, keepdim=True)[0]) / (verts_uv.max(0, keepdim=True)[0] - verts_uv.min(0, keepdim=True)[0])
            cur_mesh = Meshes(verts=[verts_transformed], faces=[faces], textures=TexturesUV(maps=[texture_map.float()], faces_uvs=[faces], verts_uvs=[verts_uv]))
            meshes_list.append(cur_mesh)
        else:
            # no texture map, take main color, but this still needs to be of type TexturesUV because py3d asks that
            color = torch.tensor(geometry.visual.material.main_color[:3])/255 # this is rgba
            texture_map = torch.tile(color, (100, 100, 1)).float()
            verts_uv = torch.ones(verts_transformed.shape[0],2)*0.5
            try:
                verts_uv = torch.tensor(geometry.visual.uv).float()
            except Exception:
                pass
            cur_mesh = Meshes(verts=[verts_transformed], faces=[faces], textures=TexturesUV(maps=[texture_map], faces_uvs=[faces], verts_uvs=[verts_uv]))
            meshes_list.append(cur_mesh)
    if len(meshes_list) == 0:
        raise Exception("all meshes fail to meet criteria")
    overall_mesh = join_meshes_as_scene(meshes_list)
    return overall_mesh


def get_rasterizer(image_size, blur_radius, faces_per_pixel, cameras, device = None):
    if device is None:
        device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel= faces_per_pixel,
        bin_size = 0,
        perspective_correct=False, # this is important, otherwise gradients will explode!!
    )
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    return rasterizer


def get_phong_shader(cameras, lights, device = None):
    if device is None:
        device = torch.device("cpu")
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
    return shader

def render_views(mesh, cameras, out_dir, save=True):
    
    num_views = len(cameras)
    # Create a bright white point light
    lights = PointLights(
        device='cuda',
        location=[[0,0,-10]],
        diffuse_color=[[1, 1, 1]]
    )
    lights = PointLights(
        device='cuda',
        location=[[0,0,10]],
    )
    rasterizer = get_rasterizer(512, 0, 1, cameras, device='cuda')
    shader = get_phong_shader(cameras, lights, device="cuda")

    # do in chunk to avoid OOM
    CHUNK_SIZE = 20
    n_chunks = num_views // CHUNK_SIZE + 1
    images_all = []
    for i in range(n_chunks):
        if i*CHUNK_SIZE == num_views:
            continue
        end_idx = min((i+1)*CHUNK_SIZE, num_views)
        curcameras = cameras[list(range(i*CHUNK_SIZE,end_idx))]
        cur_chunk_size = len(curcameras)
        fragments = rasterizer(mesh.extend(cur_chunk_size), cameras = curcameras)
        images = shader(fragments, mesh.extend(cur_chunk_size), cameras=curcameras, lights=lights)
        # make background black
        rgb = images[..., :3]      # (N, H, W, 3)
        alpha = (images[..., 3:]>0)*1.0    # (N, H, W, 1)
        # Composite over black background
        images = rgb * alpha
        images_all.append(images)
               
        if save:
            os.makedirs(out_dir, exist_ok=True)
            for j in range(num_views):
                rgb = images[j,:,:,:3].cpu().numpy()*255
                im = Image.fromarray(rgb.astype(np.uint8))
                im.save(f"{out_dir}/{i*CHUNK_SIZE+j}.jpeg")
    return torch.cat(images_all, dim=0)


def render_views_soft(mesh, cameras, out_dir, blur_radius, save=True):
    num_views = len(cameras)
    # Create a bright white point light
    lights = PointLights(
        device='cuda',
        location=[[0,0,-10]],
        diffuse_color=[[1, 1, 1]]
    )
    lights = PointLights(
        device='cuda',
        location=[[0,0,10]],
    )
    rasterizer = get_rasterizer(512, blur_radius, 5, cameras, device='cuda')
    shader = get_phong_shader(cameras, lights, device="cuda")

    # do in chunk to avoid OOM
    CHUNK_SIZE = 50
    n_chunks = num_views // CHUNK_SIZE + 1
    images_all = []
    for i in range(n_chunks):
        if i*CHUNK_SIZE == num_views:
            continue
        end_idx = min((i+1)*CHUNK_SIZE, num_views)
        curcameras = cameras[list(range(i*CHUNK_SIZE,end_idx))]
        cur_chunk_size = len(curcameras)
        fragments = rasterizer(mesh.extend(cur_chunk_size), cameras = curcameras)
        images = shader(fragments, mesh.extend(cur_chunk_size), cameras=curcameras, lights=lights)
        # make background black
        rgb = images[..., :3]      # (N, H, W, 3)
        alpha = (images[..., 3:]>0)*1.0    # (N, H, W, 1)
        # Composite over black background
        images = rgb * alpha
        images_all.append(images)
               
        if save:
            os.makedirs(out_dir, exist_ok=True)
            for j in range(num_views):
                rgb = images[j,:,:,:3].cpu().numpy()*255
                im = Image.fromarray(rgb.astype(np.uint8))
                im.save(f"{out_dir}/{i*CHUNK_SIZE+j}.jpeg")
    return torch.cat(images_all, dim=0)

def render_video(mesh, distance, out_path, views=300):
    cameras = get_cameras_circle(views, distance, device=mesh.device)
    views = render_views(mesh, cameras, "", save=False)[:,:,:,:3].cpu().numpy()*255
    views = views.astype(np.uint8)
    imageio.mimsave(out_path, views, fps=30)



if __name__=="__main__":
    mesh = glb_to_py3d("../demo.glb").cuda()
    #cameras = get_cameras_round(20, 4, device = "cuda")
    #render_views(mesh, cameras, "/data/ziqi/data/hyout/cat1/original")
