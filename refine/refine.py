import torch
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline
import trimesh
from pytorch3d.renderer.mesh.textures import TexturesUV
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
import torch.optim as optim
from render.render import glb_to_py3d, get_cameras_combinatoric, render_views_soft, get_cameras_round, render_views, render_video
import numpy as np
import os


def opt_texture(image_path, mesh_path, out_dir):
    cameras,_,_,_ = get_cameras_combinatoric([dist], [elev], [azim], device='cuda')
    
    input_img = torch.tensor(np.array(Image.open(image_path).convert("RGB"))).float().cuda() # 512,512,3, range to 255

    mesh = glb_to_py3d(mesh_path).cuda()
    texture = mesh.textures._maps_list[0] # 2048,2048,3, in range 0-1
    texture.requires_grad_(True)
    optimizer = optim.Adam([texture], lr=lr)
    optimizer.zero_grad()

    for i in range(n_iter):
        mesh.textures = TexturesUV(
            maps=texture.unsqueeze(0),  # must be batched
            faces_uvs=mesh.textures.faces_uvs_list(),
            verts_uvs=mesh.textures.verts_uvs_list()
        )
        # render
        view = render_views_soft(mesh, cameras, "", save=False)[...,:3].squeeze()*255 # 512,512,3, range 0-255
        # get loss
        l2_view = ((input_img - view)**2).mean()
        print(l2_view.item())
        l2_view.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % viz_freq == 0:
            # save
            texture_im = Image.fromarray((texture*255).detach().cpu().numpy().astype(np.uint8))
            texture_im.save(f"{out_dir}/iter{i+1}texture.jpeg")
            curview = view.detach().cpu().numpy()
            im = Image.fromarray(curview.astype(np.uint8))
            im.save(f"{out_dir}/iter{i+1}.jpeg")
            # render other views
            with torch.no_grad():
                render_video(mesh, distance=4, out_path=f"{out_dir}/iter{i+1}.mp4", views=300)



if __name__ == "__main__":
    lr = 3e-3
    n_iter = 600
    viz_freq = 50
    name = "cat1"
    image_path = '/data/ziqi/data/real_imgs/preprocessed/cat1.png'
    texture_ini_path = "/data/ziqi/data/hyout/cat1/texture_ini.pt"
    out_dir = f"/data/ziqi/data/hyout/optout/{name}/lr{lr}"
    mesh_path = '../demo.glb'
    os.makedirs(out_dir, exist_ok=True)

    # cat1
    dist = 2.5579
    elev = 2.6842
    azim = -6.1579

    opt_texture(image_path, mesh_path, out_dir)