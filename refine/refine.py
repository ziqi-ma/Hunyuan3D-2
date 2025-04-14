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
import wandb

def laplacian_smoothness_loss(image):
    """
    image: Tensor of shape (H, W, 3)
    returns: scalar loss
    """

    # image: (H, W, 3)
    # Center pixel
    center = image[1:-1, 1:-1, :]
    
    # 4-connected neighbors
    top    = image[:-2, 1:-1, :]
    bottom = image[2:, 1:-1, :]
    left   = image[1:-1, :-2, :]
    right  = image[1:-1, 2:, :]
    
    # Discrete Laplacian: -4*center + top + bottom + left + right
    lap = -4 * center + top + bottom + left + right

    # Loss: mean squared Laplacian
    loss = (lap ** 2).mean()
    return loss


def pixel_diff_loss(image):
    return ((image[:, :-1, :] - image[:, 1:, :])**2).mean() + ((image[:-1, :, :] - image[1:, :, :])**2).mean()


def opt_texture(image_path, mesh_path, out_dir, laplacian_w):
    cameras,_,_,_ = get_cameras_combinatoric([dist], [elev], [azim], device='cuda')
    
    input_img = torch.tensor(np.array(Image.open(image_path).convert("RGB"))).float().cuda() # 512,512,3, range to 255

    mesh = glb_to_py3d(mesh_path).cuda()
    texture = mesh.textures._maps_list[0] # 2048,2048,3, in range 0-1
    texture.requires_grad_(True)
    optimizer = optim.Adam([texture], lr=lr)
    optimizer.zero_grad()

    wandb.init(
        project="hytexture_opt",
        name=f"lr{lr}reg{laplacian_w}"
    )

    for i in range(n_iter):
        mesh.textures = TexturesUV(
            maps=texture.unsqueeze(0),  # must be batched
            faces_uvs=mesh.textures.faces_uvs_list(),
            verts_uvs=mesh.textures.verts_uvs_list()
        )
        # render
        view = render_views_soft(mesh, cameras, "", blur_radius = blur_radius, save=False)[...,:3].squeeze()*255 # 512,512,3, range 0-255
        # get loss
        l2_view = ((input_img - view)**2).mean()

        # get laplacian regularization on texture
        laplacian = laplacian_smoothness_loss(texture*255) #pixel_diff_loss(texture*255)#

        

        loss = l2_view + laplacian*laplacian_w

        print(f"loss:{loss.item()}, l2:{l2_view.item()}, laplacian:{laplacian*laplacian_w}")

        wandb.log({
            "iter": i+1,
            "loss":loss.item(),
            "l2": l2_view.item(),
            "laplacian": laplacian.item()
        })

        
        if (i) % viz_freq == 0 or i == n_iter-1:
            # save
            texture_im = Image.fromarray((texture*255).detach().cpu().numpy().astype(np.uint8))
            texture_im.save(f"{out_dir}/iter{i}texture.jpeg")
            curview = view.detach().cpu().numpy()
            im = Image.fromarray(curview.astype(np.uint8))
            im.save(f"{out_dir}/iter{i}.jpeg")
            # render other views
            with torch.no_grad():
                render_video(mesh, distance=4, out_path=f"{out_dir}/iter{i}.mp4", views=300)

        if (i+1) % decay_freq == 0:
            laplacian_w *= 0.5


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



if __name__ == "__main__":
    lr = 3e-3
    n_iter = 10000
    decay_freq = 2000
    viz_freq = 300
    laplacian_w = 20
    blur_radius = 0.00001

    name = "dog2"
    image_path = f'/data/ziqi/data/real_imgs/adjusted/{name}.png'
    out_dir = f"/data/ziqi/data/hyout/optout/{name}/lr{lr}lap{laplacian_w}decay{decay_freq}"
    mesh_path = f'/data/ziqi/data/hyout/{name}/out_ini.glb'
    os.makedirs(out_dir, exist_ok=True)

    # cat1
    #dist = 2.5579
    #elev = 2.6842
    #azim = -6.1579
    # lr = 1e-3
    # n_iter = 10000
    # decay_freq=3000
    #laplacian_w = 5000
    # blur_radius=0.001

    # bunny1
    #dist = 3.
    #elev = -5.
    #azim = 26.0526
    #n_iter = 1000
    #lr = 3e-3
    #decay_freq = 1000
    #viz_freq = 500
    #laplacian_w = 50
    #blur_radius=0.00001

    # cat2
    #dist = 2.5000
    #elev = -3.3333
    #azim = 21.1111
    #n_iter = 1000
    #decay_freq = 10000
    #viz_freq = 100
    #laplacian_w = 50
    #blur_radius = 0.00001

    # dog1
    #dist = 2.8000
    #elev = -0.5556
    #azim = -16.1111
    #lr = 3e-3
    #n_iter = 10000
    #decay_freq = 10000
    #viz_freq = 500
    #laplacian_w = 250
    #blur_radius = 0.00001

    # dog2
    dist = 2.6556
    elev = 2.
    azim = 5.5556


    opt_texture(image_path, mesh_path, out_dir, laplacian_w)