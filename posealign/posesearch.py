from render.render import get_cameras_combinatoric, render_views, glb_to_py3d
import torch
import numpy as np
from PIL import Image


def search_pose(mesh, cameras, target_img, dists_repeat, elevs_repeat, azims_repeat):
    views = render_views(mesh, cameras, "", save=False) #num_views,512,512,4, range 0-1
    views = views[...,:3]*255 # n_views, 512,512,3
    mse = ((views - target_img.unsqueeze(0).cuda())**2).reshape(views.shape[0],-1).mean(dim=1)
    print(mse)
    best_idx = torch.argmin(mse)
    print(mse[best_idx])
    dist = dists_repeat[best_idx]
    elev = elevs_repeat[best_idx]
    azim = azims_repeat[best_idx]
    best_frame = views[best_idx,:,:,:].cpu().numpy()
    im = Image.fromarray(best_frame.astype(np.uint8))
    im.save(f"bestframe.jpeg")
    return dist, elev, azim
    


if __name__ == "__main__":
    name = "dog2"
    mesh = glb_to_py3d(f"/data/ziqi/data/hyout/{name}/out_ini.glb").cuda()

    input_img_path = f"/data/ziqi/data/real_imgs/preprocessed/{name}.png"
    input_img = torch.tensor(np.array(Image.open(input_img_path).convert("RGB"))).float()
    dists = torch.linspace(2.5,3.2,10)
    elevs = torch.linspace(-2,2,10)
    azims = torch.linspace(0,10,10)
    cameras, dists_repeat, elevs_repeat, azims_repeat = get_cameras_combinatoric(dists, elevs, azims, device = "cuda")

    dist, elev, azim = search_pose(mesh, cameras, input_img, dists_repeat, elevs_repeat, azims_repeat)
    print(dist)
    print(elev)
    print(azim)
