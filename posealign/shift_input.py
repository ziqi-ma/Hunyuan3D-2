import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.fft

def center_object_by_alpha(image_rgba):
    """
    image_rgba: Tensor of shape (H, W, 4), values [0,1] or [0,255]
    returns: centered image (H, W, 4)
    """

    H, W, C = image_rgba.shape
    assert C == 4, "Image must have 4 channels (RGBA)."

    # Normalize if needed
    if image_rgba.max() > 1:
        image_rgba = image_rgba / 255.0

    # Use alpha channel as mask
    alpha = image_rgba[..., 3]
    mask = (alpha > 0.01).float()

    # Compute center of mass
    y_coords = torch.arange(H, device=image_rgba.device).float()
    x_coords = torch.arange(W, device=image_rgba.device).float()
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    mass = mask.sum() + 1e-8
    center_y = (mask * y_grid).sum() / mass
    center_x = (mask * x_grid).sum() / mass

    # Target center
    target_y = H / 2
    target_x = W / 2

    # Compute **integer shifts**
    shift_y = int(round((target_y - center_y).item()))
    shift_x = int(round((target_x - center_x).item()))

    # Create a new blank (transparent) canvas
    centered = torch.zeros_like(image_rgba)

    # Compute source and destination ranges
    src_y_start = max(0, -shift_y)
    src_y_end   = min(H, H - shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end   = min(H, H + shift_y)

    src_x_start = max(0, -shift_x)
    src_x_end   = min(W, W - shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end   = min(W, W + shift_x)

    # Move the pixels without interpolation
    centered[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = image_rgba[src_y_start:src_y_end, src_x_start:src_x_end, :]

    # Rescale back to 0-255 uint8
    centered = (centered.clamp(0,1) * 255).byte()

    return centered

def load_rgba_image(filepath):
    """Load RGBA image and return as torch tensor (H, W, 4), float [0,1]."""
    img = Image.open(filepath).convert('RGBA')
    img_np = np.array(img)  # (H, W, 4), uint8
    img_tensor = torch.from_numpy(img_np).float() / 255.0
    return img_tensor

'''
def save_rgba_tensor_as_png(tensor, save_path):
    """
    Save (H, W, 4) tensor as PNG
    """
    image_np = tensor.cpu().numpy()
    pil_image = Image.fromarray(image_np, mode="RGBA")
    pil_image.save(save_path)
'''

def save_rgba_tensor_as_png(tensor, save_path):
    """Save (H, W, 4) tensor as RGBA PNG."""
    tensor = (tensor.clamp(0,1) * 255).byte()
    img_np = tensor.cpu().numpy()
    img = Image.fromarray(img_np, mode='RGBA')
    img.save(save_path)

def find_shift_fft(I1, I2):
    """Find integer pixel shift (dy, dx) to best align I2 to I1."""
    if I1.ndim == 3:
        I1 = I1.mean(dim=-1)
    if I2.ndim == 3:
        I2 = I2.mean(dim=-1)

    I1 = I1 - I1.mean()
    I2 = I2 - I2.mean()

    F1 = torch.fft.fft2(I1)
    F2 = torch.fft.fft2(I2)

    R = F1 * F2.conj()
    R /= (R.abs() + 1e-8)

    r = torch.fft.ifft2(R)
    r = torch.fft.fftshift(r)
    r = r.real

    max_idx = torch.argmax(r)
    max_idx = torch.unravel_index(max_idx, r.shape)

    shift_y = max_idx[0] - I1.shape[0] // 2
    shift_x = max_idx[1] - I1.shape[1] // 2

    return shift_y, shift_x


def shift_image(image, shift_y, shift_x):
    """Shift image tensor (H, W, C) by integer (shift_y, shift_x), no interpolation."""
    H, W, C = image.shape
    shifted = torch.zeros_like(image)

    src_y_start = max(0, -shift_y)
    src_y_end   = min(H, H - shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end   = min(H, H + shift_y)

    src_x_start = max(0, -shift_x)
    src_x_end   = min(W, W - shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end   = min(W, W + shift_x)

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = image[src_y_start:src_y_end, src_x_start:src_x_end, :]
    return shifted



if __name__ == "__main__":

    # Load RGBA PNG
    #input_path = f"/data/ziqi/data/real_imgs/preprocessed/bunny1_old.png"
    #output_path = f"/data/ziqi/data/real_imgs/preprocessed/bunny1_shifted.png"

    # Load image as numpy
    #image_np = np.array(Image.open(input_path).convert("RGBA"))
    #image = torch.from_numpy(image_np).float()  # (H, W, 4)

    # Center it
    #centered_image = center_object_by_alpha(image)
    # Save
    #save_rgba_tensor_as_png(centered_image, output_path)
    
    name = "dog2"
    I1_path = "bestframe.jpeg"
    I2_path = f"/data/ziqi/data/real_imgs/preprocessed/{name}.png"
    output_path = f"/data/ziqi/data/real_imgs/adjusted/{name}.png"

    # Load images
    I1 = load_rgba_image(I1_path)
    I2 = load_rgba_image(I2_path)

    # Find shift
    shift_y, shift_x = find_shift_fft(I1, I2)
    print(f"Best shift found: (dy={shift_y}, dx={shift_x})")

    # Shift I2
    I2_shifted = shift_image(I2, shift_y, shift_x)

    # Save shifted image
    save_rgba_tensor_as_png(I2_shifted, output_path)
    print(f"Shifted image saved to {output_path}")

