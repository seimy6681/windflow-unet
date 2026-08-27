# utils.py
import torch
import torch.nn.functional as F

def calculate_photometric_residual(qv_t0, qv_t1, u_pred, v_pred):
    """
    Warps qv_t0 using predicted wind fields to guess qv_t1, 
    then returns the absolute difference from the true qv_t1.
    """
    img0 = torch.tensor(qv_t0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img1 = torch.tensor(qv_t1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    u = torch.tensor(u_pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    v = torch.tensor(v_pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    _, _, H, W = img0.shape

    # for 100 x 100 patch, 100x100, xx contains the x-coordinates (0 to 99) and yy contains the y-coordinates (0 to 99) for every single pixel on the grid.
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    yy = yy.float().unsqueeze(0).unsqueeze(0)
    xx = xx.float().unsqueeze(0).unsqueeze(0)

    sample_x = xx - u
    sample_y = yy - v

    sample_x = 2.0 * sample_x / max(W - 1, 1) - 1.0
    sample_y = 2.0 * sample_y / max(H - 1, 1) - 1.0

    grid = torch.cat((sample_x, sample_y), dim=1).permute(0, 2, 3, 1)
    warped_t1 = F.grid_sample(img0, grid, mode='bilinear', padding_mode='border', align_corners=True)
    residual = torch.abs(img1 - warped_t1)

    return residual.squeeze().numpy()