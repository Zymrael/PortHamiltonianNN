import torch

def genpoints(xmin, xmax, ymin, ymax, n_points):
    """Generate a meshgrid contained in [xmin, xmax], [ymin, ymax] with `n_points` points"""
    xx = torch.linspace(xmin, xmax, n_points)
    yy = torch.linspace(ymin, ymax, n_points)
    p = []
    for i in range(n_points):
        for j in range(n_points):
            p.append([xx[i], yy[j]])
    return torch.Tensor(p)