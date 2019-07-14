import torch

def genpoints(xmin,
              xmax,
              ymin,
              ymax,
              number_points
              ):

    xx = torch.linspace(xmin,xmax,number_points)
    yy = torch.linspace(ymin,ymax,number_points)
    p = []
    for i in range(number_points):
        for j in range(number_points):
            p.append([xx[i], yy[j]])
    return torch.Tensor(p)