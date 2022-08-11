import torch
import numpy as np
import src.PyNvCodec as nvc
import PytorchNvCodec as pnvc


def surface_to_tensor(surface: nvc.Surface) -> torch.Tensor:
    """
    Converts planar rgb surface to cuda float tensor.
    """
    if surface.Format() != nvc.PixelFormat.RGB_PLANAR:
        raise RuntimeError('Surface shall be of RGB_PLANAR pixel format')

    surf_plane = surface.PlanePtr()
    img_tensor = pnvc.DptrToTensor(surf_plane.GpuMem(),
                                   surf_plane.Width(),
                                   surf_plane.Height(),
                                   surf_plane.Pitch(),
                                   surf_plane.ElemSize())
    if img_tensor is None:
        raise RuntimeError('Can not export to tensor.')

    img_tensor.resize_(3, int(surf_plane.Height()/3), surf_plane.Width())
    img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)
    img_tensor = torch.divide(img_tensor, 255.0)
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    return img_tensor