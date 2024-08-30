import numpy as np
import torch
from connectome import Transform
from dpipe.im.box import box2slices
from dpipe.predict import add_extract_dims, patches_grid
from imops import zoom_to_shape
from skimage.color import gray2rgb, rgb2gray
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

from m3da_exp.batch_iter import SPATIAL_DIMS
from m3da_exp.dataset.utils import scale_q
from m3da_exp.im.shape_utils import crop_to_body


class CycleGAN2DPredict(Transform):
    __inherit__ = True
    _size: int = 256
    _generator: nn.Module = None
    _transform: transforms.transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    _device: str = 'cuda'

    def image(image, _generator, _transform, _size, _device):
        Tensor = torch.cuda.FloatTensor if _device == 'cuda' else torch.Tensor

        img, box = crop_to_body(image, return_box=True)
        cropped_spatial_shape = img.shape[:-1]

        image_prep = gray2rgb(np.uint8(zoom_to_shape(np.ascontiguousarray(img), (_size, _size), axis=(0, 1)) * 255))
        image_pred = np.zeros_like(image_prep, dtype=np.float32)[..., 0]

        _input = Tensor(1, 3, _size, _size)
        for z in range(image_prep.shape[2]):
            img = _transform(image_prep[..., z, :])
            var = Variable(_input.copy_(img))
            with torch.no_grad():
                out = 0.5 * (_generator(var).data + 1.0)
            out = rgb2gray(np.transpose(out.detach().cpu().numpy()[0], axes=(1, 2, 0)))
            image_pred[..., z] = out

        image_pred = zoom_to_shape(np.ascontiguousarray(image_pred), cropped_spatial_shape, axis=(0, 1))
        image_pred = scale_q(image_pred)

        image_res = np.zeros_like(image)
        image_res[box2slices(box)] = image_pred

        return image_res


class CycleGAN3DPredict(Transform):
    __inherit__ = True
    _generator: nn.Module = None
    _patch_size: tuple = (128, 128, 32)
    # _patch_stride: tuple = (128, 128, 32)
    _patch_stride: tuple = (64, 64, 16)
    _device: str = 'cuda'

    def image(image, _generator, _patch_size, _patch_stride, _device):
        Tensor = torch.cuda.FloatTensor if _device == 'cuda' else torch.Tensor

        img, box = crop_to_body(np.float32(image), return_box=True)
        img = (scale_q(img, 0, 100) - 0.5) / 0.5

        @add_extract_dims(n_add=1, n_extract=1)
        @patches_grid(np.array(_patch_size), np.array(_patch_stride), axis=SPATIAL_DIMS)
        def predict(x):
            _input = Tensor(*x.shape)
            var = Variable(_input.copy_(torch.from_numpy(x)))
            with torch.no_grad():
                out = 0.5 * (_generator(var).data + 1.0)
            return out.detach().cpu().numpy()

        image_res = np.zeros_like(image)
        image_res[box2slices(box)] = scale_q(predict(img))
        return image_res
