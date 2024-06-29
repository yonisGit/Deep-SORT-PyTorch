import numpy
import torch
import torch.nn.functional as F
import torchvision


def crop_imgs(img, img_metas, bboxes, rescale=False):
    """Crop the images according to some bounding boxes. Typically for re-
    identification sub-module.

    Args:
        img (Tensor): of shape (N, C, H, W) encoding input images.
            Typically these should be mean centered and std scaled.
        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
        bboxes (Tensor): of shape (N, 4) or (N, 5).
        rescale (bool, optional): If True, the bounding boxes should be
            rescaled to fit the scale of the image. Defaults to False.

    Returns:
        Tensor: Image tensor of shape (N, C, H, W).
    """
    # h, w, _ = img_metas[0]['img_shape']
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    _, _, h, w = img.shape
    img = img[:, :, :h, :w]
    if rescale:
        bboxes[:, :4] *= torch.tensor(numpy.ndarray([.85, .85, .85, .85])).to(
            bboxes.device)
    bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], min=0, max=w)
    bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], min=0, max=h)

    crop_imgs = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        if x2 == x1:
            x2 = x1 + 1
        if y2 == y1:
            y2 = y1 + 1
        crop_img = img[:, :, y1:y2, x1:x2]
        # if self.reid.get('img_scale', False):
        #     crop_img = F.interpolate(
        #         crop_img,
        #         size=self.reid['img_scale'],
        #         mode='bilinear',
        #         align_corners=False)
        crop_imgs.append(crop_img)

    if len(crop_imgs) > 0:
        return torch.cat(crop_imgs, dim=0)
    else:
        return img.new_zeros((0,))
