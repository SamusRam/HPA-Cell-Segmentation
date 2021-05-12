"""Package for loading and running the nuclei and cell segmentation models programmaticly."""
import os
import sys
from collections import defaultdict

import cv2
import imageio
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from skimage import transform, util

from hpacellseg.constants import (MULTI_CHANNEL_CELL_MODEL_URL,
                                  NUCLEI_MODEL_URL, TWO_CHANNEL_CELL_MODEL_URL)
from hpacellseg.utils import download_with_url

NORMALIZE = {"mean": [124 / 255, 117 / 255, 104 / 255], "std": [1 / (0.0167 * 255)] * 3}


class CellSegmentator(object):
    """Uses pretrained DPN-Unet models to segment cells from images."""

    def __init__(
            self,
            nuclei_model="./nuclei_model.pth",
            cell_model="./cell_model.pth",
            scale_factor=0.25,
            device="cuda",
            multi_channel_model=True,
    ):
        """Class for segmenting nuclei and whole cells from confocal microscopy images.

        It takes lists of images and returns the raw output from the
        specified segmentation model. Models can be automatically
        downloaded if they are not already available on the system.

        When working with images from the Huan Protein Cell atlas, the
        outputs from this class' methods are well combined with the
        label functions in the utils module.

        Note that for cell segmentation, there are two possible models
        available. One that works with 2 channeled images and one that
        takes 3 channels.

        Keyword arguments:
        nuclei_model -- A loaded torch nuclei segmentation model or the
                        path to a file which contains such a model.
                        If the argument is a path that points to a non-existant file,
                        a pretrained nuclei_model is going to get downloaded to the
                        specified path (default: './nuclei_model.pth').
        cell_model -- A loaded torch cell segmentation model or the
                      path to a file which contains such a model.
                      The cell_model argument can be None if only nuclei
                      are to be segmented (default: './cell_model.pth').
        model_width_height --
        device -- The device on which to run the models.
                  This should either be 'cpu' or 'cuda' or pointed cuda
                  device like 'cuda:0' (default: 'cuda').
        multi_channel_model -- Control whether to use the 3-channel cell model or not.
                               If True, use the 3-channel model, otherwise use the
                               2-channel version (default: True).
        """
        if device != "cuda" and device != "cpu" and "cuda" not in device:
            raise ValueError(f"{device} is not a valid device (cuda/cpu)")
        if device != "cpu":
            try:
                assert torch.cuda.is_available()
            except AssertionError:
                print("No GPU found, using CPU.", file=sys.stderr)
                device = "cpu"
        self.device = device

        if isinstance(nuclei_model, str):
            if not os.path.exists(nuclei_model):
                print(
                    f"Could not find {nuclei_model}. Downloading it now",
                    file=sys.stderr,
                )
                download_with_url(NUCLEI_MODEL_URL, nuclei_model)
            nuclei_model = torch.load(
                nuclei_model, map_location=torch.device(self.device)
            )
        if isinstance(nuclei_model, torch.nn.DataParallel) and device == "cpu":
            nuclei_model = nuclei_model.module

        self.nuclei_model = nuclei_model.to(self.device)

        self.multi_channel_model = multi_channel_model
        if isinstance(cell_model, str):
            if not os.path.exists(cell_model):
                print(
                    f"Could not find {cell_model}. Downloading it now", file=sys.stderr
                )
                if self.multi_channel_model:
                    download_with_url(MULTI_CHANNEL_CELL_MODEL_URL, cell_model)
                else:
                    download_with_url(TWO_CHANNEL_CELL_MODEL_URL, cell_model)
            cell_model = torch.load(cell_model, map_location=torch.device(self.device))
        self.cell_model = cell_model.to(self.device)
        self.scale_factor = scale_factor

    def _image_conversion(self, images):
        """Convert/Format images to RGB image arrays list for cell predictions.

        Intended for internal use only.

        Keyword arguments:
        images -- list of lists of image paths/arrays. It should following the
                 pattern if with er channel input,
                 [
                     [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                     [er_path0/image_array0, er_path1/image_array1, ...],
                     [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                 ]
                 or if without er input,
                 [
                     [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                     None,
                     [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                 ]
        """
        microtubule_imgs, er_imgs, nuclei_imgs = images
        if self.multi_channel_model:
            if not isinstance(er_imgs, list):
                raise ValueError("Please speicify the image path(s) for er channels!")
        else:
            if not er_imgs is None:
                raise ValueError(
                    "second channel should be None for two channel model predition!"
                )

        if not isinstance(microtubule_imgs, list):
            raise ValueError("The microtubule images should be a list")
        if not isinstance(nuclei_imgs, list):
            raise ValueError("The microtubule images should be a list")

        if er_imgs:
            if not len(microtubule_imgs) == len(er_imgs) == len(nuclei_imgs):
                raise ValueError("The lists of images needs to be the same length")
        else:
            if not len(microtubule_imgs) == len(nuclei_imgs):
                raise ValueError("The lists of images needs to be the same length")

        if not all(isinstance(item, np.ndarray) for item in microtubule_imgs):
            microtubule_imgs = [
                os.path.expanduser(item) for _, item in enumerate(microtubule_imgs)
            ]
            nuclei_imgs = [
                os.path.expanduser(item) for _, item in enumerate(nuclei_imgs)
            ]

            microtubule_imgs = list(
                map(lambda item: imageio.imread(item), microtubule_imgs)
            )
            nuclei_imgs = list(map(lambda item: imageio.imread(item), nuclei_imgs))
            if er_imgs:
                er_imgs = [os.path.expanduser(item) for _, item in enumerate(er_imgs)]
                er_imgs = list(map(lambda item: imageio.imread(item), er_imgs))

        if not er_imgs:
            er_imgs = [
                np.zeros(item.shape, dtype=item.dtype)
                for _, item in enumerate(microtubule_imgs)
            ]
        cell_imgs = list(
            map(
                lambda item: np.dstack((item[0], item[1], item[2])),
                list(zip(microtubule_imgs, er_imgs, nuclei_imgs)),
            )
        )

        return cell_imgs

    def pred_nuclei(self, images):
        """Predict the nuclei segmentation.

        Keyword arguments:
        images -- A list of image arrays or a list of paths to images.
                  If as a list of image arrays, the images could be 2d images
                  of nuclei data array only, or must have the nuclei data in
                  the blue channel; If as a list of file paths, the images
                  could be RGB image files or gray scale nuclei image file
                  paths.

        Returns:
        predictions -- A list of predictions of nuclei segmentation for each nuclei image.
        """

        def _preprocess(images):
            if isinstance(images[0], str):
                raise NotImplementedError('Currently the model requires images as numpy arrays, not paths.')
                # images = [imageio.imread(image_path) for image_path in images]
            self.target_shapes = [image.shape for image in images]
            images = [transform.rescale(image, self.scale_factor, multichannel=False)
                      for image in images]

            self.scaled_shapes = [image.shape for image in images]
            images = [cv2.copyMakeBorder(image, 32, (32 - image.shape[0] % 32), 32, (32 - image.shape[1] % 32),
                                         cv2.BORDER_REFLECT) for image in images]

            size_2_images = defaultdict(list)
            img_idx_2_size_idx = []
            for img_idx in range(len(images)):
                image = images[img_idx]
                img_size = image.shape[0]
                img_idx_2_size_idx.append((img_size, len(size_2_images[img_size])))
                size_2_images[img_size].append(
                    np.dstack((image[..., 2], image[..., 2], image[..., 2])) if len(image.shape) >= 3
                    else np.dstack((image, image, image)))
            for img_size, images in size_2_images.items():
                size_2_images[img_size] = np.array(images).transpose([0, 3, 1, 2])
            return size_2_images, img_idx_2_size_idx

        def _segment_helper(imgs):
            with torch.no_grad():
                mean = torch.as_tensor(NORMALIZE["mean"], device=self.device)
                std = torch.as_tensor(NORMALIZE["std"], device=self.device)
                imgs = torch.tensor(imgs).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])

                imgs = self.nuclei_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        size_2_preprocessed_imgs, img_idx_2_size_idx = _preprocess(images)

        size_2_predictins = dict()
        for img_size, preprocessed_imgs in size_2_preprocessed_imgs.items():
            predictions = _segment_helper(preprocessed_imgs)
            predictions = predictions.to("cpu").numpy()
            size_2_predictins[img_size] = predictions
        predictions_reordered = []
        for img_size, idx_per_size in img_idx_2_size_idx:
            predictions_reordered.append(size_2_predictins[img_size][idx_per_size])
        predictions = [self._restore_scaling(util.img_as_ubyte(pred), self.scaled_shapes[i])
                       for i, pred in enumerate(predictions_reordered)]
        nuc_size_medians = []
        nuc_counts = []

        for prediction in predictions:
            nuc_size_median, nuc_count = self.get_median_nucleus_size_and_nuc_count(prediction)
            nuc_size_medians.append(nuc_size_median)
            nuc_counts.append(nuc_count)

        return predictions, self.target_shapes, nuc_size_medians, nuc_counts


    def get_median_nucleus_size_and_nuc_count(self, prediction):
        blue_channel = prediction[:, :, 2].copy()
        blue_channel[prediction[..., 1] > 10] = 0
        _, _, _nuc_stats, _ = cv2.connectedComponentsWithStats(
            cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY)[1].astype('uint8'), 4)
        return np.median(_nuc_stats[1:, 4]), len(_nuc_stats) - 1


    def _restore_scaling(self, n_prediction, scaled_shape):
        """Restore an image from scaling and padding.

        This method is intended for internal use.
        It takes the output from the nuclei model as input.
        """
        n_prediction = n_prediction.transpose([1, 2, 0])
        n_prediction = n_prediction[
                       32: 32 + scaled_shape[0], 32: 32 + scaled_shape[1], ...
                       ]
        n_prediction[..., 0] = 0
        return n_prediction

    def pred_cells(self, images, precombined=False):
        """Predict the cell segmentation for a list of images.

        Keyword arguments:
        images -- list of lists of image paths/arrays. It should following the
                  pattern if with er channel input,
                  [
                      [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                      [er_path0/image_array0, er_path1/image_array1, ...],
                      [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                  ]
                  or if without er input,
                  [
                      [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                      None,
                      [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                  ]

                  The ER channel is required when multichannel is True
                  and required to be None when multichannel is False.

                  The images needs to be of the same size.
        precombined -- If precombined is True, the list of images is instead supposed to be
                       a list of RGB numpy arrays (default: False).

        Returns:
        predictions -- a list of predictions of cell segmentations.
        """

        def _preprocess(images):
            self.target_shapes = [image.shape for image in images]
            for image in images:
                if not len(image.shape) == 3:
                    raise ValueError("image should has 3 channels")

            images = [transform.rescale(image, self.scale_factor, multichannel=True)
                      for image in images]
            self.scaled_shapes = [image.shape for image in images]
            images = [cv2.copyMakeBorder(image, 32, (32 - image.shape[0] % 32), 32, (32 - image.shape[1] % 32),
                                         cv2.BORDER_REFLECT) for image in images]
            size_2_images = defaultdict(list)
            img_idx_2_size_idx = []
            for img_idx in range(len(images)):
                image = images[img_idx]
                img_size = image.shape[0]
                img_idx_2_size_idx.append((img_size, len(size_2_images[img_size])))
                size_2_images[img_size].append(image)
            for img_size, images in size_2_images.items():
                size_2_images[img_size] = np.array(images).transpose([0, 3, 1, 2])
            return size_2_images, img_idx_2_size_idx

        def _segment_helper(imgs):
            with torch.no_grad():
                mean = torch.as_tensor(NORMALIZE["mean"], device=self.device)
                std = torch.as_tensor(NORMALIZE["std"], device=self.device)
                imgs = torch.tensor(imgs).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])
                imgs = self.cell_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        if not precombined:
            images = self._image_conversion(images)
        size_2_preprocessed_imgs, img_idx_2_size_idx = _preprocess(images)
        size_2_predictins = dict()
        for img_size, preprocessed_imgs in size_2_preprocessed_imgs.items():
            predictions = _segment_helper(preprocessed_imgs)
            predictions = predictions.to("cpu").numpy()
            size_2_predictins[img_size] = predictions
        predictions_reordered = []
        for img_size, idx_per_size in img_idx_2_size_idx:
            predictions_reordered.append(size_2_predictins[img_size][idx_per_size])
        predictions = [self._restore_scaling(util.img_as_ubyte(pred), self.scaled_shapes[i])
                       for i, pred in enumerate(predictions_reordered)]
        return predictions, self.target_shapes
