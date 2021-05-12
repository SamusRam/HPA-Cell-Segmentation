"""Utility functions for the HPA Cell Segmentation package."""
import os.path
import urllib
import zipfile

import cv2
import numpy as np
import scipy.ndimage as ndi
from hpacellseg.utils_cython import filter_single_nuc_borders, get_border_nuclei, remove_labels
from skimage import filters, measure, segmentation
from skimage.morphology import (binary_erosion, closing, disk,
                                remove_small_holes, remove_small_objects)

HIGH_THRESHOLD = 0.4
LOW_THRESHOLD = HIGH_THRESHOLD - 0.25


def download_with_url(url_string, file_path, unzip=False):
    """Download file with a link."""
    with urllib.request.urlopen(url_string) as response, open(
            file_path, "wb"
    ) as out_file:
        data = response.read()  # a `bytes` object
        out_file.write(data)

    if unzip:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))


def __fill_holes(image):
    """Fill_holes for labelled image, with a unique number."""
    boundaries = segmentation.find_boundaries(image)
    image = np.multiply(image, np.invert(boundaries))
    image = ndi.binary_fill_holes(image > 0)
    image = ndi.label(image)[0]
    return image


def label_nuclei(nuclei_pred):
    """Return the labeled nuclei mask data array.

    This function works best for Human Protein Atlas cell images with
    predictions from the CellSegmentator class.

    Keyword arguments:
    nuclei_pred -- a 3D numpy array of a prediction from a nuclei image.

    Returns:
    nuclei-label -- An array with unique numbers for each found nuclei
                    in the nuclei_pred. A value of 0 in the array is
                    considered background, and the values 1-n is the
                    areas of the cells 1-n.
    """
    borders = (nuclei_pred[..., 1] > 0.05).astype(np.uint8)
    m = nuclei_pred[..., 2] * (1 - borders)

    img_copy = np.zeros_like(nuclei_pred[..., 2])
    img_copy[m > LOW_THRESHOLD] = 1
    img_copy = img_copy.astype(np.bool)
    img_copy = binary_erosion(img_copy)
    # TODO: Add parameter for remove small object size for
    #       differently scaled images.
    # img_copy = remove_small_objects(img_copy, 500)
    img_copy = img_copy.astype(np.uint8)
    markers = measure.label(img_copy).astype(np.uint32)

    mask_img = np.zeros_like(nuclei_pred[..., 2])
    mask_img[mask_img > HIGH_THRESHOLD] = 1
    mask_img = mask_img.astype(np.bool)
    mask_img = remove_small_holes(mask_img, 1000)
    # TODO: Figure out good value for remove small objects.
    # mask_img = remove_small_objects(mask_img, 8)
    mask_img = mask_img.astype(np.uint8)
    nuclei_label = segmentation.watershed(
        mask_img, markers, mask=mask_img, watershed_line=True
    )
    nuclei_label = remove_small_objects(nuclei_label, 2500)
    nuclei_label = measure.label(nuclei_label)
    return nuclei_label


def label_cell(nuclei_pred, cell_pred, target_size, median_nuc_size, nuc_count,
               split_nuclei=False, return_nuclei_label=True):
    """Label the cells and the nuclei.

    Keyword arguments:
    nuclei_pred -- a 3D numpy array of a prediction from a nuclei image.
    cell_pred -- a 3D numpy array of a prediction from a cell image.

    Returns:
    A tuple containing:
    nuclei-label -- A nuclei mask data array.
    cell-label  -- A cell mask data array.

    0's in the data arrays indicate background while a continous
    strech of a specific number indicates the area for a specific
    cell.
    The same value in cell mask and nuclei mask refers to the identical cell.

    NOTE: The nuclei labeling from this function will be sligthly
    different from the values in :func:`label_nuclei` as this version
    will use information from the cell-predictions to make better
    estimates.
    """
    default_size = 512
    thresholds_scaler = (nuclei_pred.shape[0] / default_size) ** 2

    def __wsh(
            mask_img,
            threshold,
            border_img,
            seeds,
            threshold_adjustment=0.35,
            small_object_size_cutoff=10 * thresholds_scaler,
    ):
        img_copy = np.zeros_like(mask_img)
        m = seeds * border_img  # * dt
        img_copy[m > threshold + threshold_adjustment] = 1
        img_copy = img_copy.astype(np.bool)
        img_copy = remove_small_objects(img_copy, small_object_size_cutoff).astype(
            np.uint8
        )

        mask_img = np.where(mask_img <= threshold, 0, 1)
        mask_img = mask_img.astype(np.bool)
        mask_img = remove_small_holes(mask_img, 62.5 * thresholds_scaler)
        mask_img = remove_small_objects(mask_img, 0.5 * thresholds_scaler).astype(np.uint8)
        markers = ndi.label(img_copy, output=np.uint32)[0]
        labeled_array = segmentation.watershed(
            mask_img, markers, mask=mask_img, watershed_line=True
        )
        return labeled_array

    if split_nuclei and nuc_count >= 49:
        # splitting nuclei based on cell borders
        nuclei_borders_bin = cv2.threshold(nuclei_pred[:, :, 1], 150*thresholds_scaler if thresholds_scaler <= 1 else 200, 255, cv2.THRESH_BINARY)[1]
        nuclei_borders_bin = cv2.morphologyEx((nuclei_borders_bin > 0).astype(np.uint8),
                                                        cv2.MORPH_OPEN,
                                                        np.ones((2, 2), np.uint8))
        nuclei_bin = nuclei_pred[:, :, 2] > 0
        num_cc_nuc_borders, cc_nuc_borders = cv2.connectedComponents(nuclei_borders_bin)
        cc_nuc_borders = measure.label(cc_nuc_borders > 0).astype(np.int32)

        _, nuclei_labels_thresh = cv2.threshold(nuclei_pred[:, :, 2], 150, 255, cv2.THRESH_BINARY)
        nuclei_labels_thresh_markers = nuclei_labels_thresh.copy()
        nuclei_labels_thresh_markers[nuclei_borders_bin > 0] = 0
        nuclei_labels_thresh_markers, _ = ndi.label(cv2.erode(nuclei_labels_thresh, np.ones((3, 3), np.uint8)))
        distance = ndi.distance_transform_edt(nuclei_labels_thresh)

        cc_nuclei = segmentation.watershed(-distance, markers=nuclei_labels_thresh_markers, mask=nuclei_bin)

        all_nuclei_borders_single_nuc, general_borders = filter_single_nuc_borders(nuclei_borders_bin, #nuclei_borders_inner_overlap,
                                                                  cc_nuc_borders,
                                                                  num_cc_nuc_borders,
                                                                  cc_nuclei,
                                                                  zero_intensity_threshold=0,
                                                                  border_size_threshold=5*thresholds_scaler if thresholds_scaler <= 1 else 10)

        all_nuclei_borders_single_nuc = np.asarray(all_nuclei_borders_single_nuc, np.uint8)

        line_image = np.zeros_like(cc_nuclei)
        for nuclei_borders_single_nuc in all_nuclei_borders_single_nuc:

            lines = cv2.HoughLinesP(
                np.asarray(nuclei_borders_single_nuc, np.uint8),
                rho=2,
                theta=np.pi / 60,
                threshold=5,
                #         lines=np.array([]),
                minLineLength=np.sqrt(median_nuc_size) / 10,
                maxLineGap=30 if thresholds_scaler <= 1 else 60
            )
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(line_image, (x1, y1), (x2, y2), color=255, thickness=2)

        lines_general_borders = cv2.HoughLinesP(
            np.asarray(general_borders, np.uint8),
            rho=2,
            theta=np.pi / 60,
            threshold=40,
            minLineLength=np.sqrt(median_nuc_size) / 10,
            maxLineGap=np.sqrt(median_nuc_size) / 10
        )
        if lines_general_borders is not None:
            for line in lines_general_borders:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), color=255, thickness=2)

        remove_small_objects(line_image, min_size=450 if thresholds_scaler > 1 else 150, in_place=True)

        nuclei_pred[line_image > 0, 2] = 0

    _, cell_thresh = cv2.threshold(cell_pred[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nuclei_pred[cell_thresh == 0] = 0

    nuclei_label = __wsh(
        nuclei_pred[..., 2] / 255.0,
        0.4,
        1 - (nuclei_pred[..., 1] + cell_pred[..., 1]) / 255.0 > 0.05,
        nuclei_pred[..., 2] / 255,
        threshold_adjustment=-0.25,
        small_object_size_cutoff=median_nuc_size / 9 if thresholds_scaler <= 1 else median_nuc_size / 7 #(median_nuc_size / 7 if nuc_count < 49 else median_nuc_size / 4),
    )

    border_nucs_coords = get_border_nuclei(nuclei_label.max()+1, nuclei_label,
                                           border_thresh=np.sqrt(median_nuc_size)*3,
                                           bbox_area_threshold=median_nuc_size/3)
    nuclei_label = remove_small_objects(nuclei_label, 156.25 * thresholds_scaler)
    nuclei_label = measure.label(nuclei_label)
    # this is to remove the cell borders' signal from cell mask.
    # could use np.logical_and with some revision, to replace this func.
    # Tuned for segmentation hpa images
    threshold_value = max(0.22, filters.threshold_otsu(cell_pred[..., 2] / 255) * 0.5)
    # exclude the green area first
    cell_region = np.multiply(
        cell_pred[..., 2] / 255 > threshold_value,
        np.invert(np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8)),
    )
    sk = np.asarray(cell_region, dtype=np.int8)
    distance = np.clip(cell_pred[..., 2], 255 * threshold_value, cell_pred[..., 2])
    cell_label = segmentation.watershed(-distance, nuclei_label, mask=sk)

    cell_label = remove_small_objects(cell_label, 343.75 * thresholds_scaler).astype(np.uint8)
    selem = disk(6)
    cell_label = closing(cell_label, selem)
    cell_label = __fill_holes(cell_label)
    # this part is to use green channel, and extend cell label to green channel
    # benefit is to exclude cells clear on border but without nucleus
    sk = np.asarray(
        np.add(
            np.asarray(cell_label > 0, dtype=np.int8),
            np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8),
        )
        > 0,
        dtype=np.int8,
    )
    cell_label = segmentation.watershed(-distance, cell_label, mask=sk)
    cell_label = __fill_holes(cell_label)
    cell_label = np.asarray(cell_label > 0, dtype=np.uint8)
    cell_label = measure.label(cell_label)
    cell_label = remove_small_objects(cell_label, 343.75 * thresholds_scaler)
    cell_label = measure.label(cell_label)
    cell_label = np.asarray(cell_label, dtype=np.uint16)

    if return_nuclei_label:
        nuclei_label = np.multiply(cell_label > 0, nuclei_label) > 0
        nuclei_label = measure.label(nuclei_label)
        nuclei_label = remove_small_objects(nuclei_label, 156.25 * thresholds_scaler)
        nuclei_label = np.multiply(cell_label, nuclei_label > 0)

    labels_to_remove = set()
    for x, y in border_nucs_coords:
        label_to_remove = cell_label[y, x]
        if label_to_remove != 0:
            labels_to_remove.add(label_to_remove)

    cell_label = remove_labels(cell_label, labels_to_remove)
    cell_label = measure.label(np.asarray(cell_label, dtype=np.uint16))
    cell_label = np.asarray(cell_label, dtype=np.uint16)

    cell_label = cv2.resize(
        cell_label,
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )
    cell_label = cv2.dilate(cell_label, np.ones((3, 3)), iterations=2, anchor=(0, 0))
    cell_label = cv2.dilate(cell_label, np.ones((3, 3)), iterations=2, anchor=(2, 2))
    if not return_nuclei_label:
        return cell_label

    nuclei_label = cv2.resize(
        nuclei_label,
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )

    return nuclei_label, cell_label