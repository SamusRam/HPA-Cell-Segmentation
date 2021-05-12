cimport numpy as np
cimport numpy as cnp
ctypedef np.uint8_t uint8
ctypedef np.int32_t int32
from libc.stdlib cimport malloc
from libc.stdlib cimport set as c_set
from collections import defaultdict

cnp.import_array()
cdef int[:,:] return_empty_2d_int(int dim1, int dim2):
    cdef cnp.npy_intp* dims = [dim1, dim2]
    return cnp.PyArray_SimpleNew(2, dims, cnp.NPY_INT)

cdef int[:] return_empty_1d_int(int dim):
    cdef cnp.npy_intp* dims = [dim]
    return cnp.PyArray_SimpleNew(1, dims, cnp.NPY_INT)

cdef int[:,:,:] return_empty_3d_int(int dim1, int dim2, int dim3):
    cdef cnp.npy_intp* dims = [dim1, dim2, dim3]
    return cnp.PyArray_SimpleNew(3, dims, cnp.NPY_INT)

cdef extern from "limits.h":
    cdef int INT_MAX, INT_MIN

def contains_nucleus_with_stats(int[:, :] all_cc_labeled, int cc_label, uint8[:, :] nuclei_mask, int border_threshold):
    cdef int cc_pixels_counts = 0
    cdef int cc_x_coord_sum = 0
    cdef int cc_y_coord_sum = 0
    cdef int row_i, col_j
    cdef int N = all_cc_labeled.shape[0]

    for row_i in range(N):
        for col_j in range(N):
            # check if it the cc of interest
            if all_cc_labeled[row_i, col_j] == cc_label:
                if nuclei_mask[row_i, col_j]:
                    return True, -1, -1
                else:
                    if row_i < border_threshold or row_i > N - 1 - border_threshold or col_j < border_threshold or col_j > N - 1 - border_threshold:
                        cc_x_coord_sum += row_i
                        cc_y_coord_sum += col_j
                        cc_pixels_counts += 1
    if cc_pixels_counts == 0:
        return True, -1, -1
    centroid_x = cc_x_coord_sum / cc_pixels_counts
    centroid_y = cc_y_coord_sum / cc_pixels_counts
    return False, centroid_x, centroid_y



def get_additional_markers(int num_cc_labels, int[:, :] all_cc_labeled, uint8[:, :] nuclei_mask, int border_threshold,
                           int32[:] cc_sizes, int resized_small_img_size, int full_img_size, float thresholds_scale):
    cdef int cc_label
    cdef int num_markers = 0
    cdef int[:,:] markers = return_empty_2d_int(num_cc_labels, 2)

    for cc_label in range(1, num_cc_labels):
        contains_nuc, centroid_x, centroid_y = contains_nucleus_with_stats(all_cc_labeled, cc_label, nuclei_mask, 2*border_threshold)
        if not contains_nuc:
            # if the area is too small, then it must be on the border
            if (cc_sizes[cc_label] > 10*thresholds_scale or
                min(centroid_x, centroid_y) < border_threshold or
                max(centroid_x, centroid_y) > resized_small_img_size - 1 - border_threshold):

                centroid_scaled_y = int(centroid_x * full_img_size / resized_small_img_size)
                centroid_scaled_x = int(centroid_y * full_img_size / resized_small_img_size)
                markers[num_markers, 0] = centroid_scaled_x
                markers[num_markers, 1] = centroid_scaled_y
                num_markers += 1
    return markers[:num_markers, :]


cdef int[:,:] get_poles_of_nuclei(int[:, :] nuclei_labeled_mask, int num_cc_labels):
    cdef int[:,:] nuclei_poles = return_empty_2d_int(num_cc_labels, 4)
    cdef int nuc_i, row_i, col_j, label_nuc
    cdef int N = nuclei_labeled_mask.shape[0]

    # init nuclei_poles
    for nuc_i in range(1, num_cc_labels):
        # x_min
        nuclei_poles[nuc_i, 0] = INT_MAX
        # x_max
        nuclei_poles[nuc_i, 1] = INT_MIN
        # y_min
        nuclei_poles[nuc_i, 2] = INT_MAX
        # y_max
        nuclei_poles[nuc_i, 3] = INT_MIN

    for row_i in range(N):
        for col_j in range(N):
            if nuclei_labeled_mask[row_i, col_j] != 0:
                label_nuc = nuclei_labeled_mask[row_i, col_j]
                if row_i < nuclei_poles[label_nuc, 0]:
                    nuclei_poles[label_nuc, 0] = row_i
                if row_i > nuclei_poles[label_nuc, 1]:
                    nuclei_poles[label_nuc, 1] = row_i
                if col_j < nuclei_poles[label_nuc, 2]:
                    nuclei_poles[label_nuc, 2] = col_j
                if col_j > nuclei_poles[label_nuc, 3]:
                    nuclei_poles[label_nuc, 3] = col_j
    return nuclei_poles

def get_border_nuclei(int num_cc_labels_nucs, int[:, :] nuclei_labeled_mask, int border_thresh, float bbox_area_threshold):
    cdef int cc_label_nuc
    cdef int num_border_nucs = 0
    cdef int[:,:] border_nucs_coords = return_empty_2d_int(num_cc_labels_nucs, 2)

    nuclei_poles = get_poles_of_nuclei(nuclei_labeled_mask, num_cc_labels_nucs)
    img_size = nuclei_labeled_mask.shape[0]
    for cc_label_nuc in range(1, num_cc_labels_nucs):
        x_min, x_max, y_min, y_max = nuclei_poles[cc_label_nuc, :]
        if bbox_area_threshold > (x_max - x_min) * (y_max - y_min) and (min(x_max, y_max) < border_thresh or max(x_min, y_min) > img_size - 1 - border_thresh) and (min(x_min, y_min) <= 1 or max(x_max, y_max) >= img_size - 2):
            center_scaled_y = (x_max + x_min)/2
            center_scaled_x = (y_max + y_min)/2
            border_nucs_coords[num_border_nucs, 0] = center_scaled_x
            border_nucs_coords[num_border_nucs, 1] = center_scaled_y
            num_border_nucs += 1
    return border_nucs_coords[:num_border_nucs, :]


def filter_single_nuc_borders(unsigned char[:, :] nuclei_borders_nuc_overlap, int[:, :] nuclei_borders_cc,
                              int num_nuclei_borders_cc, int[:, :] nuclei_cc, int zero_intensity_threshold, int border_size_threshold):
    cdef int N = nuclei_borders_nuc_overlap.shape[0]
    cdef int[:,:] single_nuc_borders = return_empty_2d_int(N, N)
    single_nuc_borders[:, :] = 0
    cdef int i, j, nuc_border_label, nuclei_label
    cdef uint8 border_nuc_value

    # nuc_border_cc ... corresponds to background, is there for straightforward indexing as it costs nothing
    cdef int[:] nuc_border_cc_purity = return_empty_1d_int(num_nuclei_borders_cc)
    nuc_border_cc_purity[:] = 0

    cdef int[:] nuc_border_cc_sizes = return_empty_1d_int(num_nuclei_borders_cc)
    nuc_border_cc_sizes[:] = 0
    nuclei_2_borders = defaultdict(set)

    for i in range(N):
        for j in range(N):
            nuc_border_label = nuclei_borders_cc[i, j]
            if nuc_border_label != 0:
                nuc_border_cc_sizes[nuc_border_label] += 1

                nuclei_label = nuclei_cc[i, j]
                # we're on top of the nucleus
                if nuclei_label != 0:
                    # first encounter

                    if nuc_border_cc_purity[nuc_border_label] == 0:
                        nuc_border_cc_purity[nuc_border_label] = nuclei_label
                    else:
                        if nuc_border_cc_purity[nuc_border_label] != nuclei_label:
                            nuc_border_cc_purity[nuc_border_label] = -1


    for nuc_border_label in range(num_nuclei_borders_cc):
        nuclei_label = nuc_border_cc_purity[nuc_border_label]
        nuclei_2_borders[nuclei_label].add(nuc_border_label)

    nuc_label_2_output_idx = dict()
    cdef int num_outputs = 0
    for i in range(1, num_nuclei_borders_cc):
        nuclei_label = nuc_border_cc_purity[i]
        if nuclei_label > 0 and len(nuclei_2_borders[nuclei_label]) >= 2 and nuclei_label not in nuc_label_2_output_idx and nuc_border_cc_sizes[i] >= border_size_threshold:
            nuc_label_2_output_idx[nuclei_label] = num_outputs
            num_outputs += 1

    outputs = return_empty_3d_int(num_outputs, N, N)
    outputs[:,:,:] = 0
    general_borders = return_empty_2d_int(N, N)
    general_borders[:,:] = 0
    # copy overlaps between borders and nuclei for the selected borders
    for i in range(N):
        for j in range(N):
            border_nuc_value = int(nuclei_borders_nuc_overlap[i, j])
            if border_nuc_value > zero_intensity_threshold:
                nuc_border_label = nuclei_borders_cc[i, j]
                if nuc_border_label > 0:
                    nuclei_label = nuc_border_cc_purity[nuc_border_label]
                    if nuclei_label in nuc_label_2_output_idx:
                        output_idx = nuc_label_2_output_idx[nuclei_label]
                        outputs[output_idx, i, j] = nuclei_label
                    else:
                        general_borders[i, j] = 1

    return outputs, general_borders

def remove_labels(unsigned short[:, :] nuclei_labeled_mask, set labels_to_remove):
    cdef int row_i, col_j
    cdef int N = nuclei_labeled_mask.shape[0]
    
    for row_i in range(N):
        for col_j in range(N):
            if nuclei_labeled_mask[row_i, col_j] in labels_to_remove:
                nuclei_labeled_mask[row_i, col_j] = 0
                
    return nuclei_labeled_mask