'''
#  ----------------------------------------------------------------------
#  Executable code for mitosis detection result evaluation
#
#  ---------------------------------------------------------------------- 
#  Copyright (c) 2018 Yao Lu
#  ----------------------------------------------------------------------
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#  OTHER DEALINGS IN THE SOFTWARE.
#  ---------------------------------------------------------------------- 
#  Contact: Yao Lu at <hiluyao@tju.edu.cn> or <hiluyao@gmail.com>
#  ----------------------------------------------------------------------
#  ----------------------------------------------------------------------

Given mitosis detection results and ground truth labels in the same form of an N x 3 array (t, x, y),
this code computes the precision, recall for the detection reuslts.
'''
import os
import sys
import math
import numpy as np
import pdb


def evaluate_detection(detection_result: np.ndarray, gt_labels: np.ndarray, config=None):
    '''
    :param detection_result: detection result to be evaluated, an N x 3 array, with (t, x, y) in each row
    :param gt_labels: ground truth lables, M x 3 array, with the same format of detection_result
    :config: a dictionary containing necessary params for evaluation
    :return: precision, recall
    '''
    xy_dist_threshold_default = 15
    t_dist_threshold_default = 6
    return_detection_result = False
    if config is not None:
        if 'xy_dist_threshold' not in config.keys():
            xy_dist_threshold = xy_dist_threshold_default
        else:
            xy_dist_threshold = config['xy_dist_threshold']
        if 't_dist_threshold' not in config.keys():
            t_dist_threshold = t_dist_threshold_default
        else:
            t_dist_threshold = config['t_dist_threshold']
        if 'return_detection_result' in config.keys():
            return_detection_result = config['return_detection_result']
    else:
        xy_dist_threshold = xy_dist_threshold_default
        t_dist_threshold = t_dist_threshold_default
        pass
    # check input format
    if len(detection_result.shape) != 2:
        raise TypeError('Invalid input shape: detection_result: ' + str(detection_result.shape))
    if detection_result.shape[1] != 3:
        if detection_result.shape[1] == 4:
            seq_id_list = detection_result[:, 3]
            detection_result = detection_result[:, 0:3]
            pass
        else:
            raise TypeError('Invalid input shape: detection_result: ' + str(detection_result.shape))
    if len(gt_labels.shape) != 2 or gt_labels.shape[1] != 3:
        raise TypeError('Invalid input shape: gt_labels: ' + str(gt_labels.shape))

    # add additional column to mark detection status in gt_labels and detection_result, [t, x, y, status_id, pair_id]
    # status id: 0 as False Positive, 1 as True Positive, 2 as False Negative
    gt_labels = np.concatenate((gt_labels, np.zeros((gt_labels.shape[0], 2))), axis=1)
    detection_result = np.concatenate((detection_result, np.zeros((detection_result.shape[0], 2))), axis=1)
    # pdb.set_trace()
    # search for nearest ground truth labels for each detection result coordinate
    for det_idx, detection_coord in enumerate(detection_result):
        det_gt_dist = []
        det_x = detection_coord[1]
        det_y = detection_coord[2]
        det_t = detection_coord[0]
        for gt_idx, gt_coord in enumerate(gt_labels):
            gt_x = gt_coord[1]
            gt_y = gt_coord[2]
            gt_t = gt_coord[0]
            xy_dist = math.sqrt((det_x - gt_x) ** 2 + (det_y - gt_y) ** 2)
            t_dist = abs(det_t - gt_t)
            det_gt_dist.append({'xy_dist': xy_dist, 't_dist': t_dist, 'gt_idx': gt_idx})
            pass
        # find the nearest gt label for the current detection
        det_gt_dist.sort(key=lambda x: x['xy_dist'])
        det_gt_dist_filter = [_ for _ in det_gt_dist if abs(_['xy_dist']) < xy_dist_threshold]
        if len(det_gt_dist_filter) == 0:
            # mark current detection as False Positive
            detection_result[det_idx, 3] = 0
            detection_result[det_idx, 4] = 0
            pass
        else:
            # sort by t_dist
            det_gt_dist_filter.sort(key=lambda x: abs(x['t_dist']))
            if det_gt_dist_filter[0]['t_dist'] > t_dist_threshold:
                # mark current detection as False Positive
                detection_result[det_idx, 3] = 0
                detection_result[det_idx, 4] = 0
                pass
            else:
                gt_idx = det_gt_dist_filter[0]['gt_idx']
                gt_idx = int(gt_idx)
                if gt_labels[gt_idx, 3] > 0:
                    # compare distance with previous chosen detection
                    gt_x = gt_labels[gt_idx, 1]
                    gt_y = gt_labels[gt_idx, 2]
                    gt_t = gt_labels[gt_idx, 0]

                    det_idx_pre = gt_labels[gt_idx, 4]
                    det_idx_pre = int(det_idx_pre)
                    det_x_pre = detection_result[det_idx_pre, 1]
                    det_y_pre = detection_result[det_idx_pre, 2]
                    det_t_pre = detection_result[det_idx_pre, 0]
                    xy_dist_pre = math.sqrt((det_x_pre - gt_x) ** 2 + (det_y_pre - gt_y) ** 2)
                    t_dist_pre = abs(det_t_pre - gt_t)

                    det_x_cur = detection_result[det_idx, 1]
                    det_y_cur = detection_result[det_idx, 2]
                    det_t_cur = detection_result[det_idx, 0]
                    xy_dist_cur = math.sqrt((det_x_cur - gt_x) ** 2 + (det_y_cur - gt_y) ** 2)
                    t_dist_cur = abs(det_t_cur - gt_t)
                    if xy_dist_pre <= xy_dist_cur:
                        # mark current detection as False Positive
                        detection_result[det_idx, 3] = 0
                        detection_result[det_idx, 4] = 0
                        pass
                    else:
                        # mark current detection as True Positive
                        detection_result[det_idx, 3] = 1
                        detection_result[det_idx, 4] = gt_idx
                        gt_labels[gt_idx, 3] = 2
                        gt_labels[gt_idx, 4] = det_idx
                        # mark previous detection as False Positive
                        detection_result[det_idx_pre, 3] = 0
                        detection_result[det_idx_pre, 4] = 0
                        pass
                    pass
                else:
                    # mark current detection as True Positive
                    # mark the corresponding gt_label as detected
                    gt_labels[gt_idx, 3] = 2
                    gt_labels[gt_idx, 4] = det_idx
                    detection_result[det_idx, 3] = 1
                    detection_result[det_idx, 4] = gt_idx
                pass
            pass
        pass
    tp_list = np.argwhere(detection_result[:, 3] > 0)
    
    tp = float(len(tp_list))
    fp = detection_result.shape[0] - tp
    fn = gt_labels.shape[0] - tp

    # precision = float(len(tp_list)) / detection_result.shape[0]
    # recall = float(len(tp_list)) / gt_labels.shape[0]
    # detection_result = np.concatenate((detection_result[:, 0:3],
    #                                    seq_id_list[:, np.newaxis],
    #                                    detection_result[:, 3][:, np.newaxis]),
    #                                   axis=1)
    detection_result = np.concatenate((detection_result[:, 0:3], detection_result[:, 3][:, np.newaxis]), axis=1)
    return  float(len(tp_list)), fp,  fn, detection_result, gt_labels
    pass
