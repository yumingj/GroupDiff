import copy
import glob
import json
import math
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from torch import Tensor
from torchvision.ops import masks_to_boxes

from openpose import OpenposeDetector


class GroupDiffDataGen(data.Dataset):

    def __init__(self,
                 state,
                 skeleton_path_prefix,
                 add_harmonization=False,
                 reposing_exemplar=True,
                 use_localssd=False):
        self.state = state
        self.add_harmonization = add_harmonization
        self.use_localssd = use_localssd

        if state == 'train':
            data_dir = './LV-MHP-v2/train'
            self.data_path_list = glob.glob(f'{data_dir}/images/*.jpg')
            self.parsing_dir = f'{data_dir}/parsing_annos'
            self.pose_estimation_path = f'{data_dir}/pose_estimation'
            self.data_path_list.sort()
        else:
            data_dir = './LV-MHP-v2/val'
            self.data_path_list = glob.glob(f'{data_dir}/images/*.jpg')
            self.parsing_dir = f'{data_dir}/parsing_annos'
            self.data_path_list.sort()

        self.skeleton_path_prefix = skeleton_path_prefix

        self.resize_transform_img = transforms.Resize(size=512)
        self.resize_transform_mask = transforms.Resize(
            size=512, interpolation=transforms.InterpolationMode.NEAREST)

        self.resize_transform_exemplar = transforms.Resize(size=224)

        self.apply_openpose = OpenposeDetector()

        self.reposing_exemplar = reposing_exemplar

        self.random_color_identity_group = [[(0, 0, 255), (0, 0, 200),
                                             (0, 0, 150)],
                                            [(255, 0, 0), (200, 0, 0),
                                             (150, 0, 0)],
                                            [(0, 255, 0), (0, 200, 0),
                                             (0, 200, 0)],
                                            [(255, 0, 255), (200, 0, 200),
                                             (150, 0, 150)],
                                            [(0, 255, 255), (0, 200, 200),
                                             (0, 150, 150)]]

    def transform_exemplar(self):
        transform_list = []
        transform_list += [
            transforms.RandomAffine(
                degrees=20,
                translate=(0.1, 0.1),
                scale=(0.9, 1.10),
                fill=255,
                interpolation=transforms.InterpolationMode.BILINEAR)
        ]
        if self.add_harmonization:
            transform_list += [
                transforms.ColorJitter(
                    brightness=(0.9, 1.1),
                    contrast=(0.9, 1.1),
                    saturation=(0.8, 1.3))
            ]
        transform_list += [transforms.Resize(size=512)]
        # transform_list += [transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
        #                                         (0.26862954, 0.26130258, 0.27577711))]

        return transforms.Compose(transform_list)

    def get_candidate_parsing_list_for_exemplar(self, inpaint_mask,
                                                seleted_idx,
                                                instance_parsing_list):
        candidate_parsing_list = []
        idx_in_candidate_list = 0
        count = 0
        for idx, instance_parsing in enumerate(instance_parsing_list):
            mask_binary = np.zeros(
                (inpaint_mask.shape[0], inpaint_mask.shape[1]), dtype=np.uint8)
            mask_binary[instance_parsing > 0] = 1

            if np.sum(mask_binary * inpaint_mask) == 0:
                continue

            candidate_parsing_list.append(instance_parsing)

            if idx == seleted_idx:
                idx_in_candidate_list = count

            count += 1

        return candidate_parsing_list, idx_in_candidate_list



    def warp_parsing(self, parsing, rect1, rect2):
        shape1 = parsing.shape
        h = shape1[0]
        w = shape1[1]

        rect1 = np.array(rect1, dtype=np.float32)
        rect2 = np.array(rect2, dtype=np.float32)

        # ===== homography
        H = cv2.getPerspectiveTransform(src=rect1, dst=rect2)
        # print(H)
        # H_inverse = np.linalg.inv(H)

        # img_warped = cv2.warpPerspective(src=img, M=H_inverse, dsize=(w, h))
        parsing_warped = cv2.warpPerspective(
            src=parsing, M=H, dsize=(w, h), flags=cv2.INTER_NEAREST)

        return parsing_warped

    def rotate_whole_arms(self, ori_point, point_a, point_b, alpha):
        x_0, y_0 = ori_point
        x_a, y_a = point_a
        x_b, y_b = point_b

        x_a = x_a - x_0
        y_a = y_a - y_0

        x_b = x_b - x_0
        y_b = y_b - y_0

        x_a_prime = x_a * math.cos(alpha) - y_a * math.sin(alpha)
        y_a_prime = x_a * math.sin(alpha) + y_a * math.cos(alpha)

        x_b_dif = x_b - x_a
        y_b_dif = y_b - y_a

        x_b_prime = x_b_dif * math.cos(alpha) - y_b_dif * math.sin(
            alpha) + x_a_prime + x_0
        y_b_prime = x_b_dif * math.sin(alpha) + y_b_dif * math.cos(
            alpha) + y_a_prime + y_0

        return [x_a_prime + x_0, y_a_prime + y_0], [x_b_prime, y_b_prime]

    def rotate_part_arms(self, ori_point, point_a, alpha):
        x_0, y_0 = ori_point
        x_a, y_a = point_a

        x_a = x_a - x_0
        y_a = y_a - y_0

        x_a_prime = x_a * math.cos(alpha) - y_a * math.sin(alpha)
        y_a_prime = x_a * math.sin(alpha) + y_a * math.cos(alpha)

        return [x_a_prime + x_0, y_a_prime + y_0]

    def randomly_change_pose(self, ori_coordinates, selected_person_idx):
        new_coordinates = copy.deepcopy(ori_coordinates)
        candidate = ori_coordinates['candidate']
        subset = ori_coordinates['subset']

        augmentation_type = random.uniform(0, 1)
        try:
            index_2 = int(subset[selected_person_idx][2])
            index_3 = int(subset[selected_person_idx][3])
            index_4 = int(subset[selected_person_idx][4])
            index_5 = int(subset[selected_person_idx][5])
            index_6 = int(subset[selected_person_idx][6])
            index_7 = int(subset[selected_person_idx][7])
        except:
            return new_coordinates

        if (index_2 == -1 or index_3 == -1
                or index_4 == -1) and (index_3 == -1 or index_4 == -1) and (
                    index_5 == -1 or index_6 == -1
                    or index_7 == -1) and (index_6 == -1 or index_7 == -1):
            return new_coordinates

        augmentation_type = random.uniform(0, 1)
        trial_num = 0
        while (trial_num < 5):
            if augmentation_type < 0.25:
                if index_2 == -1 or index_3 == -1 or index_4 == -1:
                    trial_num += 1
                    augmentation_type = random.uniform(0, 1)
                    continue
                # left arms
                # change from the body_idx 2
                changed_x3, changed_x4 = self.rotate_whole_arms(
                    candidate[int(subset[selected_person_idx][2])][0:2],
                    candidate[int(subset[selected_person_idx][3])][0:2],
                    candidate[int(subset[selected_person_idx][4])][0:2],
                    2 * math.pi * random.random())

                new_coordinates['candidate'][int(
                    subset[selected_person_idx][3])][0:2] = changed_x3
                new_coordinates['candidate'][int(
                    subset[selected_person_idx][4])][0:2] = changed_x4
            elif augmentation_type < 0.5:
                # left arms
                # change from the body_idx 3
                if index_3 == -1 or index_4 == -1:
                    trial_num += 1
                    augmentation_type = random.uniform(0, 1)
                    continue
                changed_x4 = self.rotate_part_arms(
                    candidate[int(subset[selected_person_idx][3])][0:2],
                    candidate[int(subset[selected_person_idx][4])][0:2],
                    2 * math.pi * random.random())
                new_coordinates['candidate'][int(
                    subset[selected_person_idx][4])][0:2] = changed_x4
            elif augmentation_type < 0.75:
                # right arms
                # change from the body_idx 5
                if index_5 == -1 or index_6 == -1 or index_7 == -1:
                    trial_num += 1
                    augmentation_type = random.uniform(0, 1)
                    continue
                changed_x6, changed_x7 = self.rotate_whole_arms(
                    candidate[int(subset[selected_person_idx][5])][0:2],
                    candidate[int(subset[selected_person_idx][6])][0:2],
                    candidate[int(subset[selected_person_idx][7])][0:2],
                    2 * math.pi * random.random())

                new_coordinates['candidate'][int(
                    subset[selected_person_idx][6])][0:2] = changed_x6
                new_coordinates['candidate'][int(
                    subset[selected_person_idx][7])][0:2] = changed_x7
            else:
                # right arms
                # change from the body_idx 5
                if index_6 == -1 or index_7 == -1:
                    trial_num += 1
                    augmentation_type = random.uniform(0, 1)
                    continue
                changed_x7 = self.rotate_part_arms(
                    candidate[int(subset[selected_person_idx][6])][0:2],
                    candidate[int(subset[selected_person_idx][7])][0:2],
                    2 * math.pi * random.random())
                new_coordinates['candidate'][int(
                    subset[selected_person_idx][7])][0:2] = changed_x7

            break

        return new_coordinates


    def reposing_exemplar_img(self, exemplar_img, parsing_map):
        _, ori_coordinates = self.apply_openpose(exemplar_img)

        if self.reposing_exemplar:
            selected_person_idx = 0
            new_coordinates = self.randomly_change_pose(
                ori_coordinates, selected_person_idx)

            connected_line_list = [[2, 3], [3, 4], [5, 6], [6, 7]]

            new_exemplar_img = exemplar_img.copy()
            for connected_line in connected_line_list:
                try:
                    index = int(
                        ori_coordinates['subset'][0][connected_line[0]])
                except:
                    continue
                if index == -1:
                    continue
                point1 = ori_coordinates['candidate'][index][0:2]

                try:
                    index = int(
                        ori_coordinates['subset'][0][connected_line[1]])
                except:
                    continue
                if index == -1:
                    continue
                point2 = ori_coordinates['candidate'][index][0:2]

                try:
                    index = int(
                        new_coordinates['subset'][0][connected_line[0]])
                except:
                    continue
                if index == -1:
                    continue
                new_point1 = new_coordinates['candidate'][index][0:2]

                try:
                    index = int(
                        new_coordinates['subset'][0][connected_line[1]])
                except:
                    continue

                if index == -1:
                    continue
                new_point2 = new_coordinates['candidate'][index][0:2]

                if (point1 == new_point1) and (point2 == new_point2):
                    continue

                # if the arm, extend the point2
                if (connected_line == [3, 4]) or (connected_line == [6, 7]):
                    # import pdb
                    # pdb.set_trace()
                    point2[0] = point2[0] + 0.6 * (point2[0] - point1[0])
                    point2[1] = point2[1] + 0.6 * (point2[1] - point1[1])

                length = ((point1[0] - point2[0])**2 +
                          (point1[1] - point2[1])**2)**0.5

                ori_rec_points = self.find_parallel_points(
                    point1, point2, 0.25 * length)

                # if the arm, extend the point2
                if (connected_line == [3, 4]) or (connected_line == [6, 7]):
                    # import pdb
                    # pdb.set_trace()
                    new_point2[0] = new_point2[0] + 0.6 * (
                        new_point2[0] - new_point1[0])
                    new_point2[1] = new_point2[1] + 0.6 * (
                        new_point2[1] - new_point1[1])

                length = ((new_point1[0] - new_point2[0])**2 +
                          (new_point1[1] - new_point2[1])**2)**0.5
                new_rec_points = self.find_parallel_points(
                    new_point1, new_point2, 0.25 * length)

                warped_exemplar = self.warp_img(exemplar_img, ori_rec_points,
                                                new_rec_points)

                masked_area = np.zeros_like(exemplar_img[:, :, 0])
                cv2.fillPoly(masked_area, [np.array(ori_rec_points)], 255)
                masked_area = masked_area * (parsing_map > 0)

                new_exemplar_img[masked_area == 255] = 255

                warped_parsing = self.warp_parsing(parsing_map, ori_rec_points,
                                                   new_rec_points)

                masked_area = np.zeros_like(exemplar_img[:, :, 0])
                cv2.fillPoly(masked_area, [np.array(new_rec_points)], 255)
                masked_area = masked_area * (warped_parsing > 0)

                new_exemplar_img[masked_area == 255] = warped_exemplar[
                    masked_area == 255]

            return new_exemplar_img, new_coordinates
        else:
            return exemplar_img, ori_coordinates

    def warp_img(self, img, rect1, rect2):
        shape1 = img.shape
        h = shape1[0]
        w = shape1[1]

        rect1 = np.array(rect1, dtype=np.float32)
        rect2 = np.array(rect2, dtype=np.float32)

        # ===== homography
        H = cv2.getPerspectiveTransform(src=rect1, dst=rect2)
        # print(H)
        # H_inverse = np.linalg.inv(H)

        # img_warped = cv2.warpPerspective(src=img, M=H_inverse, dsize=(w, h))
        img_warped = cv2.warpPerspective(src=img, M=H, dsize=(w, h))

        return img_warped

    def find_parallel_points(self, point1, point2, distance):
        # Calculate slope and intercept of the line passing through the two points
        # slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
        # intercept = point1[1] - slope * point1[0]

        # Calculate the angle of the line
        angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])

        # Calculate new points parallel to the line
        parallel_points = []
        for direction in [-1,
                          1]:  # Two directions (left and right of the line)
            new_x = point1[0] + direction * distance * np.sin(angle)
            new_y = point1[1] - direction * distance * np.cos(angle)
            parallel_points.append((int(new_x), int(new_y)))

        for direction in [1,
                          -1]:  # Two directions (left and right of the line)
            new_x = point2[0] + direction * distance * np.sin(angle)
            new_y = point2[1] - direction * distance * np.cos(angle)
            parallel_points.append((int(new_x), int(new_y)))

        return parallel_points

    def read_img(self, img_path):

        img = np.array(Image.open(img_path).convert('RGB'))

        return img

    def read_img_exemplar_mask(self, img, candidate_parsing_list):

        img_exemplar_list = []
        parsing_exemplar_list = []
        for parsing in candidate_parsing_list:
            mask_binary = np.zeros((img.shape[0], img.shape[1]),
                                   dtype=np.uint8)
            mask_binary[parsing > 0] = 1

            img_exemplar = img.copy()

            img_exemplar[mask_binary == 0] = 255.
            inner_dilated_aug = random.uniform(0, 1)
            if inner_dilated_aug < 0.2:
                structuring_element = np.ones((5, 5), dtype=bool)
                dilated_mask_binary = binary_dilation(
                    1 - mask_binary, structure=structuring_element)
                img_exemplar[dilated_mask_binary == 1] = 255.

            mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

            obj_ids = torch.unique(mask_tensor)
            obj_ids = obj_ids[1:]
            masks = mask_tensor == obj_ids[:, None, None]

            boxes = masks_to_boxes(masks)

            h, w = mask_binary.shape

            # make the bounding box slightly larger
            enlarge_ratio = 0.1
            enlarge_margin_h = int((boxes[0][3] - boxes[0][1]) * enlarge_ratio)
            enlarge_margin_w = int((boxes[0][2] - boxes[0][0]) * enlarge_ratio)

            bbox_y1, bbox_y2 = max(0,
                                   int(boxes[0][1]) - enlarge_margin_h), min(
                                       h,
                                       int(boxes[0][3]) + enlarge_margin_h)
            bbox_x1, bbox_x2 = max(0,
                                   int(boxes[0][0]) - enlarge_margin_w), min(
                                       w,
                                       int(boxes[0][2]) + enlarge_margin_w)
            img_exemplar = img_exemplar[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
            img_exemplar_list.append(img_exemplar)
            parsing_exemplar_list.append(parsing[bbox_y1:bbox_y2,
                                                 bbox_x1:bbox_x2])

        return img_exemplar_list, parsing_exemplar_list

    def transform_exemplar_and_parsing(self, exemplar_img, parsing):

        random_affine_transformation = transforms.RandomAffine(
            degrees=20,
            translate=(0.1, 0.1),
            scale=(0.9, 1.10),
            fill=255,
            interpolation=transforms.InterpolationMode.BILINEAR)
        resize_transform_img = transforms.Resize(size=512)
        resize_transform_parsing = transforms.Resize(
            size=512, interpolation=transforms.InterpolationMode.NEAREST)

        channels, height, width = exemplar_img.size()

        ret = random_affine_transformation.get_params(
            random_affine_transformation.degrees,
            random_affine_transformation.translate,
            random_affine_transformation.scale,
            random_affine_transformation.shear, [width, height])

        fill = 255
        if isinstance(exemplar_img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        exemplar_img = F.affine(
            exemplar_img,
            *ret,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=fill,
            center=random_affine_transformation.center)

        channels, _, _ = parsing.size()
        fill = 0
        if isinstance(parsing, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        parsing = F.affine(
            parsing,
            *ret,
            interpolation=transforms.InterpolationMode.NEAREST,
            fill=fill,
            center=random_affine_transformation.center)

        exemplar_img = resize_transform_img(exemplar_img)
        parsing = resize_transform_parsing(parsing)

        return exemplar_img, parsing

    def random_brush_top_down(self, skeleton_mask, ori_rec_points):
        mask = Image.new('L', (skeleton_mask.shape[1], skeleton_mask.shape[0]), 0)

        num_points = int(np.random.uniform(8, 15))

        sampled_points_top = np.linspace(ori_rec_points[0], ori_rec_points[1], num_points)
        sampled_points_top = [(int(x), int(y)) for x, y in sampled_points_top]

        sampled_points_down = np.linspace(ori_rec_points[3], ori_rec_points[2], num_points)
        sampled_points_down = [(int(x), int(y)) for x, y in sampled_points_down]

        vertex = []
        for top_point, down_point in zip(sampled_points_top, sampled_points_down):
            random_move = np.random.uniform(-0.6, 0.6)
            sampled_x, sampled_y = top_point
            sampled_x = sampled_x + int(random_move * (sampled_points_top[1][0] - sampled_points_top[0][0]))
            sampled_y = sampled_y - int(np.random.uniform(0, 1.0) * (sampled_points_down[1][1] - sampled_points_down[0][1]))
            vertex.append((sampled_x, sampled_y))

            sampled_x, sampled_y = down_point
            random_move = np.random.uniform(-0.6, 0.6)
            sampled_x = sampled_x + int(random_move * (sampled_points_top[1][0] - sampled_points_top[0][0]))
            sampled_y = sampled_y + int(np.random.uniform(0, 1.0) * (sampled_points_down[1][1] - sampled_points_down[0][1]))
            vertex.append((sampled_x, sampled_y))

        draw = ImageDraw.Draw(mask)
        min_width = 12
        max_width = 48
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)

        mask = np.asarray(mask, np.uint8) * 255

        return mask

    def load_arm_hand_masks(self, skeleton_mask, selected_person_bbox,
                            instance_parsing_list):
        area_list = []
        for instance_parsing in instance_parsing_list:
            mask_binary = np.zeros((instance_parsing.shape[0], instance_parsing.shape[1]), dtype=np.uint8)
            mask_binary[instance_parsing > 0] = 1
            area = np.sum(selected_person_bbox * mask_binary)
            area_list.append(area)

        seleted_idx = np.argmax(area_list)

        selected_parsing = instance_parsing_list[seleted_idx]

        temp_mask = np.zeros_like(selected_parsing)
        for value in [5, 7]:
            temp_mask[selected_parsing == value] = 1
        if np.sum(temp_mask) != 0:
            kernel_width = 28
            kernel_height = 45

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))
            dilated_mask = cv2.dilate(temp_mask, kernel)

            skeleton_mask[skeleton_mask == 0] = dilated_mask[skeleton_mask == 0]

        temp_mask = np.zeros_like(selected_parsing)
        for value in [6, 8]:
            temp_mask[selected_parsing == value] = 1
        if np.sum(temp_mask) != 0:
            kernel_width = 28
            kernel_height = 45

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))
            dilated_mask = cv2.dilate(temp_mask, kernel)

            skeleton_mask[skeleton_mask == 0] = dilated_mask[skeleton_mask == 0]

        return skeleton_mask, seleted_idx

    def random_brush_down_top(self, skeleton_mask, ori_rec_points):
        mask = Image.new('L', (skeleton_mask.shape[1], skeleton_mask.shape[0]), 0)

        num_points = int(np.random.uniform(8, 15))

        sampled_points_top = np.linspace(ori_rec_points[0], ori_rec_points[1], num_points)
        sampled_points_top = [(int(x), int(y)) for x, y in sampled_points_top]

        sampled_points_down = np.linspace(ori_rec_points[3], ori_rec_points[2], num_points)
        sampled_points_down = [(int(x), int(y)) for x, y in sampled_points_down]

        vertex = []
        for top_point, down_point in zip(sampled_points_down, sampled_points_top):
            random_move = np.random.uniform(-0.6, 0.6)
            sampled_x, sampled_y = top_point
            sampled_x = sampled_x + int(random_move * (sampled_points_top[1][0] - sampled_points_top[0][0]))
            sampled_y = sampled_y - int(np.random.uniform(0, 1.0) * (sampled_points_down[1][1] - sampled_points_down[0][1]))
            vertex.append((sampled_x, sampled_y))

            sampled_x, sampled_y = down_point
            random_move = np.random.uniform(-0.6, 0.6)
            sampled_x = sampled_x + int(random_move * (sampled_points_top[1][0] - sampled_points_top[0][0]))
            sampled_y = sampled_y + int(np.random.uniform(0, 1.0) * (sampled_points_down[1][1] - sampled_points_down[0][1]))
            vertex.append((sampled_x, sampled_y))

        draw = ImageDraw.Draw(mask)
        min_width = 12
        max_width = 48
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)

        mask = np.asarray(mask, np.uint8) * 255

        # import pdb
        # pdb.set_trace()

        return mask

    def random_brush_left_right(self, skeleton_mask, ori_rec_points):
        mask = Image.new('L', (skeleton_mask.shape[1], skeleton_mask.shape[0]), 0)

        num_points = int(np.random.uniform(8, 15))

        sampled_points_top = np.linspace(ori_rec_points[3], ori_rec_points[0], num_points)
        sampled_points_top = [(int(x), int(y)) for x, y in sampled_points_top]

        sampled_points_down = np.linspace(ori_rec_points[2], ori_rec_points[1], num_points)
        sampled_points_down = [(int(x), int(y)) for x, y in sampled_points_down]

        vertex = []
        for top_point, down_point in zip(sampled_points_down, sampled_points_top):
            random_move = np.random.uniform(-0.6, 0.6)
            sampled_x, sampled_y = top_point
            sampled_x = sampled_x - int(np.random.uniform(0, 1.0) * (sampled_points_top[1][0] - sampled_points_top[0][0]))
            sampled_y = sampled_y + int(random_move * (sampled_points_down[1][1] - sampled_points_down[0][1]))
            vertex.append((sampled_x, sampled_y))

            sampled_x, sampled_y = down_point
            random_move = np.random.uniform(-0.6, 0.6)
            sampled_x = sampled_x + int(np.random.uniform(0, 1.0) * (sampled_points_top[1][0] - sampled_points_top[0][0]))
            sampled_y = sampled_y + int(random_move * (sampled_points_down[1][1] - sampled_points_down[0][1]))
            vertex.append((sampled_x, sampled_y))

        draw = ImageDraw.Draw(mask)
        min_width = 12
        max_width = 48
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)

        mask = np.asarray(mask, np.uint8) * 255

        # import pdb
        # pdb.set_trace()

        return mask

    def random_brush_right_left(self, skeleton_mask, ori_rec_points):
        mask = Image.new('L', (skeleton_mask.shape[1], skeleton_mask.shape[0]), 0)

        num_points = int(np.random.uniform(8, 15))

        sampled_points_top = np.linspace(ori_rec_points[3], ori_rec_points[0], num_points)
        sampled_points_top = [(int(x), int(y)) for x, y in sampled_points_top]

        sampled_points_down = np.linspace(ori_rec_points[2], ori_rec_points[1], num_points)
        sampled_points_down = [(int(x), int(y)) for x, y in sampled_points_down]

        vertex = []
        for top_point, down_point in zip(sampled_points_top, sampled_points_down):
            random_move = np.random.uniform(-0.6, 0.6)
            sampled_x, sampled_y = top_point
            sampled_x = sampled_x + int(np.random.uniform(0, 1.0) * (sampled_points_top[1][0] - sampled_points_top[0][0]))
            sampled_y = sampled_y + int(random_move * (sampled_points_down[1][1] - sampled_points_down[0][1]))
            vertex.append((sampled_x, sampled_y))

            sampled_x, sampled_y = down_point
            random_move = np.random.uniform(-0.6, 0.6)
            sampled_x = sampled_x - int(np.random.uniform(0, 1.0) * (sampled_points_top[1][0] - sampled_points_top[0][0]))
            sampled_y = sampled_y + int(random_move * (sampled_points_down[1][1] - sampled_points_down[0][1]))
            vertex.append((sampled_x, sampled_y))

        draw = ImageDraw.Draw(mask)
        min_width = 12
        max_width = 48
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)

        mask = np.asarray(mask, np.uint8) * 255

        return mask

    def random_brush_augment(self, skeleton_mask, ori_rec_points):

        brush_direction_type = random.uniform(0, 1)
        if brush_direction_type < 0.25:
            brush_mask = self.random_brush_top_down(skeleton_mask, ori_rec_points)
        elif brush_direction_type < 0.5:
            brush_mask = self.random_brush_down_top(skeleton_mask, ori_rec_points)
        elif brush_direction_type < 0.75:
            brush_mask = self.random_brush_left_right(skeleton_mask, ori_rec_points)
        else:
            brush_mask = self.random_brush_right_left(skeleton_mask, ori_rec_points)

        skeleton_mask[skeleton_mask == 0] = brush_mask[skeleton_mask == 0]
        return skeleton_mask

    def compute_diff_mask(self, ori_coordinates, new_coordinates,
                          skeleton_mask):

        skeleton_mask = skeleton_mask * 255

        diff_skeleton_list = []
        for subset_idx, subset in enumerate(ori_coordinates['subset']):
            for skeleton_idx in range(18):
                if ori_coordinates['candidate'][
                        ori_coordinates['subset'][subset_idx]
                    [skeleton_idx]] != new_coordinates['candidate'][
                        new_coordinates['subset'][subset_idx][skeleton_idx]]:
                    diff_skeleton_list.append(f'{subset_idx}_{skeleton_idx}')

        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        for diff_skeleton in diff_skeleton_list:
            subset_idx, skeleton_idx = diff_skeleton.split('_')
            subset_idx = int(subset_idx)
            for limb in limbSeq:
                if int(skeleton_idx) + 1 in limb:
                    index_point_1 = int(
                        ori_coordinates['subset'][subset_idx][limb[0] - 1])
                    index_point_2 = int(
                        ori_coordinates['subset'][subset_idx][limb[1] - 1])

                    if index_point_1 != -1 and index_point_2 != -1:
                        point1 = ori_coordinates['candidate'][index_point_1][
                            0:2]
                        point2 = ori_coordinates['candidate'][index_point_2][
                            0:2]

                        point2[0] = point2[0] + 0.7 * (point2[0] - point1[0])
                        point2[1] = point2[1] + 0.7 * (point2[1] - point1[1])

                        length = ((point1[0] - point2[0])**2 +
                                  (point1[1] - point2[1])**2)**0.5

                        length_ratio = random.uniform(0.20, 0.40)
                        ori_rec_points = self.find_parallel_points(
                            point1, point2, length_ratio * length)

                        cv2.fillPoly(skeleton_mask, [np.array(ori_rec_points)],
                                     255)
                        skeleton_mask = self.random_brush_augment(skeleton_mask, ori_rec_points)

                    index_point_1 = int(
                        new_coordinates['subset'][subset_idx][limb[0] - 1])
                    index_point_2 = int(
                        new_coordinates['subset'][subset_idx][limb[1] - 1])

                    if index_point_1 != -1 and index_point_2 != -1:
                        point1 = new_coordinates['candidate'][index_point_1][
                            0:2]
                        point2 = new_coordinates['candidate'][index_point_2][
                            0:2]

                        point2[0] = point2[0] + 0.7 * (point2[0] - point1[0])
                        point2[1] = point2[1] + 0.7 * (point2[1] - point1[1])

                        length = ((point1[0] - point2[0])**2 +
                                  (point1[1] - point2[1])**2)**0.5

                        length_ratio = random.uniform(0.20, 0.40)
                        ori_rec_points = self.find_parallel_points(
                            point1, point2, length_ratio * length)

                        cv2.fillPoly(skeleton_mask, [np.array(ori_rec_points)],
                                     255)
                        skeleton_mask = self.random_brush_augment(skeleton_mask, ori_rec_points)
        skeleton_mask = skeleton_mask / 255

        return skeleton_mask

    def get_id_feature(self, candidate_parsing_list):

        id_feature_list = []
        for instance_parsing in candidate_parsing_list:
            bbox_mask = np.zeros(
                (instance_parsing.shape[0], instance_parsing.shape[1]),
                dtype=np.uint8)
            mask_binary = np.zeros(
                (instance_parsing.shape[0], instance_parsing.shape[1]),
                dtype=np.uint8)
            mask_binary[instance_parsing > 0] = 1

            mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

            obj_ids = torch.unique(mask_tensor)
            obj_ids = obj_ids[1:]
            masks = mask_tensor == obj_ids[:, None, None]

            boxes = masks_to_boxes(masks)

            h, w = mask_binary.shape

            enlarge_ratio = 0.1
            enlarge_margin_h = int((boxes[0][3] - boxes[0][1]) * enlarge_ratio)
            enlarge_margin_w = int((boxes[0][2] - boxes[0][0]) * enlarge_ratio)

            bbox_y1, bbox_y2 = max(0,
                                   int(boxes[0][1]) - enlarge_margin_h), min(
                                       h,
                                       int(boxes[0][3]) + enlarge_margin_h)
            bbox_x1, bbox_x2 = max(0,
                                   int(boxes[0][0]) - enlarge_margin_w), min(
                                       w,
                                       int(boxes[0][2]) + enlarge_margin_w)
            bbox_mask[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = 1
            id_feature_list.append(bbox_mask)

        return id_feature_list

    def generate_skeletion_mask(self, coordinates, skeleton_map):
        skeleton_mask = np.zeros(
            (skeleton_map.shape[0], skeleton_map.shape[1]), dtype=np.uint8)

        candidate = coordinates['candidate']
        subset = coordinates['subset']

        selected_person_idx = random.choice(range(len(subset)))

        skeleton_joint_list = []
        random_type = random.uniform(0, 1)
        if random_type < 0.35:
            skeleton_joint_list.append([2, 3])
            skeleton_joint_list.append([3, 4])
        elif random_type < 0.7:
            skeleton_joint_list.append([5, 6])
            skeleton_joint_list.append([6, 7])
        else:
            skeleton_joint_list.append([2, 3])
            skeleton_joint_list.append([3, 4])
            skeleton_joint_list.append([5, 6])
            skeleton_joint_list.append([6, 7])

        # left and right arms
        for skeleton_joint in skeleton_joint_list:
            index_point_1 = int(subset[selected_person_idx][skeleton_joint[0]])
            index_point_2 = int(subset[selected_person_idx][skeleton_joint[1]])

            if index_point_1 != -1 and index_point_2 != -1:
                point1 = candidate[index_point_1][0:2]
                point2 = candidate[index_point_2][0:2]

                point2[0] = point2[0] + 0.7 * (point2[0] - point1[0])
                point2[1] = point2[1] + 0.7 * (point2[1] - point1[1])

                length = ((point1[0] - point2[0])**2 +
                            (point1[1] - point2[1])**2)**0.5

                length_ratio = random.uniform(0.20, 0.40)
                ori_rec_points = self.find_parallel_points(
                    point1, point2, length_ratio * length)

                cv2.fillPoly(skeleton_mask, [np.array(ori_rec_points)], 255)

                # import pdb
                # pdb.set_trace()
                # Image.fromarray(skeleton_mask).save('temp_skeleton_mask.png')
                skeleton_mask = self.random_brush_augment(skeleton_mask, ori_rec_points)
                # import pdb
                # pdb.set_trace()
                # Image.fromarray(skeleton_mask).save('temp_skeleton_mask.png')

        skeleton_mask = skeleton_mask / 255

        # selected person bbox
        selected_person_bbox = np.zeros(
            (skeleton_map.shape[0], skeleton_map.shape[1]), dtype=np.uint8)
        x_list = []
        y_list = []

        for i in range(18):
            index = int(subset[selected_person_idx][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x_list.append(x)
            y_list.append(y)

        x_min = min(x_list)
        x_max = max(x_list)
        y_min = min(y_list)
        y_max = max(y_list)

        x1 = int(max(0, x_min - 0.4 * (x_max - x_min)))
        x2 = int(x_max + 0.4 * (x_max - x_min))
        y1 = int(max(0, y_min - 0.4 * (y_max - y_min)))
        y2 = int(y_max + 0.4 * (y_max - y_min))

        selected_person_bbox[y1:y2, x1:x2] = 1

        return skeleton_mask, selected_person_idx, selected_person_bbox

    def mmpose_to_openpose(self, mmpose_coordinates, bbox_threshold=0.2):
        num_persons = len(mmpose_coordinates)
        coordinates = {}
        coordinates['subset'] = []
        coordinates['candidate'] = []

        coordinate_count = 0
        for person_idx in range(num_persons):
            if mmpose_coordinates[person_idx]["bbox_score"] < bbox_threshold:
                continue
            subset = {}
            for subset_idx in range(18):
                subset[subset_idx] = -1
            for subset_idx, skeleton_idx in enumerate(
                [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4,
                 3]):
                if skeleton_idx == 17:
                    if mmpose_coordinates[person_idx]["keypoint_scores"][
                            6] < 0.1:
                        continue
                    if mmpose_coordinates[person_idx]["keypoint_scores"][
                            5] < 0.1:
                        continue
                    subset[subset_idx] = coordinate_count
                    coordinates_6 = mmpose_coordinates[person_idx][
                        "keypoints"][6]
                    coordinates_5 = mmpose_coordinates[person_idx][
                        "keypoints"][5]
                    coordinates['candidate'].append([
                        (coordinates_6[0] + coordinates_5[0]) / 2.0,
                        (coordinates_6[1] + coordinates_5[1]) / 2.0
                    ])
                    coordinate_count += 1
                else:
                    if mmpose_coordinates[person_idx]["keypoint_scores"][
                            skeleton_idx] < 0.5:
                        continue
                    subset[subset_idx] = coordinate_count
                    coordinates['candidate'].append(
                        mmpose_coordinates[person_idx]["keypoints"]
                        [skeleton_idx])
                    coordinate_count += 1

            coordinates['subset'].append(subset)

        return coordinates

    def generate_bbox_from_mask(self, mask):
        # Find the coordinates of non-zero elements in the mask
        y_coords, x_coords = np.where(mask)

        if len(y_coords) == 0 or len(x_coords) == 0:
            # No non-zero elements found (empty mask)
            return None

        # Compute the bounding box corners
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_min, x_max = np.min(x_coords), np.max(x_coords)

        # Return the bounding box coordinates as (y_min, x_min, y_max, x_max)
        return y_min, x_min, y_max, x_max

    def generate_skeletion_mask(self, coordinates, skeleton_map):
        skeleton_mask = np.zeros(
            (skeleton_map.shape[0], skeleton_map.shape[1]), dtype=np.uint8)

        candidate = coordinates['candidate']
        subset = coordinates['subset']

        selected_person_idx = random.choice(range(len(subset)))

        # left arms
        coordinates_x_list = []
        coordinates_y_list = []
        for body_idx in [2, 3, 4]:
            index = int(subset[selected_person_idx][body_idx])
            if index == -1:
                continue
            coordinates_x, coordinates_y = candidate[index][0:2]
            coordinates_x_list.append(coordinates_x)
            coordinates_y_list.append(coordinates_y)

        if len(coordinates_x_list) != 0:
            left_x = int(min(coordinates_x_list))
            up_y = int(min(coordinates_y_list))

            right_x = int(max(coordinates_x_list))
            down_y = int(max(coordinates_y_list))

            pad_width = int(max(down_y - up_y, right_x - left_x) * 0.15)

            skeleton_mask[max(0, up_y - pad_width):down_y + pad_width,
                          max(0, left_x - pad_width):right_x + pad_width] = 1

        # right arms
        coordinates_x_list = []
        coordinates_y_list = []
        for body_idx in [5, 6, 7]:
            index = int(subset[selected_person_idx][body_idx])
            if index == -1:
                continue
            coordinates_x, coordinates_y = candidate[index][0:2]
            coordinates_x_list.append(coordinates_x)
            coordinates_y_list.append(coordinates_y)

        if len(coordinates_x_list) != 0:
            left_x = int(min(coordinates_x_list))
            up_y = int(min(coordinates_y_list))

            right_x = int(max(coordinates_x_list))
            down_y = int(max(coordinates_y_list))

            pad_width = int(max(down_y - up_y, right_x - left_x) * 0.15)

            skeleton_mask[max(0, up_y - pad_width):down_y + pad_width,
                          max(0, left_x - pad_width):right_x + pad_width] = 1

        # selected person bbox
        selected_person_bbox = np.zeros(
            (skeleton_map.shape[0], skeleton_map.shape[1]), dtype=np.uint8)
        x_list = []
        y_list = []

        for i in range(18):
            index = int(subset[selected_person_idx][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x_list.append(x)
            y_list.append(y)

        x_min = min(x_list)
        x_max = max(x_list)
        y_min = min(y_list)
        y_max = max(y_list)

        x1 = int(max(0, x_min - 0.4 * (x_max - x_min)))
        x2 = int(x_max + 0.4 * (x_max - x_min))
        y1 = int(max(0, y_min - 0.4 * (y_max - y_min)))
        y2 = int(y_max + 0.4 * (y_max - y_min))

        selected_person_bbox[y1:y2, x1:x2] = 1

        return skeleton_mask, selected_person_idx, selected_person_bbox

    def expand_identity_feature(self, id_feature_list, selected_idx,
                                inpaint_mask):

        id_feature_temp = id_feature_list[selected_idx].copy()
        id_feature_temp[inpaint_mask == 1] = 1

        if np.sum(id_feature_temp) == (id_feature_temp.shape[0] *
                                       id_feature_temp.shape[1]):
            id_feature_list[selected_idx] = id_feature_temp
            return id_feature_list

        mask_tensor = torch.from_numpy(id_feature_temp).unsqueeze(0)

        obj_ids = torch.unique(mask_tensor)
        obj_ids = obj_ids[1:]
        masks = mask_tensor == obj_ids[:, None, None]

        boxes = masks_to_boxes(masks)

        bbox_y1, bbox_y2 = max(0, int(boxes[0][1])), int(boxes[0][3])
        bbox_x1, bbox_x2 = max(0, int(boxes[0][0])), int(boxes[0][2])

        id_feature_temp[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = 1

        id_feature_list[selected_idx] = id_feature_temp

        return id_feature_list

    def adjust_coordinates(self, coordinates, original_size):
        ratio = 512. / original_size
        for candidate in coordinates['candidate']:
            candidate[0] = candidate[0] * ratio
            candidate[1] = candidate[1] * ratio

        return coordinates

    def flip_skeleton_coordinates(self, coordinates):

        for subset_index in range(len(coordinates['subset'])):
            new_subset = {}
            for index in range(18):
                if index == 2:
                    new_subset[index] = coordinates['subset'][subset_index][5]
                elif index == 3:
                    new_subset[index] = coordinates['subset'][subset_index][6]
                elif index == 4:
                    new_subset[index] = coordinates['subset'][subset_index][7]
                elif index == 8:
                    new_subset[index] = coordinates['subset'][subset_index][11]
                elif index == 9:
                    new_subset[index] = coordinates['subset'][subset_index][12]
                elif index == 10:
                    new_subset[index] = coordinates['subset'][subset_index][13]
                elif index == 5:
                    new_subset[index] = coordinates['subset'][subset_index][2]
                elif index == 6:
                    new_subset[index] = coordinates['subset'][subset_index][3]
                elif index == 7:
                    new_subset[index] = coordinates['subset'][subset_index][4]
                elif index == 11:
                    new_subset[index] = coordinates['subset'][subset_index][8]
                elif index == 12:
                    new_subset[index] = coordinates['subset'][subset_index][9]
                elif index == 13:
                    new_subset[index] = coordinates['subset'][subset_index][10]
                else:
                    new_subset[index] = coordinates['subset'][subset_index][
                        index]
            coordinates['subset'][subset_index] = new_subset

        return coordinates



    def draw_bodypose(self, canvas, candidate, subset):
        stickwidth = 4
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        for i in range(18):
            for n in range(len(subset)):
                index = int(subset[n][i])
                if index == -1:
                    continue
                x, y = candidate[index][0:2]
                cv2.circle(
                    canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
        for i in range(17):
            for n in range(len(subset)):
                index = [subset[n][point - 1] for point in limbSeq[i]]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = [candidate[int(point)][0] for point in index]
                X = [candidate[int(point)][1] for point in index]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                           (int(length / 2), stickwidth),
                                           int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
        # plt.imshow(canvas[:, :, [2, 1, 0]])
        return canvas

    def crop_img_mask(self, img, human_mask, bbox_mask, bbox_coor, face_mask,
                      target_person_face_mask, skeleton_map, skeleton_mask,
                      coordinates, instance_parsing_list):
        h, w, _ = img.shape

        x1, y1, x2, y2 = bbox_coor

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        bbox_h, bbox_w = y2 - y1, x2 - x1

        enlarge_bbox_ratio = 1.1

        enlarged_bbox = int(max([bbox_h, bbox_w]) * enlarge_bbox_ratio)

        cropped_size = min([h, w, enlarged_bbox])
        cropped_size = cropped_size // 2 * 2

        crop_y1 = center_y - cropped_size // 2
        crop_y2 = center_y + cropped_size // 2
        crop_x1 = center_x - cropped_size // 2
        crop_x2 = center_x + cropped_size // 2
        if crop_y1 < 0:
            crop_y1 = 0
            crop_y2 = cropped_size

        if crop_y2 > h:
            crop_y1 = h - cropped_size
            crop_y2 = h

        if crop_x1 < 0:
            crop_x1 = 0
            crop_x2 = cropped_size

        if crop_x2 > w:
            crop_x1 = w - cropped_size
            crop_x2 = w

        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        human_mask = human_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        bbox_mask = bbox_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        face_mask = face_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        target_person_face_mask = target_person_face_mask[crop_y1:crop_y2,
                                                          crop_x1:crop_x2]
        skeleton_map = skeleton_map[crop_y1:crop_y2, crop_x1:crop_x2]
        skeleton_mask = skeleton_mask[crop_y1:crop_y2, crop_x1:crop_x2]

        cropped_instance_parsing_list = []
        for instance_parsing in instance_parsing_list:
            cropped_instance_parsing_list.append(
                instance_parsing[crop_y1:crop_y2, crop_x1:crop_x2])

        for candidate in coordinates['candidate']:
            candidate[0] = candidate[0] - crop_x1
            candidate[1] = candidate[1] - crop_y1

        # import pdb
        # pdb.set_trace()
        current_width = img.shape[0]
        for subset in coordinates['subset']:
            for index in range(17):
                if subset[index] == -1:
                    continue
        return img, human_mask, bbox_mask, face_mask, target_person_face_mask, skeleton_map, skeleton_mask, coordinates, cropped_instance_parsing_list

    def occlusion_deleting(self, bbox_mask):
        indices = np.where(bbox_mask != 0)
        x_min, y_min = np.min(indices, axis=1)
        x_max, y_max = np.max(indices, axis=1)

        inpaint_mask = np.zeros((bbox_mask.shape[0], bbox_mask.shape[1]),
                                dtype=np.uint8)
        random_length = int(random.uniform(0.2, 0.4) * (y_max - y_min))

        location = random.choice([0, 1])
        if location == 0:
            inpaint_mask[:,
                         max(0, y_max - random_length // 2):y_max +
                         random_length // 2] = 1
        else:
            inpaint_mask[:,
                         max(0, y_min - random_length // 2):y_min +
                         random_length // 2] = 1

        return inpaint_mask

    def get_id_color_map(self, id_feature_list):
        random.shuffle(self.random_color_identity_group)

        color_list = []
        identity_map = np.zeros(
            (id_feature_list[0].shape[0], id_feature_list[0].shape[1], 3))
        count_map = np.zeros(
            (id_feature_list[0].shape[0], id_feature_list[0].shape[1]))
        for idx, mask in enumerate(id_feature_list):
            color_group_idx = idx % 5
            random_color = random.choices(
                self.random_color_identity_group[color_group_idx], k=1)[0]
            temp_mask = np.zeros(
                (id_feature_list[0].shape[0], id_feature_list[0].shape[1], 3),
                dtype=np.uint8)

            temp_mask[mask == 1] = random_color
            # import pdb
            # pdb.set_trace()
            # identity_color_indicator.append(random_color)
            identity_map += temp_mask
            # import pdb
            # pdb.set_trace()
            count_map += mask
            color_list.append(random_color)

        count_map[count_map == 0] = 1
        count_map = count_map[:, :, np.newaxis]
        # import pdb
        # pdb.set_trace()
        identity_map = identity_map / count_map
        identity_map = identity_map.astype(np.uint8)

        return identity_map, color_list

    def reposing_add(self, bbox_mask, inpaint_mask):
        indices = np.where(bbox_mask != 0)
        y_min, x_min = np.min(indices, axis=1)
        y_max, x_max = np.max(indices, axis=1)

        augmentation = random.uniform(0.4, 1.0)
        random_length = int(augmentation * (y_max - y_min))
        inpaint_mask[y_min + random_length:y_max, x_min:x_max] = 1

        return inpaint_mask

    def harmonization_add(self, img, bbox_mask, human_mask, inpaint_mask):

        img_augmented = Image.fromarray(img)
        transform = transforms.ColorJitter(
            brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.5))
        img_augmented = np.array(transform(img_augmented))

        revised_img = img.copy()
        revised_img[human_mask == 1, :] = img_augmented[human_mask == 1, :]
        inpaint_mask[bbox_mask == 1] = 1

        return revised_img, inpaint_mask

    def occlusion_add(self, img, human_mask, bbox_mask):
        indices = np.where(bbox_mask != 0)
        y_min, x_min = np.min(indices, axis=1)
        y_max, x_max = np.max(indices, axis=1)

        bbox_mask_revised = bbox_mask.copy()
        column_mask = random.uniform(0, 1)
        if column_mask < 0.5:
            bbox_mask_revised[:, x_min:x_max] = 1

        inpaint_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        inpaint_mask[(bbox_mask_revised - human_mask) > 0] = 1

        return inpaint_mask, bbox_mask_revised

    def read_mask_for_delete(self, selected_parsing_idx,
                             instance_parsing_list):
        mask = instance_parsing_list[selected_parsing_idx]

        mask_binary = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        mask_binary[mask > 0] = 1

        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        obj_ids = torch.unique(mask_tensor)
        obj_ids = obj_ids[1:]
        masks = mask_tensor == obj_ids[:, None, None]

        boxes = masks_to_boxes(masks)

        bbox_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        h, w = mask.shape

        bbox_y1, bbox_y2 = max(0, int(boxes[0][1])), min(h, int(boxes[0][3]))
        bbox_x1, bbox_x2 = max(0, int(boxes[0][0])), min(w, int(boxes[0][2]))
        bbox_mask[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = 1

        face_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for idx, parsing in enumerate(instance_parsing_list):

            if idx == selected_parsing_idx:
                target_person_face_mask = np.zeros(
                    (mask.shape[0], mask.shape[1]), dtype=np.uint8)

            for i in range(1, 5):
                face_mask[parsing == i] = 1

                if idx == selected_parsing_idx:
                    target_person_face_mask[parsing == i] = 1

        return mask_binary, bbox_mask, (
            bbox_x1, bbox_y1, bbox_x2,
            bbox_y2), face_mask, target_person_face_mask

    def read_mask(self, selected_parsing_idx, instance_parsing_list):
        mask = instance_parsing_list[selected_parsing_idx]

        # mask_name = mask_path.split('/')[-1][:-4]
        # img_id = int(mask_name.split('_')[0])
        num_persons = len(instance_parsing_list)
        person_id = selected_parsing_idx + 1
        # num_persons = int(mask_name.split('_')[-2])
        # person_id = int(mask_name.split('_')[-1])

        mask_binary = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        mask_binary[mask > 0] = 1

        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        obj_ids = torch.unique(mask_tensor)
        obj_ids = obj_ids[1:]
        masks = mask_tensor == obj_ids[:, None, None]

        boxes = masks_to_boxes(masks)

        bbox_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        h, w = mask.shape

        # make the bounding box slightly larger
        enlarge_ratio = 0.05
        enlarge_margin_h = int((boxes[0][3] - boxes[0][1]) * enlarge_ratio)
        enlarge_margin_w = int((boxes[0][2] - boxes[0][0]) * enlarge_ratio)

        if person_id > 1:
            # left_person = f'{mask_path[:-4][:-len(mask_name)]}/{img_id}_{num_persons:02d}_{(person_id-1):02d}.png'
            mask_left = instance_parsing_list[selected_parsing_idx - 1]

            mask_binary_left = np.zeros(
                (mask_left.shape[0], mask_left.shape[1]), dtype=np.uint8)
            mask_binary_left[mask_left > 0] = 1

            mask_tensor_left = torch.from_numpy(mask_binary_left).unsqueeze(0)

            obj_ids_left = torch.unique(mask_tensor_left)
            obj_ids_left = obj_ids_left[1:]
            masks_left = mask_tensor_left == obj_ids_left[:, None, None]

            boxes_left = masks_to_boxes(masks_left)

            enlarge_margin_left = min(
                enlarge_margin_w,
                int((boxes_left[0][2] - boxes_left[0][0]) * 0.05))
        else:
            enlarge_margin_left = enlarge_margin_w

        if person_id < num_persons:
            # right_person = f'{mask_path[:-4][:-len(mask_name)]}/{img_id}_{num_persons:02d}_{(person_id+1):02d}.png'
            mask_right = instance_parsing_list[selected_parsing_idx + 1]

            mask_binary_right = np.zeros(
                (mask_right.shape[0], mask_right.shape[1]), dtype=np.uint8)
            mask_binary_right[mask_right > 0] = 1

            mask_tensor_right = torch.from_numpy(mask_binary_right).unsqueeze(
                0)

            obj_ids_right = torch.unique(mask_tensor_right)
            obj_ids_right = obj_ids_right[1:]
            masks_right = mask_tensor_right == obj_ids_right[:, None, None]

            boxes_right = masks_to_boxes(masks_right)

            enlarge_margin_right = min(
                enlarge_margin_w,
                int((boxes_right[0][2] - boxes_right[0][0]) * 0.05))
        else:
            enlarge_margin_right = enlarge_margin_w

        bbox_y1, bbox_y2 = max(0,
                               int(boxes[0][1]) - enlarge_margin_h), min(
                                   h,
                                   int(boxes[0][3]) + enlarge_margin_h)
        bbox_x1, bbox_x2 = max(0,
                               int(boxes[0][0]) - enlarge_margin_left), min(
                                   w,
                                   int(boxes[0][2]) + enlarge_margin_right)
        bbox_mask[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = 1

        face_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        for idx, parsing in enumerate(instance_parsing_list):

            if idx == selected_parsing_idx:
                target_person_face_mask = np.zeros(
                    (mask.shape[0], mask.shape[1]), dtype=np.uint8)

            for i in range(1, 5):
                face_mask[parsing == i] = 1

                if idx == selected_parsing_idx:
                    target_person_face_mask[parsing == i] = 1

        return mask_binary, bbox_mask, (
            bbox_x1, bbox_y1, bbox_x2,
            bbox_y2), face_mask, target_person_face_mask

    def remove_background(self, img, instance_parsing_list):

        mask_binary = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for instance_parsing in instance_parsing_list:
            mask_binary[instance_parsing > 0] = 1

        img[mask_binary == 0] = 255

        return img

    def load_instance_parsing_maps(self, parsing_path_list):

        parsing_list = []
        for parsing_path in parsing_path_list:
            mask = np.array(Image.open(parsing_path).convert('RGB'))[:, :, 0]
            parsing_list.append(mask)

        return parsing_list


    def __getitem__(self, index):

        while True:
            try:
                img_path = self.data_path_list[index]
                img_name = img_path.split('/')[-1][:-4]
                parsing_path_list = glob.glob(
                    f'{self.parsing_dir}/{img_name}_*.png')

                instance_parsing_list = self.load_instance_parsing_maps(
                    parsing_path_list)

                coordinates_path = f'{self.pose_estimation_path}/{img_name}.json'
                # a function to load json file from locally
                mmpose_coordinates = json.load(open(coordinates_path, ))

                coordinates = self.mmpose_to_openpose(mmpose_coordinates)

                img = self.read_img(img_path)

                # remove background
                remove_bg_aug = random.uniform(0, 1)
                if remove_bg_aug < 0.2:
                    img = self.remove_background(img, instance_parsing_list)

                canvas = np.zeros_like(img)
                skeleton_map = self.draw_bodypose(canvas,
                                                  coordinates['candidate'],
                                                  coordinates['subset'])

                skeleton_mask, selected_person_idx, selected_person_bbox = self.generate_skeletion_mask(
                    coordinates, skeleton_map)

                new_coordinates = self.randomly_change_pose(
                    coordinates, selected_person_idx)

                skeleton_mask = self.compute_diff_mask(coordinates,
                                                       new_coordinates,
                                                       skeleton_mask)

                skeleton_mask, selected_parsing_idx = self.load_arm_hand_masks(
                    skeleton_mask, selected_person_bbox, instance_parsing_list)

                # parsing_path = random.choice(candidate_parsing_path_list)
                # augmentation types:
                # 1) occlusion for adding a person;
                # 2) harmonization when adding a person;
                # 3) reposing when adding a person;
                # 4) occlusion for removing a persons
                augmentation_type = random.uniform(0, 1)
                if augmentation_type < 0.8:
                    # add person cases
                    human_mask, bbox_mask, bbox_coor, face_mask, target_person_face_mask = self.read_mask(
                        selected_parsing_idx, instance_parsing_list)

                    # define the crop region
                    img, human_mask, bbox_mask, face_mask, target_person_face_mask, skeleton_map, skeleton_mask, coordinates, instance_parsing_list = self.crop_img_mask(
                        img, human_mask, bbox_mask, bbox_coor, face_mask,
                        target_person_face_mask, skeleton_map, skeleton_mask,
                        coordinates, instance_parsing_list)
                    assert np.sum(bbox_mask) != 0

                    revised_img = img.copy()

                    # for each add person type, we need to deal with the occlusion region
                    inpaint_mask, bbox_mask_revised = self.occlusion_add(
                        img, human_mask, bbox_mask)

                    inpaint_mask[skeleton_mask == 1] = 1

                    reposing_aug = random.uniform(0, 1)
                    if reposing_aug > 0.4:
                        inpaint_mask = self.reposing_add(
                            bbox_mask, inpaint_mask)

                    inpaint_mask[face_mask == 1] = 0

                    # dilated inpaint mask
                    dilate_inpaint_aug = random.uniform(0, 1)
                    if dilate_inpaint_aug < 0.4:
                        structuring_element = np.ones((5, 5), dtype=bool)
                        inpaint_mask = binary_dilation(
                            inpaint_mask,
                            structure=structuring_element).astype(np.uint8)

                    inpaint_mask_after_reposing = inpaint_mask.copy()

                    harmonization_aug = random.uniform(0, 1)
                    # add_harmonization = 0.1
                    if harmonization_aug < 0.5 and self.add_harmonization:
                        revised_img, inpaint_mask = self.harmonization_add(
                            img, bbox_mask_revised, human_mask, inpaint_mask)

                        # exclude the surrounding person from the inpainting regions
                        inpaint_mask[(face_mask -
                                      target_person_face_mask) == 1] = 0
                    # else:
                    #     inpaint_mask[bbox_mask_revised == 1] = 1
                    # inpaint_mask[(face_mask - target_person_face_mask) == 1] = 0
                else:
                    human_mask, bbox_mask, bbox_coor, face_mask, target_person_face_mask = self.read_mask_for_delete(
                        selected_parsing_idx, instance_parsing_list)

                    # define the crop region
                    img, human_mask, bbox_mask, face_mask, target_person_face_mask, skeleton_map, skeleton_mask, coordinates, instance_parsing_list = self.crop_img_mask(
                        img, human_mask, bbox_mask, bbox_coor, face_mask,
                        target_person_face_mask, skeleton_map, skeleton_mask,
                        coordinates, instance_parsing_list)
                    assert np.sum(bbox_mask) != 0

                    inpaint_mask = self.occlusion_deleting(human_mask)
                    revised_img = img.copy()

                    inpaint_mask[skeleton_mask == 1] = 1
                    inpaint_mask[face_mask == 1] = 0

                    inpaint_mask_after_reposing = inpaint_mask.copy()

                # load the exemplar image
                candidate_parsing_list, idx_in_candidate_list = self.get_candidate_parsing_list_for_exemplar(
                    inpaint_mask, selected_parsing_idx, instance_parsing_list)

                if len(candidate_parsing_list) == 0:
                    index = random.randint(0, len(self.data_path_list) - 1)
                    continue

                # get_indicator
                id_feature_list = self.get_id_feature(candidate_parsing_list)

                # expand the id feature list using the inpaint mask
                id_feature_list = self.expand_identity_feature(
                    id_feature_list, idx_in_candidate_list, inpaint_mask)

                id_color_map, color_list = self.get_id_color_map(
                    id_feature_list)

                img_exemplar_list, parsing_exemplar_list = self.read_img_exemplar_mask(
                    img, candidate_parsing_list)
                for idx, (img_exemplar, parsing) in enumerate(
                        zip(img_exemplar_list, parsing_exemplar_list)):
                    incomplete_exemplar_aug = random.uniform(0, 1)
                    if incomplete_exemplar_aug < 0.4:
                        length = img_exemplar.shape[0]
                        random_portion = random.uniform(0.2, 0.6)
                        # the masked part should be directly cropped out, rather than applying the mask
                        # img_exemplar[-int(random_portion * length):, :] = 255
                        img_exemplar = img_exemplar[:-int(random_portion *
                                                          length), :]
                        parsing = parsing[:-int(random_portion * length), :]
                    img_exemplar_list[idx] = img_exemplar
                    parsing_exemplar_list[idx] = parsing

                img = torch.from_numpy(img).permute(2, 0, 1)
                id_color_map = torch.from_numpy(id_color_map).permute(2, 0, 1)
                skeleton_map = torch.from_numpy(skeleton_map).permute(2, 0, 1)
                revised_img = torch.from_numpy(revised_img).permute(2, 0, 1)
                inpaint_mask = torch.from_numpy(inpaint_mask).unsqueeze(0)
                human_mask = torch.from_numpy(human_mask).unsqueeze(0)
                skeleton_mask = torch.from_numpy(skeleton_mask).unsqueeze(0)

                exemplar_img_list = []
                exemplar_skeleton_map_list = []
                exemplar_skeleton_coordinates_list = []
                exemplar_color_block_list = []
                for idx, (img_exemplar, parsing) in enumerate(
                        zip(img_exemplar_list, parsing_exemplar_list)):

                    img_exemplar = torch.from_numpy(img_exemplar).permute(
                        2, 0, 1)
                    height, width = img_exemplar.size(1), img_exemplar.size(2)

                    parsing = torch.from_numpy(parsing).unsqueeze(0)

                    if height == width:
                        pass
                    elif height < width:
                        diff = width - height
                        top_pad = diff // 2
                        down_pad = diff - top_pad
                        left_pad = 0
                        right_pad = 0
                        padding_size = [left_pad, top_pad, right_pad, down_pad]
                        img_exemplar = F.pad(
                            img_exemplar, padding=padding_size, fill=255)
                        parsing = F.pad(parsing, padding=padding_size, fill=0)
                    else:
                        diff = height - width
                        left_pad = diff // 2
                        right_pad = diff - left_pad
                        top_pad = 0
                        down_pad = 0
                        padding_size = [left_pad, top_pad, right_pad, down_pad]
                        img_exemplar = F.pad(
                            img_exemplar, padding=padding_size, fill=255)
                        parsing = F.pad(parsing, padding=padding_size, fill=0)

                    exemplar_img, parsing = self.transform_exemplar_and_parsing(
                        img_exemplar, parsing)
                    exemplar_img = exemplar_img.permute(1, 2, 0)
                    parsing = parsing.squeeze(0)

                    exemplar_img, new_coordinates = self.reposing_exemplar_img(
                        exemplar_img.numpy(), parsing.numpy())

                    exemplar_skeleton_map = self.draw_bodypose(
                        np.zeros_like(exemplar_img),
                        new_coordinates['candidate'],
                        new_coordinates['subset'])

                    exemplar_img = self.resize_transform_exemplar(
                        torch.from_numpy(exemplar_img).permute(
                            2, 0, 1)).permute(1, 2, 0) / 255.
                    exemplar_skeleton_map = torch.from_numpy(
                        exemplar_skeleton_map) / 255.0

                    exemplar_skeleton_coordinates_list.append(new_coordinates)
                    # flip_random = random.uniform(0, 1)
                    # flip_random = 0.1
                    # if flip_random < 0.5:
                    # flip image
                    # exemplar_img = torch.fliplr(exemplar_img)
                    # flip skeleton
                    # new_coordinates = self.flip_skeleton_coordinates(new_coordinates)
                    # canvas = np.zeros_like(exemplar_skeleton_map)
                    # exemplar_skeleton_map = self.draw_bodypose(canvas, new_coordinates['candidate'], new_coordinates['subset'])
                    # exemplar_skeleton_map = torch.from_numpy(exemplar_skeleton_map) / 255.0
                    # exemplar_skeleton_map = torch.fliplr(exemplar_skeleton_map)

                    exemplar_skeleton_map_list.append(exemplar_skeleton_map)
                    exemplar_img_list.append(exemplar_img)

                    # generate color block
                    # import pdb
                    # pdb.set_trace()
                    exemplar_color_block = torch.zeros_like(
                        exemplar_skeleton_map)
                    exemplar_color_block[:, :, 0] = color_list[idx][0]
                    exemplar_color_block[:, :, 1] = color_list[idx][1]
                    exemplar_color_block[:, :, 2] = color_list[idx][2]
                    exemplar_color_block = exemplar_color_block / 255.
                    # exemplar_color_block = torch.tensor([[[color_list[idx][0], color_list[idx][1], color_list[idx][2]]] * 224] * 224) / 255.
                    exemplar_color_block_list.append(exemplar_color_block)

                if len(exemplar_img_list) > 5:
                    index = random.randint(0, len(self.data_path_list) - 1)
                    continue

                if len(exemplar_img_list) < 5:
                    add_length = 5 - len(exemplar_img_list)
                    for _ in range(add_length):
                        exemplar_img_list.append(
                            torch.zeros_like(exemplar_img_list[0]))
                        exemplar_skeleton_map_list.append(
                            torch.zeros_like(exemplar_skeleton_map_list[0]))
                        exemplar_skeleton_coordinates_list.append(None)
                        exemplar_color_block_list.append(
                            torch.zeros_like(exemplar_color_block_list[0]))
                        id_feature_list.append(
                            np.zeros_like(id_feature_list[0]))

                id_feature_channel_list = []
                for id_feature in id_feature_list:
                    id_feature_channel_list.append(
                        torch.from_numpy(id_feature.astype(
                            np.uint8)).unsqueeze(0))

                # id_feature_channel = torch.from_numpy(np.stack(id_feature_list, axis=0, dtype=np.uint8))

                img = self.resize_transform_img(img).permute(1, 2,
                                                             0) / 127.5 - 1
                id_color_map = self.resize_transform_mask(
                    id_color_map).permute(1, 2, 0) / 255.0
                revised_img = self.resize_transform_img(revised_img).permute(
                    1, 2, 0) / 127.5 - 1
                coordinates = self.adjust_coordinates(coordinates,
                                                      inpaint_mask.size(1))
                canvas = np.zeros_like(img)
                skeleton_map = self.draw_bodypose(canvas,
                                                  coordinates['candidate'],
                                                  coordinates['subset'])
                skeleton_map = torch.from_numpy(skeleton_map) / 255.0
                inpaint_mask = self.resize_transform_mask(
                    inpaint_mask).permute(1, 2, 0)
                human_mask = self.resize_transform_mask(human_mask).permute(
                    1, 2, 0)
                skeleton_mask = self.resize_transform_mask(
                    skeleton_mask).permute(1, 2, 0)

                for idx, id_feature in enumerate(id_feature_channel_list):
                    id_feature_channel_list[idx] = self.resize_transform_mask(
                        id_feature).permute(1, 2, 0)

                inpaint_mask_after_reposing = torch.from_numpy(
                    inpaint_mask_after_reposing).unsqueeze(0)
                inpaint_mask_after_reposing = self.resize_transform_mask(
                    inpaint_mask_after_reposing).permute(1, 2, 0).squeeze(2)
                revised_img[inpaint_mask_after_reposing == 1] = 0

                flip_img = random.uniform(0, 1)
                # flip_img = 0.1
                if flip_img < 0.5:
                    img = torch.fliplr(img)
                    id_color_map = torch.fliplr(id_color_map)
                    revised_img = torch.fliplr(revised_img)
                    inpaint_mask = torch.fliplr(inpaint_mask)
                    skeleton_mask = torch.fliplr(skeleton_mask)
                    coordinates = self.flip_skeleton_coordinates(coordinates)
                    canvas = np.zeros_like(img)
                    skeleton_map = self.draw_bodypose(canvas,
                                                      coordinates['candidate'],
                                                      coordinates['subset'])
                    skeleton_map = torch.from_numpy(skeleton_map) / 255.0
                    skeleton_map = torch.fliplr(skeleton_map)

                    for idx, id_feature in enumerate(id_feature_channel_list):
                        id_feature_channel_list[idx] = torch.fliplr(id_feature)

                    # flip exemplar, the flip operation for exemplar should be consistent as the original img
                    for idx, exemplar_img in enumerate(exemplar_img_list):
                        exemplar_coordinate = exemplar_skeleton_coordinates_list[
                            idx]
                        if exemplar_coordinate is None:
                            break
                        exemplar_img_list[idx] = torch.fliplr(exemplar_img)
                        coordinates = self.flip_skeleton_coordinates(
                            exemplar_coordinate)
                        canvas = np.zeros_like(canvas)
                        exemplar_skeleton_map = self.draw_bodypose(
                            canvas, coordinates['candidate'],
                            coordinates['subset'])
                        exemplar_skeleton_map = torch.from_numpy(
                            exemplar_skeleton_map) / 255.0
                        exemplar_skeleton_map = torch.fliplr(
                            exemplar_skeleton_map)
                        exemplar_skeleton_map_list[idx] = exemplar_skeleton_map

                exemplar_img_list = torch.stack(exemplar_img_list, dim=0)
                exemplar_skeleton_map_list = torch.stack(
                    exemplar_skeleton_map_list, dim=0)
                exemplar_color_block_list = torch.stack(
                    exemplar_color_block_list, dim=0)
                id_feature_channel = torch.stack(
                    id_feature_channel_list, dim=0)

                assert img.size()[0] == 512
                assert img.size()[1] == 512

                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.data_path_list) - 1)

        return {
            'GT': img,
            'masked_image': revised_img,
            'mask': inpaint_mask,
            'text': 'A photo of group portrait.',
            'skeleton_map': skeleton_map,
            'skeleton_mask': skeleton_mask,
            'exemplar': exemplar_img_list,
            'exemplar_skeleton': exemplar_skeleton_map_list,
            'id_color_map': id_color_map,
            'exemplar_color_block': exemplar_color_block_list,
            'id_feature_channel': id_feature_channel
        }

