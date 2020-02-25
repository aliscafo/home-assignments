#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq

from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    pose_to_view_mat3x4,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    TriangulationParameters,
)

MAX_REPR_ERR = 1.0
MIN_DEPTH = 0.1
MIN_TRIANG_ANGLE_DEG = 2.0
PNP_REPROJ_ERROR = 8.0


def add_points_to_cloud(point_cloud_builder, corners1, corners2, view1, view2, intrinsic_mat, triang_params):
    correspondences = build_correspondences(corners1, corners2, ids_to_remove=point_cloud_builder.ids)
    points, ids, median_cos = triangulate_correspondences(correspondences,
                                                          view1,
                                                          view2,
                                                          intrinsic_mat,
                                                          triang_params)
    point_cloud_builder.add_points(ids, points)
    return len(points)


def track_and_find_point_cloud(intrinsic_mat, corner_storage, known_view_1, known_view_2):
    num_frames = len(corner_storage)

    view_mats = [None] * num_frames
    point_cloud_builder = PointCloudBuilder()

    print(f'Frames with known views: {known_view_1[0]} and {known_view_2[0]}')
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    triang_params = TriangulationParameters(max_reprojection_error=MAX_REPR_ERR,
                                            min_triangulation_angle_deg=MIN_TRIANG_ANGLE_DEG,
                                            min_depth=MIN_DEPTH)

    add_points_to_cloud(point_cloud_builder,
                        corner_storage[known_view_1[0]],
                        corner_storage[known_view_2[0]],
                        view_mats[known_view_1[0]],
                        view_mats[known_view_2[0]],
                        intrinsic_mat,
                        triang_params)

    while True:
        was_updated = False

        for i in range(num_frames):
            if view_mats[i] is not None:
                continue

            print(f"\nCurrent frame: {i}")

            # find intersection of current point cloud and current frame
            corners = corner_storage[i]
            corner_ids = []
            points_3d = []
            points_2d = []

            for id, corner in zip(corners.ids, corners.points):
                if id not in point_cloud_builder.ids:
                    continue
                ind_in_builder, _ = np.nonzero(point_cloud_builder.ids == id)
                corner_ids.append(id)
                points_3d.append(point_cloud_builder.points[ind_in_builder[0]])
                points_2d.append(corner)

            print(f"Size of intersection of current point cloud and current frame: {len(corner_ids)}")

            if len(corner_ids) < 5:
                continue

            points_3d = np.array(points_3d)
            points_2d = np.array(points_2d)

            retval, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d.reshape((-1, 1, 3)),
                                                             points_2d.reshape((-1, 1, 2)),
                                                             cameraMatrix=intrinsic_mat,
                                                             distCoeffs=None,
                                                             reprojectionError=PNP_REPROJ_ERROR)

            if not retval:
                print("Unsuccessful solution of PnP")
                continue

            print(f"Position found by {len(inliers)} inliers")

            view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            view_mats[i] = view_mat

            for j in range(num_frames):
                if i != j and view_mats[j] is not None:
                    points_added = add_points_to_cloud(point_cloud_builder,
                                        corner_storage[i],
                                        corner_storage[j],
                                        view_mats[i],
                                        view_mats[j],
                                        intrinsic_mat,
                                        triang_params)
                    if points_added > 0:
                        was_updated = True

            print(f"Current size of point cloud: {point_cloud_builder.points.size}")

        if was_updated is False:
            break

    for i in range(num_frames):
        if view_mats[i] is None:
            view_mats[i] = view_mats[i - 1]

    return view_mats, point_cloud_builder


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats, point_cloud_builder = track_and_find_point_cloud(
        intrinsic_mat,
        corner_storage,
        known_view_1,
        known_view_2
    )

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
