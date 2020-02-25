#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def to_uint8_image(img):
    img = img * 255.0
    img = np.round(img)
    return img.astype(np.uint8)


class CornerTracker:
    MAX_CORNERS = 1300
    INITIAL_QUALITY_LEVEL = 0.03
    QUALITY_LEVEL = 0.15
    MIN_DISTANCE = 6
    BLOCK_SIZE = 5
    CIRCLE_SIZE = 14
    MAX_LEVEL_LK = 2
    TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    def __init__(self):
        self.total_corners = 0

    def get_circles_mask(self, shape, points):
        mask = np.full(shape, 255, dtype=np.uint8)
        for x, y in points:
            cv2.circle(mask,
                       center=(x, y),
                       radius=self.MIN_DISTANCE,
                       color=0,
                       thickness=-1)
        return mask

    def find_new_corners(self, img, num_corners=MAX_CORNERS, mask=None, quality_level=INITIAL_QUALITY_LEVEL):
        points = cv2.goodFeaturesToTrack(img,
                                        mask=mask,
                                        maxCorners=num_corners,
                                        qualityLevel=quality_level,
                                        minDistance=self.MIN_DISTANCE,
                                        blockSize=self.BLOCK_SIZE)
        if points is None:
            return None, None

        num_points = points.shape[0]
        sizes = np.array([self.CIRCLE_SIZE for _ in range(num_points)])
        return points, sizes

    def get_corners(self, new_img, old_img = None, old_corners=None):
        if old_img is None:
            points, sizes = self.find_new_corners(new_img)
            ids = np.arange(len(points))
            points = points.reshape((-1, 2))
            self.total_corners = len(points)
            return FrameCorners(ids, points, sizes)
        else:
            ids = old_corners.ids
            points = old_corners.points
            sizes = old_corners.sizes

            nextPts, status, err = cv2.calcOpticalFlowPyrLK(to_uint8_image(old_img),
                                                            to_uint8_image(new_img),
                                                            prevPts=points,
                                                            nextPts=None,
                                                            winSize=(self.CIRCLE_SIZE, self.CIRCLE_SIZE),
                                                            maxLevel=self.MAX_LEVEL_LK,
                                                            criteria=self.TERM_CRITERIA)

            status = status.squeeze()
            found = np.where(status == 1)

            ids = ids[found]
            points = nextPts[found]
            sizes = sizes[found]

            mask = self.get_circles_mask(new_img.shape, points)
            if len(points) < self.MAX_CORNERS:
                new_points, new_sizes = self.find_new_corners(new_img,
                                                              self.MAX_CORNERS - len(points),
                                                              mask,
                                                              self.QUALITY_LEVEL)
                if new_points is not None:
                    new_ids = np.arange(self.total_corners, self.total_corners + len(new_points))
                    new_ids = new_ids.reshape((-1, 1))
                    new_points = new_points.reshape((-1, 2))
                    new_sizes = new_sizes.reshape((-1, 1))
                    self.total_corners += len(new_points)
                    ids = np.concatenate([ids, new_ids])
                    points = np.concatenate([points, new_points])
                    sizes = np.concatenate([sizes, new_sizes])

            points = points.reshape((-1, 2))
            return FrameCorners(ids, points, sizes)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    cornerTracker = CornerTracker()

    image_0 = frame_sequence[0]
    corners = cornerTracker.get_corners(image_0)
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        corners = cornerTracker.get_corners(image_1, image_0, corners)
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.
    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)

    corner_storage = builder.build_corner_storage()
    final_storage = without_short_tracks(corner_storage, min_len=20)

    return final_storage


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter