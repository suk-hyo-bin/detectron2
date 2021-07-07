"""
이미지를 다루는데 사용되는 함수입니다.
"""

import pickle
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from fvcore.common.file_io import PathManager


def get_patch_generator(
    image: np.ndarray, patch_size: int, overlay_size: int
) -> Tuple[List[np.ndarray], int, int]:
    """
    grid로 image를 분할하여 Patch Generator를 생성합니다.

    @param image: image
    @param patch_size: 높이와 넓이가 동일한 패치 크기
    @param overlay_size: overlay size
    @return: Patch Generator
    """
    step = patch_size - overlay_size
    row_range = (
        image.shape[0] - overlay_size if image.shape[0] - overlay_size > 0 else image.shape[0]
    )
    col_range = (
        image.shape[1] - overlay_size if image.shape[1] - overlay_size > 0 else image.shape[1]
    )
    patch_height = patch_size if image.shape[0] > patch_size else image.shape[0]
    patch_width = patch_size if image.shape[1] > patch_size else image.shape[1]

    for row in range(0, row_range, step):
        row = row if row + patch_height < image.shape[0] else image.shape[0] - patch_height
        for col in range(0, col_range, step):
            col = col if col + patch_width < image.shape[1] else image.shape[1] - patch_width

            # Set patch image
            patch_image = image[row : row + patch_height, col : col + patch_width]

            # Zero padding if patch image is smaller than patch size
            if patch_height < patch_size or patch_width < patch_size:
                pad_height = patch_size - patch_height
                pad_width = patch_size - patch_width
                patch_image = np.pad(
                    patch_image, ((0, pad_height), (0, pad_width), (0, 0)), "constant"
                )

            yield patch_image, row, col


def read_raw_image(file_name: str, format: Optional[str] = None) -> Any:
    """
    파일 이미지를 불러옵니다.

    @param file_name: 이미지 파일
    @param format: 이미지 형식
    @return: 이미지
    """
    with PathManager.open(file_name, "rb") as f:
        image = pickle.load(f)
    if format == "BGR":
        image = image[:, :, ::-1]

    return image


def check_minimum_range(min_val: int, max_val: int, min_range=256) -> Tuple[int, int]:
    """
    최소 범위에 해당하는 최솟값과 최댓값을 반환합니다.

    @param min_val: 범위를 축소시킬 값 중 최솟값
    @param max_val: 범위를 축소시킬 값 중 최댓값
    @param min_range: 축소시킬 범위
    @return: 최솟값과 최댓값
    """
    if max_val - min_val < min_range:
        margin = (min_range - (max_val - min_val)) / 2
        min_val -= margin
        max_val += margin

        if min_val < 0:
            max_val -= min_val
            min_val = 0
        if max_val > 2 ** 16:
            min_val -= 2 ** 16 - max_val
            max_val = 2 ** 16 - 1
    return min_val, max_val


def normalize_range(
    image: np.ndarray, range_max=1.0, min_percent=0, max_percent=100, dst_type=np.float32
) -> np.ndarray:
    """
    특정 범위로 normalization을 진행합니다.

    @param image: 이미지
    @param range_max: 범위 최댓값
    @param min_percent: 범위 최솟값 %
    @param max_percent: 범위 최댓값 %
    @param dst_type: 변환할 이미지 데이터 타입
    @return: 변환된 이미지
    """
    minimum_values = 10
    image = image.astype(dtype=np.float32)

    for idx in range(image.shape[2]):
        band = image[:, :, idx]
        filtered_band = band[band > minimum_values]

        # min_val = filtered_band.min() if filtered_band.any() else 0
        # max_val = filtered_band.max() if filtered_band.any() else 255
        min_val = np.percentile(band, min_percent) if filtered_band.any() else 0
        max_val = np.percentile(band, max_percent) if filtered_band.any() else 255

        min_val, max_val = check_minimum_range(min_val, max_val, min_range=256)

        cvt_range = max_val - min_val
        band = (band - min_val) / cvt_range * range_max
        band = np.clip(band, 0, range_max)
        image[:, :, idx] = band
    image = image.astype(dtype=dst_type)
    return image


def normalize_range_torch(image, range_max=1.0, min_percent=0, max_percent=1, dst_type=np.float32):
    minimum_values = 10
    image = torch.from_numpy(image.astype(dtype=np.float32)).to("cuda")
    for idx in range(image.shape[2]):
        band = image[:, :, idx]

        min_val = torch.quantile(band, min_percent) if torch.any(band > minimum_values) else 0
        max_val = torch.quantile(band, max_percent) if torch.any(band > minimum_values) else 255

        min_val, max_val = check_minimum_range(min_val, max_val, min_range=256)

        cvt_range = max_val - min_val
        band = (band - min_val) / cvt_range * range_max

        band = torch.clamp(band, 0, range_max)
        image[:, :, idx] = band
    image = image.cpu().numpy()
    return image