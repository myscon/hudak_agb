import math
import numpy as np
import shapely
import random
import rasterio
import torch
import torch.nn.functional as F

from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.ndimage import zoom
from shapely.geometry import Point, box
from tqdm import tqdm

        
def crop_and_pad(array, col_off, row_off, sh, sw):
    _, H, W = array.shape

    col_end = col_off + sw
    row_end = row_off + sh

    col_start_clamped = max(col_off, 0)
    row_start_clamped = max(row_off, 0)
    col_end_clamped = min(col_end, W)
    row_end_clamped = min(row_end, H)

    crop = array[:, row_start_clamped:row_end_clamped,
                  col_start_clamped:col_end_clamped]

    pad_left   = col_start_clamped - col_off
    pad_top    = row_start_clamped - row_off
    pad_right  = col_end - col_end_clamped
    pad_bottom = row_end - row_end_clamped

    if pad_left == pad_top == pad_right == pad_bottom == 0:
        return crop

    if isinstance(array, np.ndarray):
        return np.pad(
            crop,
            pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=np.nan,
        )
    elif isinstance(array, torch.Tensor):
        return F.pad(crop,
            (pad_left, pad_right, pad_top, pad_bottom),
            value=torch.nan)
    else:
        raise TypeError


def random_point_in_polygon(poly):
    minx, miny, maxx, maxy = poly.bounds
    while True:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        if shapely.intersects_xy(p, x, y):
            return p


def random_box_crop(poly, size):
    half = size // 2
    while True:
        tl = random_point_in_polygon(poly)
        x0, y0 = tl.x, tl.y
        bounds = x0 - half, y0 - half, x0 + half, y0 + half
        crop = box(*bounds)
        if shapely.intersects(poly, crop):
            return bounds
      
  
def _read_window(src, chip_bounds, indexes, out_shape, dtype, resampling, tensorize):
    win = from_bounds(*chip_bounds, transform=src.transform)
    dat = src.read(indexes=indexes, window=win, boundless=True, out_shape=out_shape, fill_value=src.nodata, resampling=resampling)
    if tensorize:
        dat = torch.tensor(dat).clone().to(dtype)
        dat = torch.where(dat == src.nodata, torch.nan, dat)
        if indexes is not None:
            dat = dat.unsqueeze(0)
    else:
        dat = dat.astype(dtype)
        dat = np.where(dat == src.nodata, np.nan, dat)
        if indexes is not None:
            dat = np.expand_dims(dat, axis=0)
    return dat


def read_window(src_path, chip_bounds, indexes=None, out_shape=None, dtype=torch.bfloat16, resampling=Resampling.bilinear, tensorize=True, src_pre=None):
    if src_pre is None:
        with rasterio.open(src_path) as src:
            dat = _read_window(src, chip_bounds, indexes, out_shape, dtype, resampling, tensorize)
    else:
        dat = _read_window(src_pre, chip_bounds, indexes, out_shape, dtype, resampling, tensorize)
    return dat


def cache_anc(imagery_dir, source, years, patch_size, prism_elements, prism_files, solus_files, dtype, tensorize=True):
    with rasterio.open(imagery_dir / f"{source}{years[0]}.tif") as src:
        col_max_pad = math.ceil(src.width / patch_size) * patch_size
        row_max_pad = math.ceil(src.height / patch_size) * patch_size

        transform = src.transform
        right_pad, bottom_pad = transform * (col_max_pad, row_max_pad)
        bounds = (src.bounds.left, bottom_pad, right_pad, src.bounds.top)
        patch_transform = transform * Affine.scale(patch_size, patch_size)
        _shw = (row_max_pad // patch_size, col_max_pad // patch_size)
        
    prism_dat = {}
    prism_su_dat = {}
    for e in prism_elements:
        E = e.upper()
        dat = []
        su_dat = []
        su_arrs = []
        for i, f in tqdm(enumerate(prism_files[e]), desc=f"caching prism {E}", total=len(prism_files[e])):
            arr = read_window(f, bounds, out_shape=_shw, dtype=dtype, tensorize=tensorize)
            dat.append(arr)
            if i % 12 > 5 and i % 12 <8:
                su_arrs.append(arr)
            if len(su_arrs) == 2:
                if tensorize:
                    su_arrs = torch.concat(su_arrs)
                    su_arrs = torch.mean(su_arrs, dim=0)
                else:
                    su_arrs = np.concatenate(su_arrs)
                    su_arrs = np.mean(su_arrs, axis=0)
                su_dat.append(su_arrs)
                su_arrs = []
        if tensorize:
            dat = torch.concat(dat)
            prism_dat[E] = torch.mean(dat, dim=0, keepdim=True)
            prism_su_dat[E] = torch.stack(su_dat)
        else:
            dat = np.concat(dat)
            prism_dat[E] = np.mean(dat, axis=0, keepdims=True)
            prism_su_dat[E] = np.stack(su_dat)

    solus_dat = []
    for f in tqdm(solus_files, desc=f"caching solus"):
        solus_dat.append(read_window(f, bounds, out_shape=_shw, dtype=dtype, tensorize=tensorize))
    if tensorize:
        solus_dat = torch.concat(solus_dat)
    else:
        solus_dat = np.concatenate(solus_dat)
    return {"prism_dat": prism_dat, "prism_su_dat": prism_su_dat, "solus_dat": solus_dat, "transform": transform, "patch_transform": patch_transform, "height": row_max_pad, "width": col_max_pad}


def resample_to_shape(arr, target_shape):
    zoom_factors = (1, target_shape[0] / arr.shape[1], target_shape[1] / arr.shape[2])
    return zoom(arr, zoom_factors, order=1)
