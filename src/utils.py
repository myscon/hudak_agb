import math
import shapely
import random
import rasterio
import torch
import torch.nn.functional as F

from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import Point, box
from tqdm import tqdm

        
def crop_and_pad(tensor, col_off, row_off, sh, sw):
    _, H, W = tensor.shape

    col_end = col_off + sw
    row_end = row_off + sh

    col_start_clamped = max(col_off, 0)
    row_start_clamped = max(row_off, 0)
    col_end_clamped = min(col_end, W)
    row_end_clamped = min(row_end, H)

    crop = tensor[:, row_start_clamped:row_end_clamped,
                  col_start_clamped:col_end_clamped]

    pad_left   = col_start_clamped - col_off
    pad_top    = row_start_clamped - row_off
    pad_right  = col_end - col_end_clamped
    pad_bottom = row_end - row_end_clamped

    if pad_left == pad_top == pad_right == pad_bottom == 0:
        return crop

    return F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom),
                 value=torch.nan)


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


def read_window(src_path, chip_bounds, indexes=None, out_shape=None, dtype=torch.float, resampling=Resampling.bilinear):
    with rasterio.open(src_path) as src:
        win = from_bounds(*chip_bounds, transform=src.transform)
        dat = src.read(indexes=indexes, window=win, boundless=True, out_shape=out_shape, fill_value=src.nodata, resampling=resampling)
        dat = torch.tensor(dat).clone().to(dtype)
        dat = torch.where(dat == src.nodata, torch.nan, dat)
        if indexes is not None:
            dat = dat.unsqueeze(0)
        return dat


def cache_anc(imagery_dir, source, years, patch_size, prism_elements, prism_files, solus_files, dtype):
    with rasterio.open(imagery_dir / f"{source}{years[0]}.tif") as src:
        col_max_pad = math.ceil(src.width / patch_size) * patch_size
        row_max_pad = math.ceil(src.height / patch_size) * patch_size

        right_pad, bottom_pad = src.transform * (col_max_pad, row_max_pad)
        bounds = (src.bounds.left, bottom_pad, right_pad, src.bounds.top)
        patch_transform = src.transform * Affine.scale(patch_size, patch_size)
        _shw = (row_max_pad // patch_size, col_max_pad // patch_size)
        
    prism_dat = {}
    for e in prism_elements:
        E = e.upper()
        prism_dat[E] = []
        for f in tqdm(prism_files[e], desc=f"caching prism {E}"):
            prism_dat[E].append(read_window(f, bounds, out_shape=_shw, dtype=dtype))
        prism_dat[E] = torch.concat(prism_dat[E])

    solus_dat = []
    for f in tqdm(solus_files, desc=f"caching solus"):
        solus_dat.append(read_window(f, bounds, out_shape=_shw, dtype=dtype))
    solus_dat = torch.concat(solus_dat)
    return {"prism_dat": prism_dat, "solus_dat": solus_dat, "patch_transform": patch_transform, "height": row_max_pad, "width": col_max_pad}
