import geopandas as gpd
import ee
import numpy as np
import os
import rasterio
import requests
import re
import yaml
import zipfile

from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from rasterio.windows import from_bounds as win_from_bounds
from rasterio.transform import from_bounds as tran_from_bounds
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
from urllib.parse import urljoin

from constants import EE_PROJECT, EPSG_CODE, CHIP_SIZE, PARALLELLISM, PIXEL_SIZE, SCALE
from constants import FISHNET_GRID, IMAGERY_DIR, PRISM_RPR_DIR, PRISM_ZIP_DIR, SOLUS_DIR, STAT_PATH
from constants import SENSOR_DIRS, SENSOR_YEARS, PRISM_ELEMENTS, PRISM_YEARS
from constants import COMPOSITE_DATE_START, COMPOSITE_DATE_END, LBANDS, SBANDS, BANDS


ee.Initialize(project=EE_PROJECT)


def get_ls_combined_sr_collection():
    lt5 = get_ls_sr_collection('LT05')
    le7 = get_ls_sr_collection('LE07')
    # SLC-OFF for Landsat 7 after 2003-05-31
    le7 = le7.filter(ee.Filter.And([
        ee.Filter.Or([ee.Filter.lte('system:time_start', 1054425600000),
                      ee.Filter.gte('system:time_start', 1336176000000)]),
        ee.Filter.lte('system:time_start', 1363564800000)
    ]))
    lc8 = get_ls_sr_collection('LC08')
    lc9 = get_ls_sr_collection('LC09')
    return lt5.merge(le7).merge(lc8).merge(lc9)


def get_ls_sr_collection(sensor: str):
    ls_col = ee.ImageCollection(f'LANDSAT/{sensor}/C02/T1_L2')
    ls_col = ls_col.map(lambda image: preprocess_ls_image(image, sensor))
    return ls_col


def preprocess_ls_image(image: ee.Image, sensor: str):
    qa = image.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    mask = cloud.multiply(shadow)
    image = image.updateMask(mask)

    if sensor == 'LC08' or sensor == 'LC09':
        image = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
                            BANDS)
    else:
        image = image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
                            BANDS)
    return image


def get_hls_combined_sr_collection():
    hlsl = get_hls_sr_collection('HLSL30')
    hlss = get_hls_sr_collection('HLSS30')
    return hlsl.merge(hlss)


def get_hls_sr_collection(sensor):
    hls_col = ee.ImageCollection(f"NASA/HLS/{sensor}/v002")
    hls_col = hls_col.map(lambda image: preprocess_hls_image(image, sensor))
    return hls_col


def preprocess_hls_image(image: ee.Image, sensor: str):
    fmask = image.select('Fmask')
    cloud  = fmask.bitwiseAnd(1 << 1).eq(0)
    shadow  = fmask.bitwiseAnd(1 << 3).eq(0)
    mask = cloud.multiply(shadow)
    image = image.updateMask(mask)
    
    if sensor == 'HLSS30':
        image = image.select(SBANDS, LBANDS).select(BANDS)
    else:
        image = image.select(BANDS)

    return image


def process_cell(cell, chips_path, collection, years=None):
    idx = cell.fid
    outpath = chips_path / f'{idx:05d}.tif'
    if outpath.exists():
        return
    
    coords = list(cell.geometry.exterior.coords)
    geo = ee.Geometry.Polygon(coords, EPSG_CODE, False)
    top_left = min(coords, key=lambda p: (p[0], -p[1]))

    payload = {'fileFormat': 'GEO_TIFF',
                'grid': {
                    'dimensions': {
                        'width': CHIP_SIZE,
                        'height': CHIP_SIZE
                    },
                    'affineTransform': {
                        'scaleX': PIXEL_SIZE,
                        'shearX': 0,
                        'translateX': top_left[0],
                        'shearY': 0,
                        'scaleY': -PIXEL_SIZE,
                        'translateY': top_left[1]
                    },
                    'crsCode': EPSG_CODE,
                }}
    try:
        if chips_path.name in ["GLO30", "NASADEM"]:
            payload["expression"] = collection
        else:
            expression = []
            ybands = [f"{i}_{B}" for i in range(len(years)) for B in BANDS]
            dbands = [f"{y}_{B}" for y in years for B in BANDS]
            for y in years:
                start_date = COMPOSITE_DATE_START.replace(year=y)
                end_date = COMPOSITE_DATE_END.replace(year=y)
                img = collection\
                    .filterBounds(geo)\
                    .filter(ee.Filter.date(start_date, end_date))\
                    .median()
                if chips_path.name == "LS":
                    img = img.multiply(0.0000275).add(-0.2)
                img = img.divide(SCALE).toUint16()
                expression.append(img)
            expression = ee.ImageCollection(expression).toBands()
            expression = expression.select(ybands, dbands)
            payload["expression"] = expression
            payload["bandIds"] = dbands
        dnld = ee.data.computePixels(payload)
        with open(outpath, 'wb') as f:
            f.write(dnld)
    except Exception as e:
        print(outpath.name, e)


def fuse_tifs(chips_path, out_path, grid, years=None, position=None):
    name = chips_path.name
    chips = list(chips_path.glob("*.tif"))

    with rasterio.open(chips[0]) as src:
        res = src.res
        count = src.count
        profile = src.profile.copy()
    bounds = rasterio.coords.BoundingBox(
        left=min(grid['left']),
        bottom=min(grid['bottom']),
        right=max(grid['right']),
        top=max(grid['top']),
    )

    width = int((bounds.right - bounds.left) / res[0])
    height = int((bounds.top - bounds.bottom) / res[1])

    transform = tran_from_bounds(*bounds, width, height)
    profile.update({
        'width': width,
        'height': height,
        'transform': transform,
        'BIGTIFF': 'YES',
        'tiled': True,
        'blockxsize': CHIP_SIZE,
        'blockysize': CHIP_SIZE,
        'compress': 'lzw'
    })
    
    means = []
    stddv = []
    if years is not None:
        count = len(BANDS)
        profile['count'] = count
        dsts = [rasterio.open(f"{out_path}/{name}{y}.tif", "w", **profile) for y in years]
        for i, p in enumerate(chips):
            if i % 100 == 0 or i == len(chips)-1:
                print(f"{name} - {i} loading {p.name}")
            with rasterio.open(p) as src:
                data = src.read()
                for i, d in enumerate(dsts):
                    win = win_from_bounds(*src.bounds, transform=transform)
                    d.write(data[6*i:6*(i+1)], window=win)

                    means.append(np.mean(data[6*i:6*(i+1)], axis=(1,2)))
                    stddv.append(np.std(data[6*i:6*(i+1)], axis=(1,2)))
        for d in dsts:
            d.close()
    else:
        profile['count'] = count
        dst = rasterio.open(f"{out_path}/{name}.tif", "w", **profile)
        for i, p in enumerate(chips):
            if i % 100 == 0 or i == len(chips)-1:
                print(f"{name} - {i} loading {p.name}")
            with rasterio.open(p) as src:
                data = src.read()
                win = win_from_bounds(*src.bounds, transform=transform)
                dst.write(data, window=win)
                
                means.append(np.mean(data, axis=(1,2)))
                stddv.append(np.std(data, axis=(1,2)))
        dst.close()
    means_arr = np.array(means)
    stddv_arr = np.array(stddv)

    global_means = means_arr.mean(axis=0)
    global_stds = np.sqrt(np.mean(stddv_arr**2 + (means_arr - global_means)**2, axis=0))
    return global_means, global_stds
  
  
def download_prism():
    BASE_URL = "https://services.nacse.org/prism/data/get"
    REGION = "us"
    RES = "800m"

    session = requests.Session()

    for element in PRISM_ELEMENTS:
        for year in PRISM_YEARS:
            for month in range(1, 13):
                date = f"{year}{month:02d}"
                url = f"{BASE_URL}/{REGION}/{RES}/{element}/{date}/lt"

                r = session.get(url, stream=True, timeout=60)
                r.raise_for_status()

                cd = r.headers.get("content-disposition", "")
                filename = cd.split("filename=")[-1].strip("\"") or f"{element}_{date}.zip"
                outpath = PRISM_ZIP_DIR / filename

                with open(outpath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
    

def process_prism_helper(series, element, position=None):
    series.sort(key=lambda x: x[0])
    meta = None
    
    for ym, zpath in tqdm(series, desc=f"{element}", total=len(series), position=position):
        tif_name = zpath.stem + ".tif"

        with zipfile.ZipFile(zpath) as zf:
            with zf.open(tif_name) as tif_bytes:
                out = PRISM_RPR_DIR / f"prism_{element}_{ym}.tif"
                with MemoryFile(tif_bytes.read()) as mem:
                    with mem.open() as src:
                        gtransform, width, height = calculate_default_transform(
                            src.crs, EPSG_CODE, src.width, src.height, *src.bounds
                        )
                        if meta is None:
                            meta = src.meta.copy()
                            meta.update({
                                "count": 1,
                                "driver": "GTiff",
                                "BIGTIFF": "YES",
                                "transform": gtransform,
                                "height": height,
                                "width": width,
                                "crs": EPSG_CODE,
                            })
                            print(meta)

                        with rasterio.open(out, "w", **meta) as dst:
                            reproject(
                                source=rasterio.band(src, 1),
                                destination=rasterio.band(dst, 1),
                                resampling=Resampling.nearest,
                            )

def process_prism():
    pattern = re.compile(
        r"prism_(?P<e>\w+)_us_30s_(?P<ym>\d{6})\.zip"
    )
    by = {}
    for z in PRISM_ZIP_DIR.glob("prism_*.zip"):
        m = pattern.match(z.name)
        if not m:
            continue
        e = m.group("e")
        ym = m.group("ym")
        
        by.setdefault(e, [])
        by[e].append((ym, z))
    with ThreadPoolExecutor(max_workers=PARALLELLISM) as ex:
        for i, (element, series) in enumerate(by.items()):
            ex.submit(process_prism_helper, series, element, i)


def download_solus100():
    index_url = "https://storage.googleapis.com/solus100pub/index.html"  # index page URL
    SOLUS_DIR.mkdir(exist_ok=True)

    resp = requests.get(index_url, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    pattern = re.compile(r"_p\.tif$", re.IGNORECASE)

    for a in tqdm(soup.find_all("a", href=True)):
        href = a["href"]
        if pattern.search(href):
            url = urljoin(index_url, href)
            fname = os.path.join(SOLUS_DIR, os.path.basename(href))

            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()

            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


def main():
    SOLUS_DIR.mkdir(exist_ok=True)
    download_solus100()
    
    PRISM_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    PRISM_RPR_DIR.mkdir(parents=True, exist_ok=True)
    download_prism()
    process_prism()

    IMAGERY_DIR.mkdir(exist_ok=True)
    grid = gpd.read_file(FISHNET_GRID)
    stat = {}
    fuse_futures = []

    _LS = get_ls_combined_sr_collection()
    _HLSL = get_hls_sr_collection("HLSL30")
    _HLSS = get_hls_sr_collection("HLSS30")
    _HLS = get_hls_combined_sr_collection()
    _GLO30 = ee.ImageCollection('COPERNICUS/DEM/GLO30').mosaic()
    _NASADEM = ee.Image("NASA/NASADEM_HGT/001")

    IMAGERY = [_LS, _HLSL, _HLSS, _HLS, _GLO30, _NASADEM] 
    
    # IMAGERY = [_LS]
    # SENSOR_DIRS = [Path('/home/server/pi/homes/truongmy/hudak_agb/LS')]
    # SENSOR_YEARS = [range(2000, 2026)]
    
    with ThreadPoolExecutor(max_workers=PARALLELLISM) as ex:
        for position, (chips_path, years, collection) in enumerate(zip(SENSOR_DIRS, SENSOR_YEARS, IMAGERY)):
            chips_path.mkdir(exist_ok=True)

            list(
                tqdm(
                    ex.map(
                        lambda r: process_cell(r, chips_path, collection, years),
                        grid.itertuples(index=False),
                    ),
                    total=len(grid),
                    desc=f"process_cell_{name}"
                )
            )

            fut = ex.submit(fuse_tifs, chips_path, IMAGERY_DIR, years, position)
            fuse_futures.append((chips_path.name, fut))

        for name, fut in fuse_futures:
            global_means, global_stds = fut.result()
            stat[f"MEANS_{name}"] = global_means.tolist()
            stat[f"STDDV_{name}"] = global_stds.tolist()

    with open(STAT_PATH, "w") as outfile:
        yaml.dump(stat, outfile)
        

if __name__ == "__main__":
    main()