import rasterio
import geopandas as gpd
import numpy as np
import multiprocessing as mp

from pathlib import Path
from rasterio.windows import from_bounds
from tqdm import tqdm

from constants import BUFF_GRID_PATH, TRAIN_OUT_DIR


def process_year(year, jitter_paths, grid, profile, outpath):
    out_mean = rasterio.open(outpath / f"{year}_mean.tif", "w", **profile)
    out_var  = rasterio.open(outpath / f"{year}_var.tif",  "w", **profile)
    out_cvar  = rasterio.open(outpath / f"{year}_cvar.tif",  "w", **profile)

    srcs = [rasterio.open(p / f"{year}.tif") for p in jitter_paths]

    try:
        for cell in tqdm(grid.itertuples(), total=len(grid), desc=f"Processing {year}"):
            bounds = cell.geometry.bounds
            stack = []
            for src in srcs:
                try:
                    window = from_bounds(*bounds, transform=src.transform)
                    arr = src.read(window=window, boundless=True)
                    stack.append(arr)
                except Exception as e:
                    continue
            if len(stack) == 0:
                continue

            stack = np.stack(stack)
            mean = stack.mean(axis=0)
            var  = stack.var(axis=0)
            std = stack.std(axis=0)
            try:
                window = from_bounds(*bounds, transform=out_mean.transform)
                out_mean.write(mean, window=window)
                out_var.write(var, window=window)
                out_cvar.write((var-mean) / std , window=window)
            except Exception as e:
                continue
    finally:
        out_mean.close()
        out_var.close()
        out_cvar.close()
        for src in srcs:
            src.close()


def collate_jitters(outpath, jitter_dir, plot_grid=BUFF_GRID_PATH):
    outpath.mkdir(parents=True, exist_ok=True)

    jitter_paths = sorted(jitter_dir.glob('[0-9][0-9][0-9][0-9]'))
    years = [p.stem for p in jitter_paths[0].glob("*.tif")]
    grid = gpd.read_file(plot_grid)
    with rasterio.open(jitter_paths[0] / f"{years[0]}.tif") as src:
        profile = src.profile


    with mp.Pool(processes=min(len(years), 4)) as pool:
        pool.starmap(
            process_year,
            [(year, jitter_paths, grid, profile, outpath) for year in years]
        )
        

if __name__ == "__main__":
    jitter_dir = Path("/hudak_agb/data/outputs/upernet/val_rmse_231.0000_s09085_e1816/jitters")
    collate_jitters(jitter_dir / "collated",
                    jitter_dir)