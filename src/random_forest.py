import geopandas as gpd
import multiprocessing as mp
import numpy as np
import joblib
import rasterio

from rasterio.windows import from_bounds
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

from constants import (
    IMAGERY_DIR, UNITS_DIR, UNITS_PATH, TRAIN_OUT_DIR,
    PRISM_FILES, SOLUS_FILES,
    DIRS_YEARS, PRISM_ELEMENTS, PRISM_YEARS,
    EPSG_CODE, PIXEL_SIZE, CHIP_SIZE, PATCH_SIZE,
    NO_DATA, INF_NO_DATA,
)

from inference import GridUnitInfDataset, chip_writer
from utils import cache_anc, read_window, resample_to_shape


class PointTrainDataset(Dataset):
    def __init__(self, source='LS', dem='NASADEM', cache={}, imagery_dir=IMAGERY_DIR,
                 units_dir=UNITS_DIR,
                 units_path=UNITS_PATH,):
        super().__init__()

        self.source = source
        self.dem = dem
        self._cache = bool(cache)
        self._l_size = 1
        
        self.imagery_dir = imagery_dir
        self.dem_file = self.imagery_dir.glob(f"{self.dem}.tif").__next__()
        self.prism_dat = cache["prism_dat"]
        self.prism_su_dat = cache["prism_su_dat"]
        self.solus_dat = cache["solus_dat"]
        self.patch_transform = cache["patch_transform"]

        self.years = list(DIRS_YEARS[self.source])
        self.units = gpd.read_file(units_path)
        self.units = self.units[self.units['LidarYear'].isin(self.years)]
        self.unit_files = {u.with_suffix("").name.replace("-", "_"): u for u in units_dir.glob("*.tif") if "StdDev" not in u.name}

        self.imagery_fh = {}
        for year in self.units['LidarYear'].unique():
            self.imagery_fh[year] = rasterio.open(self.imagery_dir / f"{self.source}{int(year)}.tif")
        self.dem_fh = rasterio.open(self.dem_file)
        
        self.binned_pixels = {}
        self._transforms = {}

        for name, ufile in tqdm(self.unit_files.items(), desc="Caching unit files"):
            year = int(self.units[self.units['LidarUnit'] == name]['LidarYear'].item())
            with rasterio.open(ufile) as src:
                data = src.read()[0]
                nodata = src.nodata
                self._transforms[name] = src.transform

            valid = data != nodata if nodata is not None else np.isfinite(data)
            valid &= data > 10
            rows, cols = np.where(valid)
            values = data[rows, cols]

            bin_ids = (values // 10).astype(int)

            order = np.argsort(bin_ids)
            sorted_bins = bin_ids[order]
            sorted_rows = rows[order]
            sorted_cols = cols[order]
            sorted_values = values[order]
            sorted_years = np.repeat(year, len(sorted_values))

            unique_bins, split_at = np.unique(sorted_bins, return_index=True)
            split_rows = np.split(sorted_rows, split_at[1:])
            split_cols = np.split(sorted_cols, split_at[1:])
            split_values = np.split(sorted_values, split_at[1:])
            split_years = np.split(sorted_years, split_at[1:])

            self.binned_pixels[name] = {
                int(b): np.column_stack((r, c, v, y))
                for b, r, c, v, y in zip(unique_bins, split_rows, split_cols, split_values, split_years)
            }

        self.sampled_pixels = self._subsample_pooled_bins(n=500)
    
    def __len__(self):
        return len(self.sampled_pixels)
    
    def _get_cache(self, chip_bounds, year):
        win = from_bounds(*chip_bounds, self.patch_transform)
        col_off, row_off = int(win.col_off), int(win.row_off)

        out = {}
        p = []
        p_su = []
        for e in PRISM_ELEMENTS:
            E = e.upper()
            Y = int(year-PRISM_YEARS[0])
            p.append(self.prism_dat[E][:, row_off, col_off].copy())
            p_su.append(self.prism_su_dat[E][Y:Y+1, row_off, col_off].copy())
        out["PRISM"] = np.concatenate(p)
        out["PRISM_SU"] = np.concatenate(p_su)
        out["SOLUS100"] = self.solus_dat[:, row_off, col_off]
        return out

    def _subsample_pooled_bins(self, n=500, seed=None):
        rng = np.random.default_rng(seed)

        pooled = {}
        for fname, bins in self.binned_pixels.items():
            for b, rcvy in bins.items():
                entry = pooled.setdefault(b, [])
                tagged = np.column_stack((rcvy, np.full(len(rcvy), fname)))
                entry.append(tagged)

        sampled = []
        for b, chunks in pooled.items():
            all_rcvy = np.concatenate(chunks, axis=0)
            total = len(all_rcvy)

            target = n if total >= n else total // 2
            if target == 0:
                continue

            idx = rng.choice(total, size=target, replace=False)
            chosen = all_rcvy[idx]

            xs = np.empty(len(chosen), dtype=float)
            ys = np.empty(len(chosen), dtype=float)
            for fname in np.unique(chosen[:, 4]):
                mask = chosen[:, 4] == fname
                rows = chosen[mask, 0].astype(float)
                cols = chosen[mask, 1].astype(float)
                x, y = rasterio.transform.xy(self._transforms[fname], rows, cols)
                xs[mask] = x
                ys[mask] = y

            values = chosen[:, 2].astype(float)
            years = chosen[:, 3].astype(int)

            sampled.append(np.array(
                list(zip(xs, ys, values, years, [b] * len(chosen))),
                dtype=[('x', 'f8'), ('y', 'f8'), ('value', 'f8'), ('year', 'i4'), ('bin', 'i4')]
            ))

        return np.concatenate(sampled)
    
    def _get_path_bounds_year(self, idx):
        sample = self.sampled_pixels[idx]
        x, y, value, year = sample['x'], sample['y'], sample['value'], sample['year']
        bounds = x, y-(self._l_size*PIXEL_SIZE), x+(self._l_size*PIXEL_SIZE), y

        return value, bounds, year

    def __getitem__(self, idx):
        y, bounds, year = self._get_path_bounds_year(idx)
        out = {}
        out[self.source] = read_window(None, bounds, dtype=np.float32, tensorize=False, src_pre=self.imagery_fh[year]).squeeze(axis=(-2,-1))
        out[self.dem] = read_window(None, bounds, 1, dtype=np.float32, tensorize=False, src_pre=self.dem_fh).squeeze(axis=(-2,-1))
        
        out.update(self._get_cache(bounds, year))

        return np.concatenate(list(out.values())), y
    
    def close_fg(self):
        for fh in self.imagery_fh.values():
            fh.close()
        self.dem_fh.close()


def _gather_training_pool(cache, config):
    X_list, y_list = [], []
    ds = PointTrainDataset(config['img'], config['dem'], cache=cache)
    for idx in tqdm(range(len(ds)), desc="loading unit pixels", total=len(ds)):
        X, y = ds[idx]
        X_list.append(X)
        y_list.append(y)
    ds.close_fg()
    X_all = np.stack(X_list)
    y_all = np.array(y_list)
    return X_all, y_all


def build_rf_model(cache, config, n_estimators=500, random_state=42, out_dir=None):
    X_train, y_train = _gather_training_pool(cache, config)
    X_train[np.isnan(X_train)] = NO_DATA

    print(f"Training RF on {X_train.shape[0]} pixels, {X_train.shape[1]} features")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=19,
    )
    model.fit(X_train, y_train)

    if out_dir is not None:
        joblib.dump(model, out_dir / "rf_model.joblib")

    return model


def run_rf_model(model, cache, pred_dir):
    profile = {
        'count': 1,
        'width': cache["width"],
        'height': cache["height"],
        "driver": "GTiff",
        'transform': cache['transform'],
        'BIGTIFF': 'YES',
        'tiled': True,
        'blockxsize': CHIP_SIZE,
        'blockysize': CHIP_SIZE,
        'crs': EPSG_CODE,
        'nodata': INF_NO_DATA,
        'compress': 'lzw',
        'dtype': "uint16"
    }
    ds = GridUnitInfDataset(cache=cache, dtype=np.float32, tensorize=False)
    q = mp.Queue(maxsize=256)
    writer_proc = mp.Process(
        target=chip_writer,
        args=(q, pred_dir, ds.grid, ds.years, profile),
    )
    writer_proc.start()
    
    for idx in tqdm(range(len(ds)), desc="loading unit pixels"):
        X = ds[idx]
        target_shape = X[ds.source].shape[1:] 
        
        grid_idx = np.array([[X['GRID_IDX']]])
        years = np.array([[X["YEAR"]]])
        
        X["PRISM"] = resample_to_shape(X["PRISM"], target_shape)
        X["PRISM_SU"] = resample_to_shape(X["PRISM_SU"], target_shape)
        X["SOLUS100"] = resample_to_shape(X["SOLUS100"], target_shape)

        X_ = np.concatenate([X[ds.source], X[ds.dem], X["PRISM"], X["PRISM_SU"], X["SOLUS100"]])
        X_[np.isnan(X_)] = NO_DATA
        X_ = X_.reshape((X_.shape[0], -1))
        X_ = np.transpose(X_, axes=(1,0))
        preds = model.predict(X_).reshape(1, 1, CHIP_SIZE, CHIP_SIZE)
        
        q.put((grid_idx, years, preds))
    q.put(None)
    writer_proc.join()

                
def main(config):
    _OUT_DIR = TRAIN_OUT_DIR / config["name"]
    _CKPT_DIR = _OUT_DIR / 'ckpt'
    _PRED_DIR = _OUT_DIR / 'pred'

    _OUT_DIR.mkdir(exist_ok=True, parents=True)
    _CKPT_DIR.mkdir(exist_ok=True)
    _PRED_DIR.mkdir(exist_ok=True)


    cache = cache_anc(IMAGERY_DIR, config['img'], DIRS_YEARS[config['img']], PATCH_SIZE, PRISM_ELEMENTS, PRISM_FILES, SOLUS_FILES, np.float32, tensorize=False)

    model = build_rf_model(
        cache, config,
        n_estimators=500,
        random_state=42,
        out_dir=_CKPT_DIR,
    )

    run_rf_model(model, cache, _PRED_DIR)
    

if __name__ == "__main__":
    config = {  "name": 'random_forest',
                "img": "LS",
                "dem": "NASADEM"}
    main(config)