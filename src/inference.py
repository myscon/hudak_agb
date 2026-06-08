import copy
import geopandas as gpd
import numpy as np
import torch
import rasterio
import torch.multiprocessing as mp
import yaml

from affine import Affine
from rasterio.windows import from_bounds
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from constants import IMAGERY_DIR, PRISM_RPR_DIR, PRISM_FILES, SOLUS_DIR, SOLUS_FILES, STAT_PATH
from constants import DIRS_YEARS, PRISM_ELEMENTS, PRISM_YEARS
from constants import TRAIN_OUT_DIR
from constants import CHIP_SIZE, DEVICE, DTYPE, EPSG_CODE, L_SIZE, NO_DATA, INF_NO_DATA, PATCH_SIZE, PIXEL_SIZE
from constants import PLOT_GRID, FISHNET_GRID, CLIP_GRID
from constants import CLIP_SIZE, NUM_JITTERS, BATCH_SIZE

from train import PrithviUPerNet, AttnSETRPUP
from utils import read_window, crop_and_pad, cache_anc


class GridUnitInfDataset(Dataset):
    def __init__(self, source='LS', dem='NASADEM', cache=None, stat={},
                 imagery_dir=IMAGERY_DIR,
                 prism_dir=PRISM_RPR_DIR,
                 solus_dir=SOLUS_DIR,
                 grid_path=PLOT_GRID):
        self.imagery_dir = imagery_dir
        self.prism_dir = prism_dir
        self.solus_dir = solus_dir
        self.source = source
        self.dem = dem
        self.cache = cache
        self.stat = stat
        
        self.ablate = None
        self.jittered = None
        
        self.years = list(DIRS_YEARS[self.source])
        self.set_grid(grid_path)
        
        self.dem_file = self.imagery_dir.glob(f"{self.dem}.tif").__next__()
        self.prism_dat = cache["prism_dat"]
        self.solus_dat = cache["solus_dat"]
        self.patch_transform = cache["patch_transform"]

    def __len__(self):
        return len(self.grid)*len(self.years)

    def __get_cache__(self, chip_bounds, grid_cell_year):
        win = from_bounds(*chip_bounds, self.patch_transform)
        col_off, row_off = int(win.col_off), int(win.row_off)

        out = {}
        if self.ablate == "SOLUS100":
            out["SOLUS100"] = torch.normal(mean=self.stat[f"MEANS_SOLUS100"].expand(-1, L_SIZE, L_SIZE), std=self.stat[f"STDDV_SOLUS100"].expand(-1, L_SIZE, L_SIZE))
        else:
            out["SOLUS100"] = crop_and_pad(self.solus_dat, col_off, row_off, L_SIZE, L_SIZE).clone()
        
        ancp = []
        Y = int((grid_cell_year-PRISM_YEARS[0])*12)
        YM = slice(Y+6,Y+8)
        for e in PRISM_ELEMENTS:
            E = e.upper()
            pname = f"PRISM_{E}"
            prism = crop_and_pad(self.prism_dat[E], col_off, row_off, L_SIZE, L_SIZE).clone()
            if self.ablate == pname or self.ablate == "PRISM":
                out[pname] = torch.normal(mean=self.stat[f"MEANS_{pname}"].expand(360, L_SIZE, L_SIZE), std=self.stat[f"STDDV_{pname}"].expand(360, L_SIZE, L_SIZE))
                ancp.append(torch.mean(prism[YM], dim=0, keepdim=True))
            else:
                out[pname] = prism
                if self.ablate == "PRISM__":
                    ancp.append(torch.normal(mean=self.stat[f"MEANS_{pname}"].expand(-1, L_SIZE, L_SIZE), std=self.stat[f"STDDV_{pname}"].expand(-1, L_SIZE, L_SIZE)))
                else:
                    ancp.append(torch.mean(prism[YM], dim=0, keepdim=True))

        out["PRISM__"] = torch.concat(ancp)
        return out

    def set_ablate(self, ablate):
        self.ablate = ablate

    def set_grid(self, grid_path):
        self.grid_path = grid_path
        self.grid = gpd.read_file(grid_path)
    
    def jitter(self):
        rng = np.random.default_rng()
        dy = rng.integers(low=-(CHIP_SIZE-1), high=CHIP_SIZE)*PIXEL_SIZE
        dx = rng.integers(low=-(CHIP_SIZE-1), high=CHIP_SIZE)*PIXEL_SIZE
        self.jittered = Affine.translation(dx, dy)
        
    def unjitter(self):
        self.jittered = None
        
    def __getitem__(self, idx):
        grid_cell = self.grid.iloc[idx % len(self.grid)]
        grid_cell_year = self.years[idx // len(self.grid)]
        xmin, ymin, xmax, ymax = grid_cell.geometry.bounds

        if self.jittered:
            xmin, ymin = self.jittered * (xmin, ymin)
            xmax, ymax = self.jittered * (xmax, ymax)

        chip_bounds = (xmin, ymin, xmax, ymax)
        out = {"YEAR": grid_cell_year, "GRID_IDX": idx % len(self.grid)}
        
        if self.ablate == self.source:
            out[self.source] = torch.normal(mean=self.stat[f"MEANS_{self.source}"].expand(-1, CHIP_SIZE, CHIP_SIZE), std=self.stat[f"STDDV_{self.source}"].expand(-1, CHIP_SIZE, CHIP_SIZE))
        else:
            out[self.source] = read_window(self.imagery_dir / f"{self.source}{grid_cell_year}.tif", chip_bounds)
        
        if self.ablate == self.dem:
            out[self.dem] = torch.normal(mean=self.stat[f"MEANS_{self.dem}"].expand(-1, CHIP_SIZE, CHIP_SIZE), std=self.stat[f"STDDV_{self.dem}"].expand(-1, CHIP_SIZE, CHIP_SIZE))
        else:
            out[self.dem] = read_window(self.dem_file, chip_bounds, 1)
            
        out.update(self.__get_cache__(chip_bounds, grid_cell_year)) 

        return out


def chip_writer(queue, file_path, grid, years, profile, clip=None, jitter=None, count=False):
    file_refs = {y: rasterio.open(file_path/f"{y}.tif", 'w+', **profile) for y in years}
    if count:
        file_refs_c = {y: rasterio.open(file_path/f"{y}_c.tif", 'w+', **profile) for y in years}
        
    while True:
        payload = queue.get()
        if payload is None:
            break
        idxes, years, preds = payload
        preds = preds.astype(np.uint16)
        for i in range(years.shape[0]):
            year = years[i].item()
            idx = idxes[i].item()
            xmin, ymin, xmax, ymax = grid.iloc[idx].geometry.bounds
            if jitter is not None:
                xmin, ymin = jitter * (xmin, ymin)
                xmax, ymax = jitter * (xmax, ymax)
            if clip is not None:
                xmin, ymin, xmax, ymax = (
                    xmin + clip,
                    ymin + clip,
                    xmax - clip,
                    ymax - clip
                )
            try:
                win = from_bounds(xmin, ymin, xmax, ymax, file_refs[year].transform)
                arr = file_refs[year].read(window=win)
                arr = np.where(arr == INF_NO_DATA, 0, arr)
                arr += preds[i]
                file_refs[year].write(arr, window=win)
                if count:
                    win = from_bounds(xmin, ymin, xmax, ymax, file_refs_c[year].transform)
                    arr_c = file_refs_c[year].read(window=win)
                    arr_c = np.where(arr_c == INF_NO_DATA, 0, arr_c)
                    file_refs_c[year].write(arr_c+np.uint16(1), window=win)
            except Exception as e:
                print(f"Error writing chip at index {idx} for year {year} win {win} bounds {(xmin, ymin, xmax, ymax)}: {e}")


def proc_sample(samples, stats, kwargs):
    procs = {}
    for k, sample in samples.items():
        if k != "YEAR" and k != "GRID_IDX":
            sample = sample.to(DEVICE, DTYPE)
            if 'LS' in k and kwargs['norm']:
                means = stats[f"MEANS_PRITHVI"]
                stddv = stats[f"STDDV_PRITHVI"]
                sample = (sample - means) / stddv
            elif k == 'PRISM__':
                    for i, p in enumerate(PRISM_ELEMENTS):
                        P = p.upper()
                        means = stats[f"MEANS_PRISM_{P}"]
                        stddv = stats[f"STDDV_PRISM_{P}"]
                        sample[:,i:i+1] = (sample[:,i:i+1] - means) / stddv
            else:
                means = stats[f"MEANS_{k}"]
                stddv = stats[f"STDDV_{k}"]
                sample = (sample - means) / stddv
            sample[torch.isnan(sample)] = NO_DATA  
        procs[k] = sample
    return procs


def run_inference(dataset, model, model_kwargs, stats, profile, out_dir, name, clip, count):
    out_path = out_dir / name
    out_path.mkdir(exist_ok=True)
    q = mp.Queue(maxsize=256)

    if clip:
        slc = slice(CLIP_SIZE//PIXEL_SIZE, -CLIP_SIZE//PIXEL_SIZE)
        clip_val = CLIP_SIZE
    else:
        slc = slice(None)
        clip_val = None

    writer_proc = mp.Process(
        target=chip_writer,
        args=(q, out_path, dataset.grid, dataset.years, profile, clip_val, dataset.jittered, count),
    )
    writer_proc.start()
    inf_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    with torch.no_grad():
        for samples in tqdm(inf_loader, desc=name):
            procs = proc_sample(samples, stats, model_kwargs)
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                pred = model(procs)
            pred[pred < 0] = 0
            pred = pred.to(torch.uint16)
            q.put((procs['GRID_IDX'].cpu(), procs["YEAR"].cpu(), pred[:, :, slc, slc].cpu().numpy()))
    q.put(None)
    writer_proc.join()


def main(name, pattern):
    with open(STAT_PATH) as f:
        stat = yaml.safe_load(f)
    
    stats = {}
    for k, s in stat.items():
        stats[k] = torch.tensor(s, device=DEVICE, dtype=DTYPE).view(-1, 1, 1)

    _CKPT_DIR = TRAIN_OUT_DIR / name / "ckpt"
    _CKPT_PATH = sorted(list(_CKPT_DIR.glob(f"{pattern}*.pt")))[0]
    _OUT_DIR = TRAIN_OUT_DIR / name / _CKPT_PATH.with_suffix("").name
    _OUT_DIR.mkdir(exist_ok=True, parents=True)
    
    ckpt = torch.load(_CKPT_PATH)
    
    # instantiating model and optimizers
    model = PrithviUPerNet(**ckpt["model_kwargs"])
    # model = AttnSETRPUP(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE, DTYPE)
    model.eval()
    
    cache = cache_anc(IMAGERY_DIR, ckpt["model_kwargs"]['img'], DIRS_YEARS[ckpt["model_kwargs"]['img']], PATCH_SIZE, PRISM_ELEMENTS, PRISM_FILES, SOLUS_FILES, DTYPE)
    inf_dataset = GridUnitInfDataset(ckpt["model_kwargs"]['img'], ckpt["model_kwargs"]['dem'], stat=stats, cache=cache)
    transform = inf_dataset.patch_transform * Affine.scale(1/PATCH_SIZE, 1/PATCH_SIZE)
    profile = {
        'count': 1,
        'width': cache["width"],
        'height': cache["height"],
        "driver": "GTiff",
        'transform': transform,
        'BIGTIFF': 'YES',
        'tiled': True,
        'blockxsize': CHIP_SIZE,
        'blockysize': CHIP_SIZE,
        'crs': EPSG_CODE,
        'nodata': INF_NO_DATA,
        'compress': 'lzw',
        'dtype': "uint16"
    }
    run_inference(inf_dataset, model, ckpt["model_kwargs"], stats, profile, _OUT_DIR, f"no_overlap", False, False)
    
    # this could probably be parallelized but i will leave it
    for ablate in ['NASADEM', 'SOLUS100', 'PRISM__', 'PRISM', *[f"PRISM_{p.upper()}" for p in PRISM_ELEMENTS]]:
        inf_dataset.set_ablate(ablate)
        run_inference(inf_dataset, model, ckpt["model_kwargs"], stats, profile, _OUT_DIR, f"ran_{ablate}", False, False)
    
    # slide and avg
    inf_dataset.set_grid(CLIP_GRID)
    run_inference(inf_dataset, model, ckpt["model_kwargs"], stats, profile, _OUT_DIR, f"slide_avg", False, True)
    
    # random jitter
    out_dir = _OUT_DIR / "jitters"
    out_dir.mkdir(exist_ok=True)
    for position in range(NUM_JITTERS):
        if position > 0:
            inf_dataset.jitter()
            profile.update({"transform": inf_dataset.jittered * profile["transform"]})
        run_inference(inf_dataset, model, ckpt["model_kwargs"], stats, profile, out_dir, f"{position:04d}", True, False)

    # # full inference
    # inf_dataset.set_grid(FISHNET_GRID)
    # run_inference(inf_dataset, model, ckpt["model_kwargs"], stats, profile, _OUT_DIR, "full_inference")

       
if __name__ == "__main__":
    name = "upernet"
    pattern = "val_rmse_"
    main(name, pattern)