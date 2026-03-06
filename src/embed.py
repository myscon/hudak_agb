import copy
import torch
import torch.multiprocessing as mp
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from download import  DIRS_YEARS, IMAGERY_DIR, STAT_PATH, PRISM_ELEMENTS, CHIP_SIZE, EPSG_CODE
from train import DEVICE, DTYPE, SDTYPE, PATCH_SIZE, NO_DATA, TOUT_DIR, PRISM_FILES, SOLUS_FILES, NO_DATA, L_SIZE, AttnSETRPUP
from inference import proc_sample, chip_writer, GridUnitInfDataset
from utils import cache_anc


BATCH_SIZE = 32


def forward(model, x, attend):
    anc = []
    anc.append(model.dem_encoder(x[model.dem]))
    anc.append(model.solus_encoder(x["SOLUS100"]))
    
    for p in PRISM_ELEMENTS:
        P = f"PRISM_{p.upper()}"
        anc.append(model.prism_encoders[P](x[P]))
    anc.append(model.prism_encoder(x["PRISM__"]))
    
    anc = torch.cat(anc, dim=1)
    if model.prithvi_mode == "vit":
        latent = model.prithvi_encoder(x[model.img])
        latent = latent['last_hidden_state'][:,1:,:]
    elif model.prithvi_mode == "conv":
        latent = model.prithvi_encoder(x[model.img])
        latent = latent.flatten(2).transpose(1, 2)
    else:
        latent = model.prithvi_encoder.forward_features(x[model.img].unsqueeze(2))
        latent = latent[:,1:,:]
    if attend:
        latent = model.fusion(latent, anc)
    else:
        B, L, D = latent.shape
        latent = latent.transpose(1, 2).reshape(B, D, L_SIZE, L_SIZE)
    return latent


def embed(dataset, model, model_kwargs, stats, profile, out_dir, name, attend=True):
    out_path = out_dir / name
    out_path.mkdir(exist_ok=True)
    q = mp.Queue(maxsize=256)

    writer_proc = mp.Process(
        target=chip_writer,
        args=(q, out_path, dataset.grid, dataset.years, profile),
    )
    writer_proc.start()
    
    inf_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    with torch.no_grad():
        for samples in tqdm(inf_loader, desc=name):
            procs = proc_sample(samples, stats, model_kwargs)
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                embed = forward(model, procs, attend)
            q.put((procs['GRID_IDX'].cpu(), procs["YEAR"].cpu(), embed.cpu().numpy()))
    q.put(None)
    writer_proc.join()


def main(name, pattern):
    with open(STAT_PATH) as f:
        stat = yaml.safe_load(f)
    
    stats = {}
    for k, s in stat.items():
        stats[k] = torch.tensor(s, dtype=DTYPE).view(-1, 1, 1)

    _CKPT_DIR = TOUT_DIR / name / "ckpt"
    _CKPT_PATH = sorted(list(_CKPT_DIR.glob(f"{pattern}*.pt")))[0]
    _OUT_DIR = TOUT_DIR / name / _CKPT_PATH.with_suffix("").name
    _OUT_DIR.mkdir(exist_ok=True, parents=True)
    
    ckpt = torch.load(_CKPT_PATH)
    
    # instantiating model and optimizers
    model = AttnSETRPUP(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    
    cache = cache_anc(IMAGERY_DIR, ckpt["model_kwargs"]['img'], DIRS_YEARS[ckpt["model_kwargs"]['img']], PATCH_SIZE, PRISM_ELEMENTS, PRISM_FILES, SOLUS_FILES, SDTYPE)
    inf_dataset = GridUnitInfDataset(ckpt["model_kwargs"]['img'], ckpt["model_kwargs"]['dem'], stat=stats, cache=cache)
    profile = {
        'count': 1024,
        'width': cache["width"] // PATCH_SIZE,
        'height': cache["height"] // PATCH_SIZE,
        "driver": "GTiff",
        'transform': inf_dataset.patch_transform,
        'BIGTIFF': 'YES',
        'tiled': True,
        'blockxsize': CHIP_SIZE,
        'blockysize': CHIP_SIZE,
        'crs': EPSG_CODE,
        'nodata': NO_DATA,
        'compress': 'lzw',
        'dtype': "float32"
    }

    embed(inf_dataset, model, ckpt["model_kwargs"], stats, copy.deepcopy(profile), _OUT_DIR, "pre_ca_embed", attend=False)
    # embed(inf_dataset, model, ckpt["model_kwargs"], stats, copy.deepcopy(profile), _OUT_DIR, "post_ca_embed", attend=True)


if __name__ == "__main__":
    name = "initial_attempt"
    pattern = "val_rmse_"
    main(name, pattern)