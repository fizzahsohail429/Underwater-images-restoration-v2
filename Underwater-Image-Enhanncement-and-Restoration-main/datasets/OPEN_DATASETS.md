# Open Source Underwater Datasets

This project supports popular open datasets. Download from the official pages, then use the provided prep scripts to structure them as `input/` and optionally `target/`.

- UIEB (paired): `https://li-chongyi.github.io/proj_uieb.html`
- EUVP (paired subsets): `https://irvlab.cs.umn.edu/resources/euvp-dataset`
- UODD (no-ref detection dataset): `https://github.com/LehiChiang/Underwater-object-detection-dataset`
- UTDAC2020 (no-ref challenge data): `https://aistudio.baidu.com/datasetdetail/215376`
- RUOD (no-ref detection dataset): `https://github.com/dlut-dimt/RUOD`
- UDD (no-ref detection dataset): `https://github.com/chongweiliu/UDD_Official`

## Prepare paired datasets (UIEB/EUVP)
- UIEB:
```bash
python Underwater-Image-Enhanncement-and-Restoration-main/datasets/prepare_uieb.py --uieb_root /path/to/UIEB --out Underwater-Image-Enhanncement-and-Restoration-main/datasets/UIEB_paired
```
- EUVP:
```bash
python Underwater-Image-Enhanncement-and-Restoration-main/datasets/prepare_euvp.py --euvp_root /path/to/EUVP --subset paired --out Underwater-Image-Enhanncement-and-Restoration-main/datasets/EUVP_paired
```

Train (paired):
```bash
python -m Underwater-Image-Enhanncement-and-Restoration-main.experiments.blur_dark_2025.run_experiment \
  --data_root Underwater-Image-Enhanncement-and-Restoration-main/datasets/UIEB_paired \
  --workdir runs/uieb_2025 --image_size 256 --batch_size 8 --epochs 20 --augment_blur_dark
```

## Prepare no-reference datasets (UODD/UTDAC2020/RUOD/UDD)
- UODD:
```bash
python Underwater-Image-Enhanncement-and-Restoration-main/datasets/prepare_uodd.py --uodd_root /path/to/UODD --out Underwater-Image-Enhanncement-and-Restoration-main/datasets/UODD_noref
```
- UTDAC2020:
```bash
python Underwater-Image-Enhanncement-and-Restoration-main/datasets/prepare_utdac2020.py --utdac_root /path/to/UTDAC2020 --out Underwater-Image-Enhanncement-and-Restoration-main/datasets/UTDAC2020_noref
```
- RUOD:
```bash
python Underwater-Image-Enhanncement-and-Restoration-main/datasets/prepare_ruod.py --ruod_root /path/to/RUOD --out Underwater-Image-Enhanncement-and-Restoration-main/datasets/RUOD_noref
```
- UDD:
```bash
python Underwater-Image-Enhanncement-and-Restoration-main/datasets/prepare_udd.py --udd_root /path/to/UDD --out Underwater-Image-Enhanncement-and-Restoration-main/datasets/UDD_noref
```

Evaluate (no-ref) with UIQM/UCIQE:
```bash
python -m Underwater-Image-Enhanncement-and-Restoration-main.nextgen2025.eval_noref \
  --data_root Underwater-Image-Enhanncement-and-Restoration-main/datasets/UODD_noref \
  --checkpoint runs/blur_dark/checkpoints/model_epoch_20.pt
```
