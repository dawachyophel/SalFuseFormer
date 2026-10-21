# SalFuseFormer

Official code for:

> E. Dhar, D. C. Lepcha, A. Ali, S. Mallik, L. Gaur, S. Syed-Abdul.
> **SalFuseFormer: A saliency-guided fusion transformer framework for clinically motivated multi-modal medical image fusion.**
> *Biomedical Signal Processing and Control*, 2026 (in press). Article BSPC_111701. DOI: _to be added on publication_.

SalFuseFormer adds two training-free modules to a domain-fine-tuned SwinFusion backbone:

- **Saliency-masked edge injection (EI):** sharpens MRI boundaries in low-activity regions.
- **Adaptive chrominance fusion (ACF):** preserves PET/SPECT false-colour encoding in metabolically active hotspots.

Both modules are driven by one HSV-based functional-activity saliency map. ACF is applied to MRI-PET and MRI-SPECT, and bypassed for CT-MRI.

---

## About this repository

This repository contains **only the proposed method**. The scripts are the exact versions that produced the results in the published article, copied without modification. `MANIFEST.txt` lists the SHA-256 checksum of every file.

Baseline methods (DenseFuse, FusionGAN, DDcGAN, U2Fusion, EMFusion, SeAFusion, CDDFuse, CoCoNet, MUFusion, SwinFusion) are **not** included. Their outputs must be generated with each method's official code and released weights (see [Comparison methods](#comparison-methods)).

Because the scripts are unmodified, some use hard-coded relative paths (for example `Dataset/testsets/...`, `results/...`, `finetune_checkpoints/...`). Run every script from the repository root, with the folder layout described below.

---

## Repository structure

```
SalFuseFormer/
├── medswin_acf_combined.py          # SalFuseFormer pipeline: saliency, EI, ACF, YCrCb reconstruction (Sec. 3.3–3.7)
├── finetune_pet_mri_correct.py      # Domain-specific backbone fine-tuning (Sec. 3.2, 4.4)
├── test_swinfusion.py               # Backbone inference (fused luminance)
├── eval_option_c.py                 # Fine-tuned backbone evaluation and inference
├── medswin_enhance.py               # Edge-injection-only variant (ablation config B)
├── enhance_fusion.py, Original_/    # Helpers imported by medswin_enhance.py
├── adaptive_chroma_fusion.py        # ACF-only variant (ablation config C)
├── run_ct_spect_metrics.py          # Pipeline + metrics for CT-MRI / MRI-SPECT
├── compute_harvard_statistics.py    # Per-image metrics, Harvard (Tables 2–4)
├── compute_petmri_statistics.py     # Table 2 + Wilcoxon tests
├── compute_ctmri_statistics.py      # Table 3 + Wilcoxon tests (18 pairs)
├── compute_spect_statistics.py      # Table 4 + Wilcoxon tests
├── compute_ablation_revised.py      # Table 7 (component ablation)
├── run_sensitivity_new.py           # Table 8 (τ and α sweeps)
├── measure_flops.py                 # Table 6 (params / FLOPs / time)
├── regenerate_adni_proper.py        # ADNI-3 slice extraction, JET colouring, brain masking
├── run_salfuse_adni_proper.py       # SalFuseFormer inference on ADNI-3
├── compute_adni_proper_metrics.py   # Table 9
├── run_adni_roi.py                  # ROI CNR on 13 AD biomarker regions
├── compute_roi_statistics.py        # ROI Wilcoxon tests (Sec. 7.3)
├── compute_colourmap_robustness.py  # Colourmap sensitivity r-values (Fig. 15)
├── collect_fig1_components.py, generate_fig5.py … generate_fig12.py,
│   generate_fig14_revised.py, generate_fig_colourmap.py   # Figures
├── models/  utils/  data/  options/swinir/train_swinfusion_med.json   # SwinFusion backbone code
├── finetune_checkpoints/best_model.pth   # Fine-tuned backbone (epoch 11)
├── harvard_petmri_SalFuseFormer_perimage.csv   # Published per-image MRI-PET metrics (reference)
└── MANIFEST.txt
```

---

## Requirements

The original environment was:

| Component | Version |
|---|---|
| Python | 3.9 |
| PyTorch | 2.5.1 (CUDA 12.6) |
| OpenCV | 4.8 |
| NumPy | 1.24 |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB |

Other packages used: scikit-image, SciPy, pandas, matplotlib, nibabel (ADNI-3), thop (FLOPs) and timm (Swin backbone).

```bash
pip install torch==2.5.1 opencv-python==4.8.* numpy==1.24.* scikit-image scipy pandas matplotlib nibabel thop timm
```

---

## Data

**Harvard Whole Brain Atlas** (public): http://www.med.harvard.edu/AANLIB/home.html

Expected layout:

```
Dataset/testsets/PET-MRI/MRI          Dataset/testsets/PET-MRI/PET_RGB
Dataset/testsets/CT-MRI/CT            Dataset/testsets/CT-MRI/MRI
Dataset/testsets/MRI-SPECT/MRI        Dataset/testsets/MRI-SPECT/SPECT
```

The test sets are:

- MRI-PET: 10 pairs
- CT-MRI: 18 pairs. Pairs `41022.png` and `41023.png` are excluded inside the scripts.
- MRI-SPECT: 20 pairs

Fine-tuning uses 269 MRI-PET training pairs, which do not overlap with the test sets.

**ADNI-3** (restricted): available to qualified researchers at http://adni.loni.usc.edu after accepting the ADNI Data Use Agreement. ADNI images are **not** redistributed here. `regenerate_adni_proper.py` rebuilds the 147 MRI/FDG-PET slice pairs from the co-registered NIfTI volumes. FreeSurfer `recon-all` outputs (Desikan-Killiany) are needed for the ROI analysis.

---

## Weights

| File | Source |
|---|---|
| `finetune_checkpoints/best_model.pth` | Included. SalFuseFormer fine-tuned backbone, best validation SSIM at epoch 11. |
| `Model/Medical_Fusion-PET-MRI/Medical_Fusion/models/10000_G.pth`, `10000_E.pth` | Original SwinFusion pretrained weights (starting point for fine-tuning). Download from the official SwinFusion repository: https://github.com/Linfeng-Tang/SwinFusion |

---

## Reproducing the results

Hyperparameters (Table 1):

| Parameter | MRI-PET | CT-MRI | MRI-SPECT |
|---|---|---|---|
| Injection strength α | 0.05 | 0.18 | 0.05 |
| Saliency threshold τ | 0.20 | 0.40 | 0.20 |
| Gaussian smoothing σ | 5.0 | 5.0 | 5.0 |
| ACF chroma blend β | 0.25 | N/A (bypassed) | 0.25 |
| Edge smoothing σₑ | 1.5 | 1.5 | 1.5 |

### 1. Fine-tune the backbone (optional; the checkpoint is provided)

```bash
python finetune_pet_mri_correct.py      # writes finetune_checkpoints/best_model.pth
```

Training settings are 30 epochs, Adam, learning rate 2e-4, batch size 4, 128×128 patches, with the best checkpoint selected by validation SSIM.

### 2. Generate fused images

The fine-tuned backbone produces the fused luminance (`test_swinfusion.py`, `eval_option_c.py`). The SalFuseFormer enhancement is then applied with `medswin_acf_combined.py` (run `python medswin_acf_combined.py --help` for its arguments). The statistics scripts read the fused images from:

| Task | Folder |
|---|---|
| MRI-PET | `results/SalFuseFormer_FINAL_PET-MRI` |
| CT-MRI | `results/SwinFusion_CT-MRI_MedSwinACF_fixed` |
| MRI-SPECT | `results/SalFuseFormer_SPECT_colour` |

### 3. Script for each table and figure

| Result | Script(s) |
|---|---|
| Table 2 (MRI-PET) | `compute_harvard_statistics.py`, `compute_petmri_statistics.py` |
| Table 3 (CT-MRI) | `compute_harvard_statistics.py`, `compute_ctmri_statistics.py` |
| Table 4 (MRI-SPECT) | `compute_harvard_statistics.py`, `compute_spect_statistics.py` |
| Table 5 (ranks) | Derived from Tables 2–4 |
| Table 6 (efficiency) | `measure_flops.py`* |
| Table 7 (ablation) | `compute_ablation_revised.py` (configs B/C from `medswin_enhance.py`, `adaptive_chroma_fusion.py`) |
| Table 8 (sensitivity) | `run_sensitivity_new.py` → `results/sensitivity_new/{tau,alpha}_sweep.csv` |
| Table 9 (ADNI-3) | `regenerate_adni_proper.py` → `run_salfuse_adni_proper.py` → `compute_adni_proper_metrics.py` |
| Fig. 1 | `collect_fig1_components.py` |
| Figs. 5–7 | `generate_fig5.py`, `generate_fig6.py`, `generate_fig7.py` |
| Fig. 8 | `generate_fig8.py` |
| Fig. 9 | `generate_fig9.py` |
| Fig. 10 | `generate_fig10.py` |
| Fig. 11 | `generate_fig11.py` |
| Fig. 12 | `generate_fig12.py` |
| Fig. 14 (ROI CNR) | `run_adni_roi.py` → `compute_roi_statistics.py` → `generate_fig14_revised.py` |
| Fig. 15 (colourmaps) | `compute_colourmap_robustness.py`, `generate_fig_colourmap.py` |

\* `measure_flops.py` contains absolute Windows paths to a local copy of the original SwinFusion code and weights. Edit those paths to point at your copy. Parameter and FLOP counts depend only on the architecture, which the fine-tuned model shares.

---

## Comparison methods

The statistics and figure scripts compare SalFuseFormer against baseline outputs stored at `comparison_results/<Method>/<Task>/`. Those outputs are not part of this repository. Generate them with each method's official implementation and released pretrained weights, as described in Sec. 4.2 of the paper. A script skips any method whose folder is missing.

---

## Citation

```bibtex
@article{dhar2026salfuseformer,
  title   = {SalFuseFormer: A saliency-guided fusion transformer framework for clinically motivated multi-modal medical image fusion},
  author  = {Dhar, Eshita and Lepcha, Dawa Chyophel and Ali, Aaliya and Mallik, Saurav and Gaur, Loveleen and Syed-Abdul, Shabbir},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {In press}
}
```

## Acknowledgements

The backbone code (`models/`, `utils/`, `data/`, `options/`, `test_swinfusion.py`) is from [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion) (Ma et al., IEEE/CAA JAS 2022). Please cite it as well. This work was supported by the National Science and Technology Council (NSTC), Taiwan, under grant 113-2923-E-038-002-MY3.

## Contact

Dawa Chyophel Lepcha and Shabbir Syed-Abdul (corresponding authors), Taipei Medical University.
