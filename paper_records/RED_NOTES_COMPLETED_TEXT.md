# Text to paste over the red notes

The repeat inference is in `paper_records/repeat_inference/`. It did not overwrite the retained score files. Maximum absolute difference from the retained tables: 2.4e-6 dB PSNR and 5.5e-7 SSIM. Python 3.11.9, PyTorch 2.7.1+cpu, Windows, 12 CPUs, 10 PyTorch threads, `num_workers=0`.

ORCID iDs are not in the experiment files. Leave those two fields for the authors. Do not invent a Zenodo DOI.

## Page 1, after the keywords

Author contributions: F.S. developed the model, ran the experiments, and drafted the manuscript. M.-Á.L.-N. and R.A.-C. supervised the study and revised the manuscript. All authors approved the submitted version.

Conflicts of interest: The authors declare no conflict of interest.

The funding statement appears at the end of the article. No additional funding was received.

Confirm the contribution and conflict sentences with both supervisors before submission. Insert the ORCID iDs only after F.S. and M.-Á.L.-N. supply them. The printed corresponding address remains phizasohail429@gmail.com unless F.S. replaces it.

## End of Section 3.6.1, after Table 3

A hash audit of the 890 raw files and 890 reference files recorded no missing pair, no exact duplicate within either folder, and no exact raw duplicate across the training, validation, and test lists. The filenames do not contain a scene or sequence identifier, so the intervals in Section 3.6.3 are image-level intervals. The 60 UIEB images without references remain outside the 890-pair manifest. All reported scores use this UIEB partition.

## End of Section 3.6.2

Random-number and data-loader generator states were reinitialized at the epoch-5 resumption, so epochs 5–30 did not continue a single random stream. The batch log contains 3,580 finite rows. Epochs 1, 4, 6, and 30 each contain 119 batch rows. Epoch 5 contains 129 rows: batch identifiers 0–9 occur twice, and identifiers 10–118 occur once. The repeated identifiers come from appending rows when training resumed at epoch 5. The original log was retained. The selected checkpoint is the epoch-29 model of this continued 10−4 run. The non-finite trial used a learning rate of 10−3 and is not the reported model.

A second inference pass loaded the four selected checkpoints and scored the same 89 filenames with the same preprocessing. It used Python 3.11.9, PyTorch 2.7.1+cpu, 10 PyTorch threads, and `num_workers=0`. The largest absolute difference from the retained per-image PSNR and SSIM values was 2.4×10−6 dB and 5.5×10−7, respectively. The retained tables were not replaced. Training was not repeated from a new initialization, so the bootstrap intervals remain conditional on these checkpoints.

## End of Section 3.6.3

UIQM and UCIQE were computed on the denormalized images, clamp(0.5u+0.5, 0, 1), for the same 89 test files. UIQM follows Panetta, Gao, and Agaian, with UICM 10% asymmetric alpha-trimming, UISM as the EME of the Sobel magnitude, UIConM as PLIP log-AMEE, and weights 0.0282, 0.2953, and 0.6765. UCIQE follows Yang and Sowmya: 0.4680 times the chroma standard deviation, plus 0.2745 times luminance contrast, plus 0.2576 times mean saturation, in CIE Lab. Both are no-reference scores. They were not used for checkpoint selection. LPIPS and CIEDE2000 were not computed. The Sobel term in Equation (4) is a training loss, not a test metric. The paired comparisons in Section 4 remain the PSNR and SSIM definitions above.

## End of Section 3.6.4

The checkpoints scored in Tables 8 and 9 are the highest-validation-PSNR files. E-URM, epoch 29, has 884,012,496 bytes and SHA-256 `176c0673f6fa3d4d0ae03ff7ce7537109185cfba418b79fbc9f63675719220c9`. Without ACC, epoch 30, has 883,963,691 bytes and SHA-256 `4baf4ea28a680511898cffbc538b3433754c7c0d3b0a6d9ebdf2acb21ef9794e`. The convolutional control, epoch 29, has 921,192,901 bytes and SHA-256 `c43412b2d6179a5fcb7eb610d20e802e910f596ea2650f8b06b7adc8e5b13813`. Without MSA blocks, epoch 30, has 506,541,008 bytes and SHA-256 `0f77ec1205c9116f6873a36982b988ca409385e24e2f5ca3de79a804865dabbb`. The second inference pass in Section 3.6.2 used these four files. Water-Net, FUnIE-GAN, and PCAFA-Net were not scored. The without-MSA variant replaces each multiscale block with identity. Latency and memory of the trained checkpoints were not measured.

## New table in Section 4.3, after Table 8

Table 11. No-reference means on the same 89 test images. UIQM and UCIQE were not used to select checkpoints.

| Method | UIQM | UCIQE |
|---|---:|---:|
| Unprocessed input | 2.240 | 23.193 |
| E-URM | 2.248 | 28.969 |
| Without ACC | 2.254 | 28.424 |
| Convolutional control | 2.237 | 28.048 |
| Without MSA blocks | 2.251 | 27.597 |

One sentence under the table: UIQM changed little relative to the unprocessed input, whereas UCIQE was higher for every trained configuration. These scores do not replace the paired PSNR and SSIM results.

## End of Section 6, before Funding

Future work. The reported evidence is the locked UIEB partition and one training run of each configuration in Table 4. Later work can test the saved E-URM checkpoint on the paired and unpaired EUVP collections separately, and on UVEB with splits kept at the video-sequence level. The same test list can be used to score Water-Net, FUnIE-GAN, and PCAFA-Net once their official checkpoints are available, with training-image overlap recorded. A further variant can remove only the attention gates while retaining the rest of each multiscale block. Repeated training from new initializations, with generator states restored whenever a run is continued, would show how far the means move. LPIPS, CIEDE2000 on calibrated color targets, and a detection or segmentation comparison are also outside the present results.

## Replace the Data Availability statement

UIEB is described in Ref. [4]. The model implementation, training and evaluation code, split lists, configuration, pair audit, and main E-URM checkpoint are available in the versioned source repository. The four selected checkpoints listed in Section 3.6.4, the per-image tables, and the repeat-inference records are retained with the authors. A permanent public archive DOI has not yet been assigned.

## Appendix A

Delete Appendix A from the submitted PDF. Sections 1–6 do not cite it. Do not move Table A1 into Section 4. Those timings are untrained forward passes.
