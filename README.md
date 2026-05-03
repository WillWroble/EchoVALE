# EchoVALE: Vision-Aligned Language Encoder for Pediatric Echocardiography

## Abstract

Using a pretrained JEPA backbone, EchoVALE reframes clinical echocardiographic classification from multiple independent binary prediction tasks to the single continuous task of projecting clinical language onto echo video representation space. Without any label supervision, EchoVALE outperforms supervised diagnostic classifiers on their own benchmark, achieving 0.85 N-weighted mean AUROC across thousands of Fyler codes on held-out 2020+ studies. We show how this approach improves general clinical understanding compared to binary classifiers, and has downstream applications in zero-shot classification, natural language line validation, and full report generation.

## Introduction

Current approaches to automated echocardiographic interpretation treat each clinical finding as an independent binary classification task. This scales poorly: adding a new finding requires new labels and retraining, rare findings are irrecoverable without sufficient positive examples, and the full clinical content of an echocardiogram cannot be captured by any fixed set of classifiers.

EchoVALE replaces this paradigm with a single continuous task: given an echocardiographic study and any clinical text, score how well the text describes what is in the echo. For report generation, the system scores a retrieval pool of ~180K known clinical lines against a study simultaneously, producing a probability field over clinical language space.

The line encoder learns from three converging signals:

1. **Language prior**: Frozen ClinicalBERT provides a pretrained clinical language geometry. Lines describing related findings are already neighbors in BERT space before any training.
2. **Co-occurrence structure**: The skip-gram like objective reshapes this space by report context. Lines that co-occur in the same reports are pulled together; lines that never co-occur are pushed apart. This captures clinical relationships that language alone cannot (e.g. "status post Fontan" and "single ventricle physiology" co-occur despite being linguistically dissimilar).
3. **Physiological/visual signal**: Cross-attention over frozen video embeddings grounds the text representations in actual echocardiographic appearance. Lines are scored against what the echo physically shows, not just what tends to appear in reports.

A novel clinical description inherits geometric proximity to related findings from the language prior, inherits co-occurrence context from the skip-gram objective, and is grounded in visual evidence by cross-attention. When the model makes errors, they are typically confusions between visually or clinically similar findings that sit near each other in the learned space, not random failures.

## Methods

### Vision Backbone: EchoJEPA (`video_pretraining_v2/`)

EchoJEPA ViT-B pretrained on BCH pediatric echo data, initialized from V-JEPA 2.1 weights.
- 217K studies, ~11.4M AVI files
- ViT-B (86.8M params, 768d native)
- Masked latent prediction: predicts representations of masked spatiotemporal patches from visible patches, using positional embeddings of masked positions as prediction targets. No labels, no clinical text in the training signal.
- 4 clips (16 frames, 224x224) sampled per video; all clips from a study (~200 total) fed directly into the line encoder
- JEPA outperforms PanEcho for the line encoder across all Fyler frequency ranges. JEPA tokens encode relational/predictive structure (trained for mutual spatiotemporal prediction), not just category-relevant features.

### Line Encoder: CrossAttentionPool (`line_tokenizer_v2/`)

**LineEncoder**: Bio_ClinicalBERT (~110M params, 768d CLS, last layer unfrozen) + learned linear projection. Encodes any clinical text string into a line embedding. 

**CrossAttentionPool**: Line embeddings cross-attend over the full set of ~200 frozen video clip embeddings with learned W_Q, W_K, W_V (~18,306,048 trainable params including last BERT layer). Each line asks its own question of the video. A line about tricuspid regurgitation attends to different clip tokens than a line about LV dilation. The output is a line-relative study representation: the video content most relevant to that specific clinical query.

**Scoring**: Dot product between the cross-attention output (line-relative study representation) and the raw line embedding, passed through sigmoid, yielding a per-line relevance score.

**Training objective**: Skip-gram BCE. Each field-study pair is treated as a separate training sample. Lines co-occurring in the same field of the same report are positives; randomly sampled lines are negatives. Each line scored independently with no softmax forcing competition between lines. Following VL-JEPA (Chen et al., Meta FAIR, 2024), EchoVALE operates entirely in embedding space rather than autoregressively generating tokens, further constraining the answer space to binary relevance and concentrating representational power on the query side.

**Current best**: v11 (JEPA base, W_V, post-CA MLP, 4x BERT expansion). no Post-CA MLP (v6) performs better on pre 2020 studies but hurts overall Fyler AUROCs. Ablations ongoing (v12-v16).

### Generation (`generate_lines.py`)

- Scores ~180K clinical lines from the 2020+ training set against a study's video embeddings
- Mutual KNN + connected components hotspot clustering (default K=10)
- Output: `{field: {hotspots, reference, n_hotspots, n_active}}`
- Unmapped reference lines included with `score: None`
- Embedding caching via `get_or_encode_lines`

## Results


### Fyler Code Classification (2020+ evaluation set)

v11, JEPA backbone, excluding trivial/ruled-out labels, using the best line variant per code:
- Mean AUROC: 0.8260
- Median AUROC: 0.8767
- N-weighted mean: 0.851
- ~412 Fyler codes flagged as genuinely unpredictable from echo alone (non-cardiac medical, ECG-only, admin/procedural, medications, family history); listed in `unpredictable_fyler_codes.txt` but still included in this data. 
- Excluding unpredictable codes: ~0.88 mean AUROC

### Per-Diagnosis AUROCs (2020+ evaluation set)
 
Comparison of EchoVALE line encoder on JEPA (v6, v11) and PanEcho backbones against supervised EchoFocus CLIP probes. Best line shown for v11.
 
| Diagnosis | Fyler | n | JEPA v11 | JEPA v6 | PanEcho LE | EchoFocus CLIP | Best Line |
|---|---|---|---|---|---|---|---|
| Single RV | 210 | 68 | 0.981 | 0.973 | 0.858 | 0.894 | Single right ventricle (Probable) |
| Single LV | 220 | 302 | 0.977 | 0.970 | 0.916 | 0.967 | Double inlet single left ventricle |
| TOF | 1050 | 2213 | 0.963 | 0.952 | 0.878 | 0.923 | Tetralogy of Fallot |
| ALCAPA | 3101 | 102 | 0.963 | 0.929 | 0.850 | 0.945 | Anomalous origin of LMCA from PA (Consequent to surgery) |
| HLHS | 300 | 589 | 0.962 | 0.946 | 0.927 | 0.941 | Hypoplastic left heart syndrome |
| Tri atresia | 400 | 219 | 0.959 | 0.947 | 0.906 | 0.870 | Tricuspid atresia (Primary cardiac) |
| TOF+PA | 1080 | 694 | 0.958 | 0.949 | 0.873 | 0.917 | Tetralogy of Fallot with pulmonary atresia |
| Truncus | 500 | 130 | 0.957 | 0.952 | 0.895 | 0.893 | Truncus arteriosus |
| AVCD | 1120 | 476 | 0.946 | 0.935 | 0.844 | 0.901 | Complete common atrioventricular canal |
| DTGA | 700 | 558 | 0.946 | 0.936 | 0.856 | 0.858 | D-TGA (Primary cardiac) |
| Pulm atresia | 1000 | 561 | 0.943 | 0.934 | 0.861 | 0.909 | Pulmonary atresia, congenital |
| PDA | 2100 | 3840 | 0.941 | 0.934 | 0.880 | 0.925 | Patent ductus arteriosus (Small) |
| DORV | 600 | 1017 | 0.936 | 0.927 | 0.854 | 0.882 | Double outlet right ventricle |
| LccTGA | 800 | 269 | 0.935 | 0.891 | 0.835 | 0.854 | L-TGA (Prior history of) |
| MS | 1510 | 1681 | 0.931 | 0.923 | 0.901 | 0.934 | Mitral stenosis (Residual\|Moderate-to-severe) |
| Ebstein | 1750 | 486 | 0.930 | 0.913 | 0.866 | 0.860 | Ebstein malformation (Primary cardiac) |
| Crit AS | 1411 | 1080 | 0.925 | 0.911 | 0.872 | 0.911 | Aortic stenosis, valvar (Residual\|Moderate) |
| IAA | 1250 | 52 | 0.917 | 0.897 | 0.787 | 0.668 | Interrupted aortic arch (Prior history of) |
| Coarct | 1200 | 1151 | 0.915 | 0.910 | 0.788 | 0.857 | Coarctation of the aorta |
| Crit PS | 1611 | 1296 | 0.908 | 0.898 | 0.820 | 0.870 | Pulmonary stenosis, valvar (Moderate-to-severe) |
| RAA | 2720 | 524 | 0.903 | 0.889 | 0.711 | 0.794 | Right aortic arch |
| Ring | 2760 | 192 | 0.885 | 0.877 | 0.718 | 0.721 | Vascular ring (Prior history of) |
| LPA sling | 2981 | 42 | 0.873 | 0.869 | 0.699 | 0.671 | PA sling, LPA from RPA (Prior history of) |
| AP window | 560 | 44 | 0.869 | 0.828 | 0.695 | 0.678 | Aortopulmonary window |
| VSD | 1300 | 1261 | 0.861 | 0.847 | 0.769 | 0.817 | VSD (Residual\|Large) |
| ASD | 2000 | 2080 | 0.842 | 0.826 | 0.693 | 0.817 | ASD, secundum (Small-to-moderate) |
| PAPVR | 2030 | 525 | 0.832 | 0.814 | 0.749 | 0.733 | Partially anomalous pulmonary veins (Probable\|RUPV) |
| BAV | 1401 | 1318 | 0.822 | 0.804 | 0.686 | 0.787 | Bicuspid aortic valve (Consequent to surgery) |
| DAA | 2761 | 63 | 0.803 | 0.774 | 0.753 | 0.660 | Double aortic arch (Probable) |
| LSVC | 2823 | 1697 | 0.803 | 0.784 | 0.725 | 0.824 | Left SVC to coronary sinus (Consequent to surgery) |
| TAPVR | 900 | 198 | 0.777 | 0.760 | 0.705 | 0.657 | TAPVR (Prior history of) |
| AnomCA | 3100 | 308 | 0.768 | 0.700 | 0.658 | 0.686 | Congenital coronary anomaly |
| Cor tri | 3031 | 61 | 0.765 | 0.806 | 0.700 | 0.635 | Cor triatriatum sinister (Primary cardiac\|Prior history of) |
 
JEPA v11 outperforms v6 on 31/32 diagnoses. JEPA line encoder outperforms PanEcho line encoder on 32/32 diagnoses. EchoVALE (JEPA v11) outperforms supervised EchoFocus CLIP probes on 30/32 diagnoses (exceptions: MS and LSVC, both high-n codes where the supervised probe has sufficient training examples to be competitive).

### vs. Supervised Classification

Using the same Fyler evaluation framework, a supervised classification architecture (binary probes over frozen video embeddings) was compared against EchoVALE across all evaluated Fyler codes. EchoVALE outperforms on the majority of codes, including common ones, despite never seeing a Fyler label during training.

### External Validation

- Outside referral cohort (~3,300 studies, 58 countries)
- On SG diagnosis labels, JEPA line encoder ties study-CLIP supervised probes (14 vs 13 wins on primary diagnoses)
- Strong zero-shot results on rare/structural CHD codes where line encoder beats ridge probes

## Downstream Applications

All are different readouts of the same trained system:

**Zero-shot classifier**: Score any clinical line against any study. No retraining when new codes appear. Encode the clinical description and score it.

**Line validator**: Score human-written report text against the echo. Any text <512 tokens (BERT context window). Natural deployment path for clinical report QA.

**Heatmap / report generation**: Score the full 180K line retrieval pool against a study. Hotspots identify clusters of related/variant lines for each finding. Output is a probability field over clinical language space.

## Related Work

- **EchoFocus-CHD** (Lukyanenko, Mayourian et al.): Supervised CHD detection on PanEcho. Preprint: doi:10.64898/2026.01.24.26344771
- **EchoJEPA** (Munim et al., arXiv 2026): Latent predictive foundation model trained on 18M echocardiograms across 300K adult patients. Demonstrates that JEPA pretraining produces representations robust to ultrasound speckle and acquisition artifacts, with strong zero-shot transfer to pediatric data. EchoVALE uses an independently trained JEPA backbone (ViT-B, initialized from V-JEPA 2.1) on BCH pediatric data as its vision encoder.
- **VL-JEPA** (Chen et al., Meta FAIR, Dec 2024): Vision-language JEPA that predicts continuous answer embeddings instead of generating tokens. EchoVALE shares the embedding-space prediction thesis but differs architecturally (cross-attention scoring vs. joint self-attention prediction) and constrains the answer space to binary relevance.
- **EchoPrime** (Vukadinovic et al., Nature 2025): Vision-Language infoNCE contrastive FM architecture for echos. Also uses Clinical BERT for line encoding but pre aggregates the entire report instead of keeping clinical lines seperate.  


## Folder Structure

```
video_pretraining_v2/       # EchoJEPA clip encoding
line_tokenizer_v2/          # Line encoder, CrossAttentionPool, training, eval, generation
  model.py                  # LineEncoder, CrossAttentionPool
  train.py                  # Skip-gram BCE training
  encode.py                 # Encode lines through trained LineEncoder
  generate_lines.py         # Heatmap generation (mutual KNN clustering)
  eval_fyler.py             # Fyler code AUROC evaluation
  eval_lines.py             # Line embedding UMAPs and diagnostics
  results/
    v6/                     # Current best
echofocus_mini/             # LVEF probes over video embeddings
eval_supervised.py          # Probe evaluation
UMAPs/
  ├── video_embeddings/
  ├── study_embeddings/
  └── Line_embeddings/
      ├── BCE_*/            # Vision-aligned line embeddings
      └── BERT_*/           # Raw ClinicalBERT line embeddings
unpredictable_fyler_codes.txt
line_filters.txt
```

## Environment

- **Cluster**: E3, partitions `bch-gpu-pe` / `bch-compute`; 
- **Conda env**: `/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env`
- **JEPA checkpoint**: `/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitb_bch/latest.pt`
- **Video embeddings**: `video_pretraining_v2/embeddings/jepa_clips_4x768_fixed.npz` (40M x 768)
- **Reports**: `echo_reports_v3.csv`
