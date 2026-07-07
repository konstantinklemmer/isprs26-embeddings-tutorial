# Towards Geospatial Embeddings — ISPRS 2026 Tutorial

Code and notebooks for the **ISPRS Congress 2026** tutorial
*"Towards Geospatial Embeddings: Investigating Accurate and Accessible Deep Geospatial
Feature Representations"*
([tutorial info](https://www.isprs2026toronto.com/tutorial-session-information)).

The tutorial pairs lectures with two hands-on coding sessions. Both notebooks are designed to
run **start-to-finish on a free Google Colab CPU runtime** — no GPU, and no Earth Engine
account required.

📑 **Lecture slides:** [`slides/ISPRS_Tutorial_EE.pdf`](slides/ISPRS_Tutorial_EE.pdf).

## The two demos

| | Notebook | What you do | Open in Colab |
|---|---|---|---|
| **Demo 1** | [`demo1_using_earth_embeddings.ipynb`](notebooks/demo1_using_earth_embeddings.ipynb) | *Use* pre-made embeddings for prediction: coarse global **SatCLIP** location embeddings → ecoregion/biome; fine-grained **AlphaEarth** pixel embeddings → Canadian crop-type mapping. Includes a satellite-imagery baseline and **geographic transfer** — hold out whole continents (a configurable leave-one-continent-out sweep) and map *where* out-of-domain predictions fail. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konstantinklemmer/isprs26-embeddings-tutorial/blob/main/notebooks/demo1_using_earth_embeddings.ipynb) |
| **Demo 2** | [`demo2_producing_earth_embeddings.ipynb`](notebooks/demo2_producing_earth_embeddings.ipynb) | *Produce* your own embeddings with the training-free **MOSAIKS** random convolutional features, see how **spectral bands** and **image size** change downstream accuracy, then compare against a **pretrained SSL4EO-S12 foundation model** loaded from the Hub. Ends with a similarity ("find places like this") map. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konstantinklemmer/isprs26-embeddings-tutorial/blob/main/notebooks/demo2_producing_earth_embeddings.ipynb) |

## How the data works

To keep the live sessions fast and dependency-free, each notebook downloads small, **pre-packaged
datasets** from a public Hugging Face dataset repo
([`kklmmr/isprs26-earth-embeddings`](https://huggingface.co/datasets/kklmmr/isprs26-earth-embeddings)) —
no login, no Earth Engine. The scripts that build those files from the original open sources live in
[`scripts/`](scripts/) — see [`scripts/README.md`](scripts/README.md) to regenerate them.

```
notebooks/   the two demo notebooks (the things participants run)
slides/      lecture slides (PDF)
scripts/     offline data-prep (organizers only; some steps need a GEE project)
tutorial/    small reference library (RCF featurizer) used by tests/prep
tests/       unit tests for the reusable code
local/       staged data before upload to Hugging Face (git-ignored)
```

## Data sources & licensing (all free / open)

- **SatCLIP** location embeddings — Klemmer et al., Microsoft ([repo](https://github.com/microsoft/satclip)).
- **AlphaEarth Foundations** "Satellite Embedding" — produced by Google and Google DeepMind; CC-BY 4.0
  ([Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)).
- **RESOLVE Ecoregions 2017** — CC-BY 4.0 ([ecoregions.world](https://ecoregions.world/)).
- **AAFC Annual Crop Inventory** — Agriculture and Agri-Food Canada, Open Government Licence – Canada.
- **Sentinel-2** — Copernicus / ESA.
- **EuroSAT** — Helber et al., 2019 (Sentinel-2 land-cover patches).
- **SSL4EO-S12** pretrained ResNet-18 — Wang et al., 2022; weights pulled from the public HF repo
  [`torchgeo/resnet18_sentinel2_all_moco`](https://huggingface.co/torchgeo/resnet18_sentinel2_all_moco) (via TorchGeo).

## Going further

### Free LGND Embeddings API access for researchers

Academics can get **premium access to [LGND](https://lgnd.ai)'s Embeddings API for free** —
learn more and apply via the [research tier](https://lgnd.ai/resources/research-tier).

### Readings

- Klemmer, Konstantin, et al. "Earth Embeddings: Towards AI-centric Representations of our Planet." *IEEE GRSM* (2026). [[EarthArXiv]](https://eartharxiv.org/repository/view/11083/)
- Fang, Heng, et al. "Earth Embeddings as Products: Taxonomy, Ecosystem, and Standardized Access." *arXiv* (2026). [[arXiv]](https://arxiv.org/abs/2601.13134)
- Rolf, Esther, et al. "Mission Critical — Satellite Data is a Distinct Modality in Machine Learning." *ICML* (2024). [[arXiv]](https://arxiv.org/abs/2402.01444)
- Corley, Isaac, et al. "No One Knows the State of the Art in Geospatial Foundation Models." *arXiv* (2026). [[arXiv]](https://arxiv.org/abs/2605.12678)
- Betti, Livia, et al. "What's in an Earth Embedding? An Explainability Analysis of Location Encoders." *arXiv* (2026). [[arXiv]](https://arxiv.org/abs/2606.24997)
- Kaur, Amandeep, et al. "Pretrain Where? Investigating How Pretraining Data Diversity Impacts Geospatial Foundation Model Performance." *CVPR* (2026). [[arXiv]](https://arxiv.org/abs/2604.21104)
- van der Plas, Thijs L., et al. "Better Together: Evaluating the Complementarity of Earth Embedding Models." *arXiv* (2026). [[arXiv]](https://arxiv.org/abs/2605.18667)
- Gilch, Luis, et al. "How to Embed Matters: Evaluation of EO Embedding Design Choices." *CVPR* (2026). [[arXiv]](https://arxiv.org/abs/2603.10658)
- Vinge, Rikard, et al. "NeuCo-Bench: A Novel Benchmark Framework for Neural Embeddings in Earth Observation." *CVPR* (2026). [[arXiv]](https://arxiv.org/abs/2510.17914)
- Corley, Isaac, et al. "From Pixels to Patches: Pooling Strategies for Earth Embeddings." *arXiv* (2026). [[arXiv]](https://arxiv.org/abs/2603.02080)

### Talks

- [Bad Tables: Why You Shouldn't Trust Results Tables in RS Foundation Model Papers](https://youtu.be/oKkzFSrKzEA) — Anthony Fuller
- [2025 LIDS Seminar — Konstantin Klemmer (Microsoft Research)](https://www.youtube.com/watch?v=oDZrZXSakfY) — MIT LIDS
- [Earth Embeddings: Learning Mental Maps in Neural Nets](https://youtu.be/Mnjrh-uc2Os) — AI + Environment Summit 2025

### Community

- **TorchGeo** — join the community on [Slack](https://torchgeo.org/).

## License

Code released under the MIT License (see `LICENSE`). Data subject to the licenses above.
