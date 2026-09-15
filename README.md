# Semantic Segmentation with U-Net

Pixel-level semantic segmentation on street-driving scenes, using a U-Net built from scratch in PyTorch and served through a FastAPI endpoint.

[Bahasa Indonesia](README.id.md)

## Overview

- **Task:** 13-class semantic segmentation (road, vehicles, trees, sky/unlabeled, and others) on 256×256 RGB images captured from the CARLA driving simulator.
- **Architecture:** U-Net (encoder-decoder with skip connections), implemented from scratch in PyTorch — no pretrained backbone.
- **Loss:** Multiclass Dice Loss (`segmentation-models-pytorch`).
- **Metric:** Dice Score (`torchmetrics.segmentation.DiceScore`, index-format input, micro-averaged).

## Results

| | |
|---|---|
| Dataset size | 1,000 images, 13 classes |
| Train / test split | 800 / 200 (80/20) |
| Epochs | 200 |
| Best checkpoint | epoch 196 |
| Eval loss (best) | 0.1068 |
| Dice Score (best) | 0.9756 |

Numbers above come from a completed 200-epoch training run in Google Colab (GPU runtime); the checkpoint with the lowest evaluation loss is the one saved as `best.pt`.

![Training and evaluation curves](assets/training-curve.png)

![Sample segmentation output](assets/sample-segmentation-result.png)

## API demo

The serving notebook exposes an interactive `/docs` page (Swagger UI) through the FastAPI + ngrok tunnel. The screenshot below shows a successful `POST /segment-image/` request returning a `200 OK` with the base64-encoded mask.

![Serving endpoint demo](assets/api-demo.png)

## Project structure

```
semantic-segmentation-unet/
├── notebooks/
│   ├── semantic-segmentation-unet-training.ipynb
│   └── semantic-segmentation-unet-serving.ipynb
├── assets/
│   ├── training-curve.png
│   ├── sample-segmentation-result.png
│   └── api-demo.png
├── README.md
├── README.id.md
├── LICENSE
└── .gitignore
```

## How to run

1. **Training** — open `notebooks/semantic-segmentation-unet-training.ipynb` in Google Colab (GPU runtime recommended) and run all cells. This downloads the CARLA capture dataset, trains the U-Net for 200 epochs, and saves the best checkpoint (`best.pt`) to Google Drive.
2. **Serving** — open `notebooks/semantic-segmentation-unet-serving.ipynb`, mount the same Google Drive, and run all cells. This starts a FastAPI server exposed through ngrok with a `/segment-image/` endpoint that accepts an image upload and returns a base64-encoded segmentation mask.

## Notes on implementation

- Postprocessing originally mapped model output to a mask using a fixed probability threshold, which turned out to produce empty (all-background) masks even when the underlying model scored well on Dice. Switched to per-pixel `argmax`, which gives a deterministic single-class prediction per pixel and matches what the evaluation metric already assumes.
- Model weights (`best.pt`) are not included in this repo — run the training notebook to generate your own, or bring your own checkpoint.
- The serving notebook expects an ngrok auth token stored as a Colab secret (`NGROK_AUTH_TOKEN`), not hardcoded in the notebook.

## Dataset

[CARLA capture dataset](https://github.com/ongchinkiat/LyftPerceptionChallenge) (`carla-capture-20180513A.zip`), RGB + segmentation label pairs generated from the CARLA simulator.

## Tech stack

PyTorch, segmentation-models-pytorch, torchmetrics, albumentations, FastAPI, pyngrok, Google Colab.

## License

MIT — see [LICENSE](LICENSE).
