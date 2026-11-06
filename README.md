# Computer Vision Analysis of Drosophila Embryo

Live staging of *Drosophila* embryos by nuclear cycle (NC10–NC14+) from brightfield microscopy video. The repo covers preprocessing and labeling of training data, training of CNN + HMM staging models, and closed-loop use on the microscope via [PyCLM](../pyclm), where embryos are optogenetically stimulated once they reach a target stage.

## Pipeline

1. **Embryo extraction** ([src/processing/extract_embryo.py](src/processing/extract_embryo.py))
   Segments embryos with a custom Cellpose model (`embryomodel` in the Cellpose model directory) and rotates/centers/scales each one into a canonical 800×800 frame.
   - `EmbryoExtractor`: one embryo per field of view.
   - `MultipleEmbryoExtractor`: several embryos per field of view, each tracked across frames under a stable ID.

2. **Staging models** ([src/classification/](src/classification/))
   An EfficientNet-B3 CNN ([cnn_classifier.py](src/classification/cnn_classifier.py)) takes a sliding window of recent frames as input channels and outputs per-frame stage probabilities. An LSTM ([lstm_classifier.py](src/classification/lstm_classifier.py)) can optionally be stacked on the CNN features. A temporal model then turns the CNN output into a stage prediction that only moves forward. There are three variants, each with a trainer and a live predictor:

   | Variant | Trainer / Predictor | Temporal model |
   |---|---|---|
   | Pure HMM | [pure_hmm/](src/classification/pure_hmm/) | Learned transition matrix + online Viterbi |
   | Hybrid HMM | [hybrid_hmm/](src/classification/hybrid_hmm/) | Log-normal stage-duration model × CNN probabilities (see [design.md](src/classification/hybrid_hmm/design.md)) |
   | Combined HMM | [combined_hmm/](src/classification/combined_hmm/) | Pure HMM for early stages, switching to the hybrid duration model once the CNN predicts `switch_state` or later |

   Shared logic lives in [base_hmm_trainer.py](src/classification/base_hmm_trainer.py) and [base_hmm_predictor.py](src/classification/base_hmm_predictor.py). Predicted states are `undetectable, NC10, NC10M, NC11, NC11M, NC12, NC12M, NC13, NC13M, NC14+` (`M` = mitosis).

3. **Closed-loop control** ([PyCLM/](PyCLM/))
   [pyclm_classifier.py](PyCLM/pyclm_classifier.py) registers PyCLM pattern methods (`classify_embryo`, `classify_multiple_embryos`). At each timepoint they extract the embryo(s), run a live predictor, log the predicted stages to `PyCLM/classification_logs.json`, and project a stimulation pattern onto embryos that have reached the target stage. Experiment configs are the `.toml` files in that folder.

## Training data

Trainers expect this layout under `data/training_data/`:

```
data/training_data/
  labels.yaml          # stage annotations
  brightfield/         # raw stacks, <embryo>.tif
  processed_tifs/      # extracted 800x800 stacks (generated when preprocess_images=True)
```

`labels.yaml` maps each embryo to its frame interval and the first frame of each stage:

```yaml
0629_pos010:
  time_between_frames: 60
  undetectable: 1
  NC10: 14
  NC10M: 22
  ...
```

## Training and running

The `__main__` block of each trainer and predictor sets its own paths and hyperparameters; edit those, then run, e.g.:

```bash
uv run python src/classification/combined_hmm/combined_hmm_trainer.py
uv run python src/classification/combined_hmm/combined_hmm_predictor.py
```

Training writes to `models/<variant>/`: CNN/LSTM checkpoints, confusion-matrix heatmaps, per-embryo progression plots, and a `<variant>_model_info.json` with the model paths, train/val split, and learned transition and duration parameters. Predictors and the PyCLM script load the model from this JSON file.

## Project layout

```
src/
  processing/         embryo segmentation/extraction
  classification/     CNN/LSTM classifiers, HMM trainers and live predictors
PyCLM/                closed-loop microscope control and experiment configs
data/                 training/inference data (gitignored)
models/               trained checkpoints and model info (gitignored)
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/). `pyclm` is installed as an editable dependency from a sibling checkout at `../pyclm`, so clone that first.

```bash
uv sync
```
