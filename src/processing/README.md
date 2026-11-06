# Processing

**[embryo_labeler.py](embryo_labeler.py)**: A napari tool for labeling nuclear-cycle stages. For each stage, click a button to mark the frame where that stage starts. Each confirm writes the embryo's entry to `labels.yaml` and moves its images into a `labeled/` folder. It accepts a single file, a flat folder, or a folder of channel subfolders, where matching positions open together as layers.

```bash
python src/processing/embryo_labeler.py <path> --yaml labels.yaml [--labeled-dir DIR] [--time-between-frames 60] [--include-labeled]
```

**[extract_embryo.py](extract_embryo.py)**: Segments embryos with a custom Cellpose model and rotates, centers and scales each one into a canonical 800×800 frame.
- `EmbryoExtractor` handles one embryo per field of view. It processes single frames (live) or whole videos.
- `MultipleEmbryoExtractor` handles several embryos per field of view and tracks each one across frames under a stable ID. It is used by `ClassifyMultipleEmbryos` in PyCLM.
