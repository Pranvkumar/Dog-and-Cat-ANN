# Dog vs Cat ANN

This project contains a TensorFlow/Keras notebook for binary image classification of cats vs dogs using a dense neural network (ANN) pipeline.

## Contents

- `dog_vs_cat_ann_pipeline.ipynb`: End-to-end training, evaluation, and visualization notebook.
- `afhq/`: Dataset folder with train/validation splits.

## Expected Dataset Layout

```text
afhq/
  train/
    cat/
    dog/
    wild/
  val/
    cat/
    dog/
    wild/
```

Note: The notebook is configured for binary classification with `cat` and `dog` classes. The `wild` class is present in the dataset but is not used by default.

## Environment

Recommended Python packages:

- tensorflow
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## How To Run

1. Open `dog_vs_cat_ann_pipeline.ipynb`.
2. Run the import and runtime setup cell first.
3. Run the data loading cell.
4. Run the model training and evaluation cell.

The notebook automatically checks GPU availability and falls back to CPU if GPU is not available.

## Notes

- Default image size: `64x64`.
- Batch size is adaptive:
  - `64` when GPU is available
  - `32` on CPU
- Training uses EarlyStopping and ReduceLROnPlateau callbacks.
