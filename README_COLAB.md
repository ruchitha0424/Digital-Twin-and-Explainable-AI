# Cardio Digital Twin — Google Colab Package

This ZIP package contains everything you need to run the multi-model machine learning benchmarking and evaluation suite directly in Google Colab.

## Contents of this Package:
1. `cardio_model_benchmark_colab.ipynb` - The complete, self-contained Google Colab notebook.
2. `data/cardio_train.csv` - The benchmark 70,000-sample clinical cardiovascular dataset.
3. `data/cardio_cleaned.csv` - Cleaned baseline dataset.
4. `models/data_cleaner.py` - Modular clinical data cleaner and hemodynamic feature engineering script.
5. `models/train_all_models.py` - Standalone multi-model benchmarking script.
6. `TECHNIQUES_AND_METHODOLOGY_NOTE.md` - Full technical reference containing mathematical formulas and clinical rationales for research writing.

---

## How to Use in Google Colab (2 Easy Options):

### Option 1: Direct Notebook Upload (Simplest)
1. Unzip this package on your computer.
2. Go to [colab.research.google.com](https://colab.research.google.com/) and click **Upload**.
3. Choose `cardio_model_benchmark_colab.ipynb`.
4. Run the notebook. When prompted by the upload button in Cell 2, upload `data/cardio_train.csv`.
5. That's it! The notebook will train all 7 models, evaluate all 10 metrics, plot 7 high-res figures, and automatically download `cardio_colab_benchmark_results.zip`.

### Option 2: Upload the Entire ZIP into Colab
1. In Google Colab, open a new notebook or upload `cardio_model_benchmark_colab.ipynb`.
2. Upload this `cardio_colab_package.zip` into the Colab Files tab on the left.
3. In a code cell, run:
   ```python
   !unzip -o cardio_colab_package.zip
   ```
4. All data, scripts, and notebooks will be extracted into the Colab runtime ready to execute immediately.
