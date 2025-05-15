# PRISM Dataset: Code for Data Processing and Visualization

This repository contains supporting code for the **PRISM dataset**, a curated collection of building geometries paired with simulation-ready environmental performance outputs (e.g., CFD, sky exposure). The code is intended to help users preprocess simulation outputs and explore dataset contents using a provided example geometry.

---

## Contents

This repository includes the following components:

### `vtk_to_npy.py`
Converts simulation output files from `.vtk` format to NumPy arrays (`.npy`) for efficient loading and downstream processing.  
 *Based on the [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library). See Attribution section below.*

###  `calc_rot_npy.py`
Post-processes `.npy` performance arrays by applying rotations to points to account for varying wind directions.

### `visualizePRISM.ipynb`
A Jupyter notebook to visualize all available modalities (geometry and performance) for a **single sample instance**, available in the dataset as **building_345**. Designed to support quick exploration and understanding of dataset structure.

---

## Dataset Access

The **PRISM dataset** is publicly hosted on **Harvard Dataverse**:

> 🔗 [https://dataverse.harvard.edu/previewurl.xhtml?token=57c1017c-2ff4-4b78-8f3e-4608b3ccb5ea](https://dataverse.harvard.edu/previewurl.xhtml?token=57c1017c-2ff4-4b78-8f3e-4608b3ccb5ea)

A complete **sample geometry folder**, **building_345**, is included in the dataset to support rapid testing and inspection. This sample contains both **raw simulation outputs** (e.g., `.vtk` files) and their corresponding **pre-processed representations** (e.g., `.npy` arrays), allowing users to test visualization, processing, and modeling pipelines directly. This repository assumes the sample folder is available locally and that relevant paths in the notebook and scripts are updated accordingly.

---

## License

The **code** in this repository is released under the [MIT License](https://opensource.org/licenses/MIT), permitting use, modification, distribution, and commercial reuse with attribution.

The **PRISM dataset** is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. This allows use, redistribution, and adaptation for any purpose, including commercial applications, provided that appropriate credit is given to the dataset creators.

---

## Attribution

The script `vtk_to_npy.py` is adapted in part from the [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library), which is also released under the MIT License.
Please refer to their `README.md` and `requirements.txt` for any additional dependencies or setup instructions specific to their framework.

---

## Requirements

The code relies on standard Python scientific computing libraries. Required packages include:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `plotly`
- `trimesh`
- `joblib`
- `pathlib` (standard library)
- `os` (standard library)

Install with:

```bash
pip install numpy pandas matplotlib scipy plotly trimesh joblib
