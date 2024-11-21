### 3DCellPol
Official implementation of 3DCellPol an automated approach for joint detection and pairing of cell organelles, as described in [3DCellPol: Joint Detection and Pairing of Cell Structures to Compute Cell Polarity](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4947066).

This repository contains the Python implementation of CellPol for both 2D and 3D images.

## Important Note

* The implementation for 3D vector prediction will be incorporated into our recently proposed [3DVascNet's](https://github.com/HemaxiN/3DVascNet/wiki) graphical user interface.
* The instructions below are for the **3D implementation** provided in the `3D` folder. If you are using the **2D implementation** in the `2D` folder, the steps are almost identical. 

For the 2D implementation, there are only two differences:
1. **No need to specify the z-dimension**: You do not need to provide the patch dimensions along the z-direction.
2. **No need to specify z-direction resolution**: The resolution along the z-direction is also not required.

All other steps remain the same.

## How to cite

```bibtex
@article{,
  title={3DCellPol: Joint Detection and Pairing of Cell Structures to Compute Cell Polarity},
  author={Narotamo, Hemaxi and Franco, Cl{\'a}udio A and Silveira, Margarida},
  journal={},
  year={},
  publisher={}
}
```

## Overview of the Project



## Datasets

* Real Nucleus-Golgi Retinal Dataset (3D)
* Cytoplasm-Nucleus Dataset (2D)
  This dataset was presented in [Cx22: A new publicly available dataset for deep learning-based segmentation of cervical cytology images](https://www.sciencedirect.com/science/article/pii/S0010482522009027) and it is available [here](https://github.com/LGQ330/Cx22).
* Real Nucleus-Golgi Cells Dataset (2D)

## Requirements

The code was initially developed in Python 3.5.2, using Keras 2.2.4. 

Now it has been updated to work in Python 3.10, the required packages are listed in [requirements.txt](https://github.com/HemaxiN/3DCellPol/blob/main/requirements.txt).

# Testing the Pre-Trained Model on Your Own Dataset

# Train on your own dataset

# Prediction

# Polar/Angular Histograms

CellPol outputs can be used to study the polarization and migration of cells. To help visualize this, we provide a script to visualize the angular histograms for the distribution of vector orientations. Specifically:

- 2D vectors: in the XY plane.
- 3D vectors: in the XY, YZ, and XZ planes.

In these polar histograms:

- The angular axis represents direction in degrees (0° to 360°).
- The radial axis represents vector density, showing how frequently vectors are oriented in each direction.
- Example of a polar histogram:

![](https://github.com/HemaxiN/3DCellPol/blob/main/images/polar_histogram_example.png)
