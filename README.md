### 3DCellPol
Official implementation of 3DCellPol, an automated approach for joint detection and pairing of cell organelles, as described in [3DCellPol: Joint Detection and Pairing of Cell Structures to Compute Cell Polarity](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4947066).

This repository contains the Python implementation of CellPol for both 2D and 3D images.


## Important Notes

* *Important*: We welcome collaborations to expand the dataset for 3DCellPol training and evaluation. Feel free to contact me hemaxi[dot]narotamo[at]tecnico.ulisboa.pt

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

Our approach predicts centroid heatmaps and embedding maps separately for nuclei and Golgi. Firstly, the centroids are obtained by extracting the local maximizers from the centroid heatmaps.  
The embedding for each detected centroid is obtained by extracting the value, at the centroid's position, from the embedding heatmap, represented by vertical dashed grey arrows in the following figure.
After that, pairing is based on the distance between a nucleus embedding and a Golgi embedding. Nuclei and Golgi with the closest embeddings are paired, as long as the distances between their centroids and between their embeddings are smaller than certain thresholds. 

![](https://github.com/HemaxiN/3DCellPol/blob/main/images/overview.jpg)
**Schematic representation of 3DCellPol** It receives an image with nuclei and Golgi as input, and outputs the centroid heatmaps and embedding maps of nuclei and Golgi. The centroids are decoded by extracting the local maxima from the centroid heatmaps. The corresponding embeddings are obtained from the embedding maps at the centroids' positions (vertical dashed grey arrows). A nucleus and a Golgi belonging to the same cell should have similar embeddings. Thus, the distances between the embeddings of nuclei and Golgi are used to compute the vectors.


## Datasets

* Real Nucleus-Golgi Retinal Dataset (3D): microscopy subvolumes of mouse retinal nuclei and Golgi and corresponding subvolumes with centroid heatmaps and embedding maps required for training, it is available [here](https://huggingface.co/datasets/Hemaxi/3DCellPol).
* Cytoplasm-Nucleus Dataset (2D)
  This dataset was presented in [Cx22: A new publicly available dataset for deep learning-based segmentation of cervical cytology images](https://www.sciencedirect.com/science/article/pii/S0010482522009027) and it is available [here](https://github.com/LGQ330/Cx22).

## Requirements

The code was initially developed in Python 3.5.2, using Keras 2.2.4. 

Now it has been updated to work in Python 3.10, the required packages are listed in [requirements.txt](https://github.com/HemaxiN/3DCellPol/blob/main/requirements.txt).

# Testing the Pre-Trained Model on Your Own Dataset

You can test the pre-trained model, available [here](https://huggingface.co/Hemaxi/3DCellPol) in your dataset using the ```predict_main.py``` as described in the [Prediction section](#prediction). If the pre-trained model does not work well on your images, you can train the 3DCellPol model using your dataset.

# Train on your own dataset

To train the 3DCellPol model, you need images and corresponding 3D polarity vectors.
The 3D nucleus-Golgi vectors can be annotated using our [Vector Annotation Tool](https://github.com/HemaxiN/VectorAnnotationTool). This tool outputs a .csv files for each image containing the paired nuclei and Golgi centroids. 
To create the training sub-volumes use the file [create_dataset_main.py](https://github.com/HemaxiN/3DCellPol/blob/main/3D/create_dataset_main.py), after updating the directory paths, image resolution information and other parameters. The `gt_dir` is the path to the folder containing the .csv files obtained with the [Vector Annotation Tool](https://github.com/HemaxiN/VectorAnnotationTool).
Then, run the file [train_main.py](https://github.com/HemaxiN/3DCellPol/blob/main/3D/train_main.py) to create the ```train``` and ```val``` folders in ```data_dir```. The ```data_dir```has the following structure: 

```
data_dir
    ├──  train
    |   ├── images  0.npy, 1.npy, ..., N.npy (sub-volumes of microscopy images of vessels  (X_dim, Y_dim, Z_slices, 2))
    |   ├── vectors 0nuclei.npy, 0golgi.npy, 1nuclei.npy, 1golgi.npy, ..., Nnuclei.npy, Ngolgi.npy  (arrays with the positions of the nuclei centroids and the centroids of the corresponding Golgi centroids)
    |   └── outputs   0.npy, 1.npy, ..., N.npy (sub-volumes of Gaussian heatmaps and embedding maps  (X_dim, Y_dim, Z_slices, 4))
    └──  val
        ├── images  0.npy, 1.npy, ..., M.npy (sub-volumes of microscopy images of vessels  (X_dim, Y_dim, Z_slices, 2))
        ├── vectors 0nuclei.npy, 0golgi.npy, 1nuclei.npy, 1golgi.npy, ..., Mnuclei.npy, Mgolgi.npy  (arrays with the positions of the nuclei centroids and the centroids of the corresponding Golgi centroids)
        └── outputs   0.npy, 1.npy, ..., M.npy (sub-volumes of Gaussian heatmaps and embedding maps  (X_dim, Y_dim, Z_slices, 4))
```

# Prediction

### 1. Predict the results for the validation subvolumes
- Begin by updating the directory paths, image resolution information and other parameters in the [predict_validation.py](https://github.com/HemaxiN/3DCellPol/blob/main/3D/optimization/predict_validation.py) file and run it.  
- Repeat the process for the following files:
  - [select_best_thresholds_golgi.py](https://github.com/HemaxiN/3DCellPol/blob/main/3D/optimization/select_best_thresholds_golgi.py)
  - [select_best_thresholds_nuclei.py](https://github.com/HemaxiN/3DCellPol/blob/main/3D/optimization/select_best_thresholds_nuclei.py)

### 2. Compute Optimal Thresholds
- Run the following files, after updating the directory paths, to output the optimal thresholds for predictions based on the validation set:
  - [roc_curve_golgi.py](https://github.com/HemaxiN/3DCellPol/blob/main/3D/optimization/roc_curve_golgi.py)
  - [roc_curve_nuclei.py](https://github.com/HemaxiN/3DCellPol/blob/main/3D/optimization/roc_curve_nuclei.py)

### 3. Run Predictions
- Finally, update the directory paths, the computed threshold values, image resolution information and other parameters in the [predict_main.py](https://github.com/HemaxiN/3DCellPol/blob/main/3D/predict_main.py) file and run it to test the model.

# Evaluation

Update the diretory paths and image resolution information in [evaluation_main.py](https://github.com/HemaxiN/3DCellPol/blob/main/3D/evaluation_main.py). The evaluation results (.csv file) will be saved in `eval_dir`.

# Polar/Angular Histograms

CellPol outputs can be used to study the polarization and migration of cells. To help visualize this, we provide a script to visualize the angular histograms for the distribution of vector orientations. Specifically:

- 2D vectors: in the XY plane.
- 3D vectors: in the XY, YZ, and XZ planes.

In these polar histograms:

- The angular axis represents direction in degrees (0° to 360°).
- The radial axis represents vector density, showing how frequently vectors are oriented in each direction.
- Example of a polar histogram:

![](https://github.com/HemaxiN/3DCellPol/blob/main/images/polar_histogram_example.png)

To visualize these plots, update the directory paths and select the plane (xy, yz, or zx) by updating variable `plane_` in [angular_histograms.py](https://github.com/HemaxiN/3DCellPol/blob/main/3D/angular_histograms.py) file and run it.
