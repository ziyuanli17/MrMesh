# MrMesh: MR Images to Meshes
## _Automatically generate cardiac segmentations, contours, and meshes from SAX MR images_
[![Overview](https://drive.google.com/uc?export=view&id=1xK7LrTNp9QFfqOYjSPxX0U1QitpnuZ4Y)](https://drive.google.com/uc?export=view&id=1GEgoLsMAe0Ea-mp36cnNft__zp9MSVN)
### Latest Release: [Here](https://) 

## Features
- Generate 2D pixelwise segmentations (currently only support left ventricle) using [mUNet](https://jcmr-online.biomedcentral.com/articles/10.1186/s12968-018-0471-x) 
- Process mUnet outliers using [EF](https://docs.google.com/document/d/15KwaYJncmZG8PiatVQ2rn-Fxw9KLR3YiaoLPZSIU4w4/edit?usp=sharing), a novel computer vision based approach to generate anatomically plausible segmentations
- Extract endocardium and epicardium contours from the segmentations 
- Generate 2D & 3D surface meshes, and 3D volume mesh
- Refine meshes to a desired grid resolution

# User Instruction
## Preparations
1. Download the [stable release](https://github.com/ziyuanli17/CardioFree/releases) and extract the ziped file
2. Run **GUI.exe** (Currently only support Windows machine)

[![Overview](https://drive.google.com/uc?export=view&id=18pGqp7AB0WF2xRfH28gvIx5gHtXuUnS7)](https://drive.google.com/uc?export=view&id=1GEgoLsMAe0Ea-mp36cnNft__zp9MSVN)

**NOTE**: the input folder contains two example dicom files. example1: segmentation works using mUNet only.example2: mUnet outlier, EF required. If you are using your own file(s), make sure they match the examples' file type and dimensions (2D images + time series + slices along z)
3. Wait until the mUnet segmentation finishes and the results are displayed

**NOTE**: All outputs will be saved in the **Output folder**

Ready to generate all hell through a 3-step magic?

## (1) Segmentation: mUnet
The UNet segmentations are displayed along with their MRI at end of distole. Click on **EF Postprocessing** if the segmentations are undesirable. Otherwise, skip ***(2)*** and go to ***Step (3)***

[![Overview](https://drive.google.com/uc?export=view&id=1gEwGDEfoeGT5IndnJo5cx452tusP1GOd)](https://drive.google.com/uc?export=view&id=1GEgoLsMAe0Ea-mp36cnNft__zp9MSVN)
## (2) Postrocessing: EF
In EF prostprocessing you can adjust the parameters described by the [publication](https://docs.google.com/document/d/15KwaYJncmZG8PiatVQ2rn-Fxw9KLR3YiaoLPZSIU4w4/edit?usp=sharing) and generate optimal segmentations. 
**NOTE**: Please make sure you refer to the **Parameter Optimization** section before you start adjusting.
[![Overview](https://drive.google.com/uc?export=view&id=1LQ_zQIOpU2p_cq6eZ0VpeDALuaMMwu8U)](https://drive.google.com/uc?export=view&id=1GEgoLsMAe0Ea-mp36cnNft__zp9MSVN)
**NOTE**: After you finish adjusting the slide bars, click on **Run EF Postprocessing** to display the new results.

If the software fails to automatically detects LV center, you may click on **LV Manual Localization** and click on the desired LV center to use.
[![Overview](https://drive.google.com/uc?export=view&id=1qXS8Wb6_y23GW6oVv9SyYl7A_dUXuxOj)](https://drive.google.com/uc?export=view&id=1GEgoLsMAe0Ea-mp36cnNft__zp9MSVN)

If you wish to see more details, click on **Show EF Process** that shows the stepwise results by each operation described in the [publication](https://docs.google.com/document/d/15KwaYJncmZG8PiatVQ2rn-Fxw9KLR3YiaoLPZSIU4w4/edit?usp=sharing).
[![Overview](https://drive.google.com/uc?export=view&id=10jDehKj2oUogPbfODrdMYkMCA11srTmb)](https://drive.google.com/uc?export=view&id=1GEgoLsMAe0Ea-mp36cnNft__zp9MSVN)

## (3) Mesh Generation
When you are satisfied with your segmentations, click on **Mesh Generation**. Then, you can adjust all the mesh parameters (in most applications the default should be fine) but you may refer to the [publication](https://www.mdpi.com/2313-433X/4/1/16) for details. 

In brief:
- MAX_DEFORMATION_ITS: Number of iterations to run SurfCo (for surface mesh generation). More iterations usually result in finer resolution and more deformed (toward the coutours) mesh but larger computational cost.
- TARGETREDUCTION: controls how long the edges are shortened in each iteration. An inrease/decrease leads to **lower/higher** grid resolution of the 3D surface mesh.  **NOTE**: adjust very gently (e.g. +-0.01).
- RESOLUTION: An inrease/decrease leads to **higher/lower** grid resolution of the 3D surface mesh. Controls the density of intial mesh built (before deformation). Less sensitive.
- PERCENTAGE: Strength of pulling force toward the countours when building the surface mesh.
- SMOOTH_STRENGTH and SMOOTH_LAMBDA: Controls smoothing strength of the 3D surface generated.
- REFINEITER2D and REFINEITER3D: Number of times to divide the lenghth of 2D surface and 3D volume meshes. More iterations result in increased grid resolution.

[![Overview](https://drive.google.com/uc?export=view&id=1oKo2XVM3P1lfhLuqtQjnU3Xgaz-nkkU4)](https://drive.google.com/uc?export=view&id=1GEgoLsMAe0Ea-mp36cnNft__zp9MSVN)

# Developer Instruction
## Dependencies
Read requirements.txt for a list of python packages used and their references.

To install all, run the command below.
```sh
pip install -r requirements.txt
```
## Additional Pacakges
2 pip-uninstallable packages are required but are already included in this repo (No need to reinstall). For details, refer to the git repositories below
**Segmentation**: [mUNet](https://jcmr-online.biomedcentral.com/articles/10.1186/s12968-018-0471-x)
**Surface Mesh Generation**: [SurfCo](https://github.com/BenVillard/surfco/blob/master/README.md)

