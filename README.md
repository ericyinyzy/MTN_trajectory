**MTN_trajectory**: Multimodal Transformer Networks for Pedestrian Trajectory Prediction
=======

![logo](https://github.com/ericyinyzy/MTN_trajectory/blob/main/logo/MTN.png)

😎PyTorch(1.6.0) training, evaluating models for MTN_trajectory.
For details see [Multimodal Transformer Networks for Pedestrian Trajectory Prediction](https://doi.org/10.24963/ijcai.2021/174) by Ziyi Yin, Ruijin Liu, Zhiliang Xiong, Zejian Yuan.

## Data Preparation
* PIE Dataset

Enter the PIE directory.

```
cd path/to/MTN_trajectory/PIE/
```

Download and extract PIE dataset:  
```
git clone https://github.com/aras62/PIE.git
mv PIE PIE_dataset
unzip -d PIE_dataset/ PIE_dataset/annotations.zip
unzip -d PIE_dataset/ PIE_dataset/annotations_attributes.zip
unzip -d PIE_dataset/ PIE_dataset/annotations_vehicle.zip
mv PIE_dataset/pie_data.py ./
```
Download and extract [optical flow representations of PIE](https://drive.google.com/file/d/1RhsaPAAm90L8pZLJIrzd1VRN4_z09_na/view?usp=sharing) from google drive. 
We expect the directory structure to be follwing: 
```
path/to/MTN_trajectory/
    PIE/
        PIE_dataset/
        PIE_model/
        flow/
        transformer/
        pie_data.py
        individual_TF.py
        baselineUtils.py
        train_pie.py
        test_pie.py
    JAAD/
```

* JAAD Dataset

Enter the JAAD directory.

```
cd path/to/MTN_trajectory/JAAD/
```

Download and extract JAAD dataset:  
```
git clone https://github.com/ykotseruba/JAAD.git
mv JAAD JAAD_dataset
mv JAAD_dataset/jaad_data.py ./
```
Download and extract [optical flow representations of JAAD](https://drive.google.com/file/d/1Zmf7H_mKlmnCmB-wn4X8EfFqMe3Z4w33/view?usp=sharing) from google drive. 
We expect the directory structure to be follwing: 
```
path/to/MTN_trajectory/
    PIE/
    JAAD/
        JAAD_dataset/
        JAAD_model/
        flow/
        transformer/
        jaad_data.py
        individual_TF.py
        baselineUtils.py
        train_jaad.py
        test_jaad.py
```

## Set Envirionment

* Linux ubuntu 16.04


```
conda create -n MTN python=3.7.9
```

After you create the environment, activate it

```
conda activate MTN 
```

Then

```
pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.19.4
pip install scikit-learn==0.23.2
pip install opencv-python==4.4.0.46
pip install tqdm
```

## Training and Evaluation

* on PIE Dataset

To train a model on PIE dataset, run following codes:
```
cd path/to/MTN_trajectory/PIE
python train_pie.py
```
Saved model files (every 10 epoches) are in ./PIE_model during training.

To evaluate, run:
```
python test_pie.py
```
* on JAAD Dataset

To train a model on JAAD dataset, run following codes:
```
cd path/to/MTN_trajectory/JAAD
python train_jaad.py
```
Saved model files (every 10 epoches) are in ./JAAD_model during training.

To evaluate, run:
```
python test_jaad.py
```
You can test your own model by setting `model_path` in `test_jaad.py` or `test_pie.py`. According to our experiments, the test result from each training model may have slight differences as distinct initializations and GPU settings. 
On PIE dataset, the MSE results are taken from 440 to 460. 
On JAAD dataset, the MSE results are taken from 995 to 1030.

## Citation
```
@InProceedings{MTN_trajectory,
author = {Ziyi Yin and Ruijin Liu and Zhiliang Xiong and Zejian Yuan},
title = {Multimodal Transformer Networks for Pedestrian Trajectory Prediction},
booktitle = {IJCAI},
year = {2021}
}
```
## License
MTN_trajectory is released under BSD 3-Clause License. Please see [LICENSE](LICENSE) file for more information.


## Acknowledgements

[PIE dataset](https://github.com/aras62/PIE)

[JAAD dataset](https://github.com/ykotseruba/JAAD)

[Trajectory-Transformer](https://github.com/FGiuliari/Trajectory-Transformer)

[RAFT](https://github.com/princeton-vl/RAFT)
