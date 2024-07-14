<div align="center">

<h1>GroupDiff: Diffusion-based Group Portrait Editing</h1>

</div>


## Dataset

Our dataset is developed based on NUS LV Multiple-Human Parsing Dataset v2.0. Please download the source data from the [link](https://drive.google.com/file/d/1YVBGMru0dlwB8zu1OoErOazZoc8ISSJn/view?usp=sharing).

We use [MMPose](https://github.com/open-mmlab/mmpose) to estimate the pose using the "vitpose_h" model. You can download the pose estimation results from this [link](https://drive.google.com/file/d/1_ivJ5jTv0p-gdcZ8XLvTix_ymg7KOJTL/view?usp=sharing).

After downloading the dataset, unzip the file and put them under the dataset folder with the following structure:
```
./LV-MHP-v2
├── train
    ├── images
    ├── parsing_annos
    └── pose_estimation
└── shhq_dataset
    ├── images
    ├── parsing_annos
    └── pose_estimation
```

Based on the preprocessed data, we propose a comprehensive training data generation engine to synthesize paired data. The data generation codes can be found [here](https://github.com/yumingj/GroupDiff/blob/main/data/group_diff_data_gen.py).
