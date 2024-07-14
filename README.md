<div align="center">

<h1>GroupDiff: Diffusion-based Group Portrait Editing</h1>

</div>


## Dataset

Our dataset is developed based on NUS LV Multiple-Human Parsing Dataset v2.0. Please download the source data from the [link](https://drive.google.com/file/d/1YVBGMru0dlwB8zu1OoErOazZoc8ISSJn/view?usp=sharing).

We use [MMPose](https://github.com/open-mmlab/mmpose) to estimate the pose using the "vitpose_h" model.

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

Based on these data processing, we propose a comprehensive training data generation engine to synthesize paired data. The data generation codes can be found [here]().
