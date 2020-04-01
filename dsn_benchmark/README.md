# Benchmark Dataset
We provide images that violate safety properties defined in ["Quantifying DNN Model Robustness to the Real-World Threats".](https://github.com/advboxes/perceptron-benchmark/blob/master/dsn_benchmark/dsn_2020_per.pdf) These images are generated against various learning tasks including image classifiers, object detectors, and cloud-based content moderators.

# Dataset Structure
The safety violation images are compressed in a tarball with respect to corresponding Safety Properrty. Once extracted, you can find the following directory structure.
```
{SafetyProperty}/image_file_name/
                                |-- image_file (the original .png file from ImageNet, MSCOCO, or NSFW dataset)
                                |-- {model1_under_test}.tif
                                |-- {model2_under_test}.tif
                                |-- {model3_under_test}.tif
                                |-- ...
```

The {SafetyProperty} is one of the properties proposed, e.g., RotationMetric, BrightnessMetric, etc., 


# Download
## ImageNet Pre-trained Models
We randomly pick 1k images from ImageNet dataset and run against 15 different pre-trained models. The safety-violation dataset can be found in the following table. 

| Category | Safety Property | Download URLs  |
| ------ | ------ | ----- |
| Luminance | Brightness | [tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_BrightnessMetric.tar.gz), md5: f774b31d68142a0d447cb4507bab9401 |
|  | Contrast Reduction |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_ContrastReductionMetric.tar.gz) md5: fbd0a05d129ff5fecfdc65e6b0561b72 |
|  Transformation | Rotation |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_RotationMetric.tar.gz) md5: 397d6eb95df6db0fea5ec9c4847fc047|
|  | Horizontal Translation|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_HorizontalTranslationMetric.tar.gz) md5: ceb3c073d9acaa34b38079a5a46ee181|
|  | Vertical Translation|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_VerticalTranslationMetric.tar.gz)  md5: 9fcd0532e4a3d72f5eef1b6dba4179a2|
|  | Spatial Translation|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_SpatialMetric.tar.gz) md5: a040c204723a8928c597aa84e1f12104 |
| Blur | Motion Blur|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_MotionBlurMetric.tar.gz) md5: b8969ee95620340c35f32a33f8c02d5b |
|  | Gaussian Blur|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_GaussianBlurMetric.tar.gz)  md5: 97df318af3848f7c768a54e8f114ddd3|
| Corruption | Uniform Noise|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_AdditiveUniformNoiseMetric.tar.gz) md5: a8bc6b390094ce1f4e36f5882e2c5df2 |
|  |  Gaussian Noise |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_AdditiveGaussianNoiseMetric.tar.gz) md5: dd7c50c1d94b6ea7674f749c6ba0ca26 |
|  |  Blended Uniform Noise |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_BlendedUniformNoiseMetric.tar.gz) md5: 266ffaa5adf387eedf40caffc8e6f9b7 |
|  |  Salt & Pepper Noise |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_SaltAndPepperNoiseMetric.tar.gz) md5: 8c46c04a40e2fff4a6750a63eff3e144 |
| Weather |  Fog |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_FogMetric.tar.gz) md5: 22c0ea13219835b0334bc2c285862388|
|  |  Snow |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_SnowMetric.tar.gz) md5: a2fdfa22df6741777312c435200dc166 |
| | Frost|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/imagenet/imagenet_FrostMetric.tar.gz) md5: 7f5c3da9b1d272d98af71c4b869aa1de |



## Object Detection Models
The safety violation images obtained by running test against 3 different state-of-the-art object detection models including YOLOv3, SSD300, and Retina-Resnet50. To make comparison with the image classifier, we also generated safety violation images against Resnet152. The image datasets can be found in the following table.

| Category | Safety Property | Download URLs  |
| ------ | ------ | ----- |
| Luminance | Brightness | [tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_BrightnessMetric.tar.gz), md5: 7b33d5bf50a1797b0551b537f2387af7 |
|  | Contrast Reduction |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_ContrastReductionMetric.tar.gz), md5:  4eaddaa596c0e545d08b425332fc9698|
|  Transformation | Rotation |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_RotationMetric.tar.gz) md5: 263e3e23d8e1e10609efafc76885d643 |
|  | Horizontal Translation|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_HorizontalTranslationMetric.tar.gz) md5: edcc7881e53c870b5663cfa7ec4f5e46|
|  | Vertical Translation|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_VerticalTranslationMetric.tar.gz)  md5: 669a7a1db0375cc1e0be56fdfac804be|
|  | Spatial Translation|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_SpatialMetric.tar.gz) md5: 8d10786171790f007c0e2812e4fd5040 |
| Blur | Motion Blur|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_MotionBlurMetric.tar.gz) md5: 10f0efc9b88bac8e32059d15ba1bc223 |
|  | Gaussian Blur|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_GaussianBlurMetric.tar.gz)  md5: 80aec2a7659de10ba425647382fa1933 |
| Corruption | Uniform Noise|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_AdditiveUniformNoiseMetric.tar.gz) md5: d22797ac55f2cc123844d6f6b4fcb185 |
|  |  Gaussian Noise |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_AdditiveGaussianNoiseMetric.tar.gz) md5: b3dcf3a1dfc03d7105229eae1872daa7 |
|  |  Blended Uniform Noise |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_BlendedUniformNoiseMetric.tar.gz) md5: c424a1e490f8f8858eea51f65b3d69fa |
|  |  Salt & Pepper Noise |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_SaltAndPepperNoiseMetric.tar.gz) md5: d0be4dcdef1415f92f93b8e1792af03b |
| Weather |  Fog |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_FogMetric.tar.gz) md5: fcf3a765a3123fb690b133f9b464a7c1 |
|  |  Snow |[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_SnowMetric.tar.gz) md5: dfa984218f70348858b72e7d1621f23b |
| | Frost|[tarball](https://perceptron-benchmark.s3-us-west-1.amazonaws.com/dsn2020_benchmark_data/object/object_FrostMetric.tar.gz) md5: 665fd88395fa23d7d30a84182195564f |


# Content Moderator
We compared model robustness among three commercial cloud-based classifiers with their name anonymized. Due to the nature of the NSFW data, we don't publish the dataset here. However, interested users can make a request to us for this dataset. 
