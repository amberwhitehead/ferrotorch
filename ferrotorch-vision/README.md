# ferrotorch-vision

Vision models, datasets, and transforms for ferrotorch — covering classification, detection, and segmentation.

## What it provides

### Model architectures

**Classification** (10 architectures with full forward parity):

| Architecture      | Variants              | Registry key                          |
|-------------------|-----------------------|---------------------------------------|
| ResNet            | 18, 34, 50            | `resnet18` / `resnet34` / `resnet50`  |
| VGG               | 11, 16                | `vgg11` / `vgg16`                     |
| EfficientNet      | B0                    | `efficientnet_b0`                     |
| MobileNetV2       | standard              | `mobilenet_v2`                        |
| MobileNetV3       | Small                 | `mobilenet_v3_small`                  |
| ConvNeXt          | Tiny                  | `convnext_tiny`                       |
| Swin Transformer  | Tiny                  | `swin_tiny`                           |
| ViT               | B/16                  | `vit_b_16`                            |
| DenseNet          | 121                   | `densenet121`                         |
| InceptionV3       | standard              | `inception_v3`                        |

**Object detection** (7 architectures):

| Architecture      | Notes                                     | Registry key                  |
|-------------------|-------------------------------------------|-------------------------------|
| Faster R-CNN      | ResNet-50 backbone with FPN               | `fasterrcnn_resnet50_fpn`     |
| Mask R-CNN        | Faster R-CNN + mask head                  | `maskrcnn_resnet50_fpn`       |
| Keypoint R-CNN    | Faster R-CNN + keypoint head              | `keypointrcnn_resnet50_fpn`   |
| RetinaNet         | ResNet-50 + FPN, focal loss               | `retinanet_resnet50_fpn`      |
| FCOS              | Anchor-free, ResNet-50 + FPN              | `fcos_resnet50_fpn`           |
| SSD300            | Single Shot Detector, VGG-16, 300x300     | `ssd300_vgg16`                |
| YOLO              | Single-stage real-time detector           | `yolo`                        |

**Segmentation** (4 architectures):

| Architecture      | Notes                                     | Registry key                  |
|-------------------|-------------------------------------------|-------------------------------|
| DeepLabV3         | ASPP-based, ResNet-50 backbone            | `deeplabv3_resnet50`          |
| FCN               | Fully Convolutional Network, ResNet-50    | `fcn_resnet50`                |
| LR-ASPP           | Lite R-ASPP, MobileNetV3-Large backbone   | `lraspp_mobilenet_v3_large`   |
| U-Net             | Encoder–decoder with skip connections     | `unet`                        |

### Datasets

- **`Mnist`**, **`Cifar10`**, **`Cifar100`** — automatic download/extraction with `Split` (train/test)
- **`ImageFolder`** — load arbitrary directory-based classification datasets

### Image I/O

`read_image`, `read_image_as_tensor`, `write_image`, `write_tensor_as_image`, `raw_image_to_tensor`

### Transforms

`CenterCrop`, `Resize`, `VisionNormalize`, `VisionToTensor` with `IMAGENET_MEAN`/`IMAGENET_STD`

### Model registry

`register_model`, `get_model`, `list_models`, `ModelRegistry`, `ModelConstructor`

### Feature extraction

`create_feature_extractor`, `FeatureExtractor` for intermediate layer outputs

## Quick start

```rust
use ferrotorch_vision::{Mnist, Split, VisionToTensor, VisionNormalize};
use ferrotorch_vision::models::get_model;

// Load a dataset
let dataset = Mnist::new("./data", Split::Train)?;
let sample = dataset.get(0)?;
let image_tensor = VisionToTensor.transform(&sample.image)?;
let normalized = VisionNormalize::new(vec![0.1307], vec![0.3081])
    .transform(&image_tensor)?;

// Instantiate a classification model
let model = get_model::<f32>("resnet50", 1000)?;

// Instantiate a detection model
let detector = get_model::<f32>("fasterrcnn_resnet50_fpn", 91)?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
