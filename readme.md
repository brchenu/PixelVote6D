# Notes on PVNet archi

### Skip connections

- skip_1 (1/2 res) after: resnet.relu (64 channels)
- skip_2 (1/4 res) after: resnet.layer1 (64 channels)
- skip_3 (1/8 res) after: resnet.layer2 (128 channels)
- skip_4 (1/16 res) after: resnet.layer3 (256 channels)
- skip_5 Bottleneck (1/32 res) after: resnet.layer4 (512 channels)