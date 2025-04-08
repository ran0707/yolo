import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Backbone: Modified ResNet50 to output multi-scale features
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Extract intermediate layers
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # C1
        self.layer2 = resnet.layer1  # C2
        self.layer3 = resnet.layer2  # C3
        self.layer4 = resnet.layer3  # C4
        self.layer5 = resnet.layer4  # C5

    def forward(self, x):
        c3 = self.layer3(self.layer2(self.layer1(x)))  # 512 channels, stride 8
        c4 = self.layer4(c3)                           # 1024 channels, stride 16
        c5 = self.layer5(c4)                           # 2048 channels, stride 32
        return c3, c4, c5

# Feature Pyramid Network (FPN): Combines multi-scale features
class FPN(nn.Module):
    def __init__(self, in_channels_list=[512, 1024, 2048], out_channels=256):
        super(FPN, self).__init__()
        # Lateral connections to reduce channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list
        ])
        # 3x3 convs to refine features
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        c3, c4, c5 = features
        # Top-down pathway
        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_convs[0](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        # Refine features
        p5 = self.fpn_convs[2](p5)
        p4 = self.fpn_convs[1](p4)
        p3 = self.fpn_convs[0](p3)
        return p3, p4, p5  # Strides 8, 16, 32

# Detection Head: Predicts boxes, classes, and segmentation coefficients
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_prototypes):
        super(DetectionHead, self).__init__()
        self.num_anchors = num_anchors
        # Separate heads for different predictions
        self.box_regression = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        self.objectness = nn.Conv2d(in_channels, num_anchors * 1, 3, padding=1)
        self.classification = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.segmentation_coeffs = nn.Conv2d(in_channels, num_anchors * num_prototypes, 3, padding=1)

    def forward(self, x):
        box = self.box_regression(x)          # (B, num_anchors*4, H, W)
        obj = self.objectness(x)              # (B, num_anchors*1, H, W)
        cls = self.classification(x)          # (B, num_anchors*num_classes, H, W)
        coeffs = self.segmentation_coeffs(x)  # (B, num_anchors*num_prototypes, H, W)
        return box, obj, cls, coeffs

# Protonet: Generates mask prototypes for segmentation
class Protonet(nn.Module):
    def __init__(self, in_channels, num_prototypes=32):
        super(Protonet, self).__init__()
        # Simple conv layers to generate prototypes
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, num_prototypes, 1)
        # Upsample to full image size (assuming input stride 8)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, p3):
        x = F.relu(self.conv1(p3))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        prototypes = self.upsample(x)  # (B, num_prototypes, H_full, W_full)
        return prototypes

# Custom YOLO Model
class CustomYOLO(nn.Module):
    def __init__(self, num_classes, num_prototypes=32, num_anchors=3):
        super(CustomYOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_prototypes = num_prototypes
        # Components
        self.backbone = Backbone()
        self.fpn = FPN()
        self.detection_heads = nn.ModuleList([
            DetectionHead(256, num_anchors, num_classes, num_prototypes) for _ in range(3)
        ])
        self.protonet = Protonet(256, num_prototypes)

    def forward(self, x):
        """
        Input:
            x: Tensor of shape (B, 3, H, W) - RGB image
        Output:
            outputs: List of tuples (box, obj, cls, coeffs) for each scale
            prototypes: Tensor of shape (B, num_prototypes, H, W)
        """
        # Extract features
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn((c3, c4, c5))
        # Generate prototypes from P3
        prototypes = self.protonet(p3)
        # Detection predictions at each scale
        outputs = []
        for i, feature in enumerate([p3, p4, p5]):
            box, obj, cls, coeffs = self.detection_heads[i](feature)
            outputs.append((box, obj, cls, coeffs))
        return outputs, prototypes

# Example usage
def main():
    # Hyperparameters
    num_classes = 5  # e.g., fungal infection, insect damage, etc.
    num_prototypes = 32
    num_anchors = 3
    image_size = 640

    # Initialize model
    model = CustomYOLO(num_classes=num_classes, num_prototypes=num_prototypes, num_anchors=num_anchors)
    model.eval()

    # Dummy input
    x = torch.randn(1, 3, image_size, image_size)
    outputs, prototypes = model(x)

    # Print output shapes
    for i, (box, obj, cls, coeffs) in enumerate(outputs):
        print(f"Scale {i}:")
        print(f"  Box: {box.shape}")
        print(f"  Obj: {obj.shape}")
        print(f"  Cls: {cls.shape}")
        print(f"  Coeffs: {coeffs.shape}")
    print(f"Prototypes: {prototypes.shape}")

if __name__ == "__main__":
    main()
