import torch
import torch.nn as nn
from torchvision import models
from collections import defaultdict, OrderedDict


class VGG(nn.Module):
    """Lightweight VGG used by some CIFAR checkpoints.

    This implementation registers convolution/bn/relu/pool layers as top-level
    attributes named conv0, bn0, relu0, pool0, ... so the state_dict keys
    match checkpoints that were saved with those names (e.g. 'conv0.weight').
    The forward pass applies layers in creation order and ends with a
    Linear(512, num_classes) classifier that expects the avg-pooled 512-d input.
    """

    ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self._layer_names = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            nm = f"{name}{counts[name]}"
            counts[name] += 1
            setattr(self, nm, layer)
            self._layer_names.append(nm)

        in_channels = 3
        for x in self.ARCH:
            if x != 'M':
                # conv-bn-relu
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # maxpool
                add("pool", nn.MaxPool2d(2))

        # classifier expects a 512-d pooled feature
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply the layers in order
        for nm in self._layer_names:
            layer = getattr(self, nm)
            x = layer(x)

        # avgpool spatial dims -> [N, 512]
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def get_vgg_for_cifar10(num_classes: int = 10, variant: str = "vgg16_bn") -> nn.Module:
    """Create a VGG model and adapt classifier to CIFAR-10.

    Supports several variants:
    - 'vgg16_bn' and 'vgg11_bn' produce torchvision VGGs adapted for CIFAR-10
      (default behaviour)
    - 'vgg2' returns a lightweight VGG matching the checkpoint naming used in
      some course checkpoints (top-level conv0/bn0/... and classifier Linear(512, C)).
    """
    if variant == "vgg16_bn":
        m = models.vgg16_bn(weights=None)
        # Replace last classifier layer for CIFAR-10
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m
    elif variant == "vgg11_bn":
        m = models.vgg11_bn(weights=None)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m
    elif variant in ("vgg2", "vgg_cifar", "vgg_small"):
        return VGG(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def iter_prunable_params(model: nn.Module):
    """Yield (name, weight_tensor) for Conv2d/Linear weights only."""
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            yield f"{name}.weight", mod.weight


def get_model(variant: str = "vgg16_bn") -> nn.Module:
    return get_vgg_for_cifar10(variant=variant)
