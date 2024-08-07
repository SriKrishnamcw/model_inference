import pytest
import torch
from golden_resnet import resnet50_r
from torchvision.models.resnet import resnet50
from torchvision.models import ResNet50_Weights
from comp_pcc import pcc

# Dictionary defining layer configurations
layer_configs = {
    "layer1.0": (1, 64, 56, 56),
    "layer1.1": (1, 256, 56, 56),
    "layer1.2": (1, 256, 56, 56),
    "layer2.0": (1, 256, 56, 56),
    "layer2.1": (1, 512, 28, 28),
    "layer2.2": (1, 512, 28, 28),
    "layer2.3": (1, 512, 28, 28),
    "layer3.0": (1, 512, 28, 28),
    "layer3.1": (1, 1024, 14, 14),
    "layer3.2": (1, 1024, 14, 14),
    "layer3.3": (1, 1024, 14, 14),
    "layer3.4": (1, 1024, 14, 14),
    "layer3.5": (1, 1024, 14, 14),
    "layer4.0": (1, 1024, 14, 14),
    "layer4.1": (1, 2048, 7, 7),
    "layer4.2": (1, 2048, 7, 7),
}

@pytest.fixture(scope='module')
def load_models():
    gold = resnet50(weights="IMAGENET1K_V1")
    ref = resnet50_r("resnet50-0676ba61.pth")
    return gold, ref

@pytest.fixture(params=layer_configs.items())
def load_weight(request, load_models):
    layer_name, (b, c, h, w) = request.param
    gold, ref = load_models
    gold_layer = getattr(gold, layer_name.split('.')[0])[int(layer_name.split('.')[1])]
    ref_layer = getattr(ref, layer_name.split('.')[0])[int(layer_name.split('.')[1])]
    return gold_layer, ref_layer,b, c, h, w

def test_full_bottlenecks(load_weight):
    gold, ref,b, c, h, w = load_weight

    input_tensor = torch.randn(b, c, h, w)

    gold_output = gold(input_tensor)
    ref_output = ref(input_tensor)

    pcc_value = pcc.compare_pcc(gold_output, ref_output)

    result = pcc_value[0] if isinstance(pcc_value, tuple) else pcc_value
    

if __name__ == "__main__":
    pytest.main()

