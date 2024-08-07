import pytest
import torch
from golden_resnet import resnet50_r
from torchvision.models.resnet import resnet50
from comp_pcc import pcc

@pytest.fixture
def setup_models():
    r1 = resnet50(weights="IMAGENET1K_V1")
    r2 = resnet50_r("resnet50-0676ba61.pth")
    return r1, r2

@pytest.fixture
def setup_inputs():
    return torch.rand(1, 3, 224, 224)

def test_resnet_output_similarity(setup_models, setup_inputs):
    r1, r2 = setup_models
    input_ids = setup_inputs
    
    output1 = r1(input_ids)
    output2 = r2(input_ids)
    
    val = pcc.compare_pcc(output1, output2)
    
    pcc_value = val[0] if isinstance(val, tuple) else val
    
    assert pcc_value > 0.99, f"Outputs are not similar enough: PCC={pcc_value}"

if __name__ == "__main__":
    pytest.main()

# from golden_resnet import resnet50_r
# import torch
# from torchvision.models.resnet import resnet50
# from comp_pcc import pcc


# r1=resnet50(weights="IMAGENET1K_V1")    
# print(r1)
# # print(r1)
# r2= resnet50_r("resnet50-0676ba61.pth")
# # print(r2)

# input_ids =torch.rand(1,3,224,224)
# output1 = r1(input_ids)
# output2 = r2(input_ids)
# val=pcc.compare_pcc(output1,output2)
# print(val)
