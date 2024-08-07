
import pytest
import torch
from golden_resnet import resnet50_r
from bottleneck import Bottleneck_pre
from torchvision.models.resnet import Bottleneck
from comp_pcc import pcc

@pytest.fixture
def setup_models():
    model = resnet50_r("resnet50-0676ba61.pth")
    new_state_dict = pcc.modify_state_dict_with_prefix(model, 'layer4.1.')
    r1 = Bottleneck_pre(2048, 512)
    r2 = Bottleneck(2048, 512)
    
    r1.load_state_dict(new_state_dict)
    r2.load_state_dict(new_state_dict)
    
    return r1, r2

@pytest.fixture
def setup_inputs():
    return torch.rand(1, 2048, 7, 7)

def test_bottleneck_output_similarity(setup_models, setup_inputs):
    r1, r2 = setup_models
    input_ids = setup_inputs
    
    output1 = r1(input_ids)
    output2 = r2(input_ids)
    
    val = pcc.compare_pcc(output1, output2)
    
    print(f"Debug: compare_pcc return value: {val}")
    
    if isinstance(val, tuple):
        print(f"Debug: val is a tuple with length {len(val)}")
        for i, v in enumerate(val):
            print(f"Debug: Element {i}: {v}")
    
    pcc_value = val[0] if isinstance(val, tuple) else val
    
    assert pcc_value > 0.99, f"Outputs are not similar enough: PCC={pcc_value}"

if __name__ == "__main__":
    pytest.main()

# from golden_resnet import resnet50_r
# import torch
# from bottleneck import Bottleneck_pre
# from torchvision.models.resnet import Bottleneck
# from comp_pcc import pcc
    
# model = resnet50_r("resnet50-0676ba61.pth")
# new_state_dict = pcc.modify_state_dict_with_prefix(model, 'layer4.1.')
# r1 = Bottleneck_pre(2048,512)
# r2 = Bottleneck(2048,512)
# r1.load_state_dict(new_state_dict)
# r2.load_state_dict(new_state_dict)

# input_ids =torch.rand(1,2048,7,7)
# output1 = r1(input_ids)
# output2 = r2(input_ids)
# output1_flat = pcc.flatten_tuple(output1)
# output2_flat = pcc.flatten_tuple(output2)
# val=pcc.compare_pcc(output1_flat,output2_flat)
# print(val)