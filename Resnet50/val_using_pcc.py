import torch
import numpy as np
import logging
logger = logging.getLogger(__name__)
# Function to read tensors from text files
def read_tensor_from_txt(file_path):
    with open(file_path, 'r') as f:
        tensor_data = f.read()
        # Remove 'tensor([' and '])' wrappers, brackets, and commas
        tensor_data = tensor_data.replace('tensor([', '').replace('])', '')
        tensor_data = tensor_data.replace('[', '').replace(']', '').replace(',', '')
        tensor_data = tensor_data.strip().split()
        tensor_data = [float(x) for x in tensor_data]
    return torch.tensor(tensor_data)

def comp_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, f"PCC: {1.0}"

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, f"PCC: {0.0}"

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.error("One tensor is all zero")
        return False, f"PCC: {0.0}"

    # For now, mask all infs and nans so that we check the rest... TODO
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, f"PCC: {1.0}"

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, f"PCC: {1.0}"

    return cal_pcc >= pcc, f"PCC: {cal_pcc}"

# Paths to the text files
file_path1 = 'tv/resnet_50/logits_tv_api.txt'
file_path2 = 'tv/resnet_50/logits_tv_local.txt'
print(file_path1)
# Read tensors from text files
tensor1 = read_tensor_from_txt(file_path1)
tensor2 = read_tensor_from_txt(file_path2)
val=comp_pcc(tensor2,tensor1)
print(val)