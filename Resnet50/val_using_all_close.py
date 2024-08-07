import torch

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

# Paths to the text files
file_path1 = 'tv/resnet_50/logits_tv_api.txt'
file_path2 = 'resnet_50/logits_tv_local.txt'

# Read tensors from text files
tensor1 = read_tensor_from_txt(file_path1)
tensor2 = read_tensor_from_txt(file_path2)

# Print tensors
# print("Tensor 1:", tensor1)
# print("Tensor 2:", tensor2)

# Compare tensors using allclose
are_close = torch.allclose(tensor1, tensor2, atol=1e-5)
print(f"Tensors are close: {are_close}")
