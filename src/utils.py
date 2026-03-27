import torch

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")