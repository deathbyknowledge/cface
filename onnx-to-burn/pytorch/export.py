import torch
from download import download_url_to_file
from models import InceptionResnetV1Shard1, InceptionResnetV1Shard2


if __name__ == "__main__":
    device = torch.device("cpu")
    torch.set_grad_enabled(False)
    dummy_input = torch.randn(1, 3, 160, 160, device=device)

    shard1 = InceptionResnetV1Shard1().eval().to(device)
    shard2 = InceptionResnetV1Shard2().eval().to(device)
    shard2_input = shard1(dummy_input)
    torch.onnx.export(shard1, dummy_input, "shard1.onnx", verbose=True, opset_version=16)
    torch.onnx.export(shard2, shard2_input, "shard2.onnx", verbose=True, opset_version=16)
