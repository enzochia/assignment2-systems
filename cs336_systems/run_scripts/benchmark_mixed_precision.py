import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        logging.info(f"dtype of fc1 output: {x.dtype}")
        x = self.relu(x)
        logging.info(f"dtype of relu output: {x.dtype}")
        x = self.ln(x)
        logging.info(f"dtype of layerNorm output: {x.dtype}")
        x = self.fc2(x)
        logging.info(f"dtype of logit output: {x.dtype}")
        return x


batch_size = 32
in_features = 64
dtype = torch.bfloat16
text_input = torch.randn(batch_size, in_features, device=device)
toy_model = ToyModel(in_features, in_features).to(device)

train_context = torch.amp.autocast(device_type=device, dtype=dtype)


logging.info("##################### Without AMP #####################")
for param_name, param in toy_model.named_parameters():
    logging.info(f"dtype of {param_name:<25} | {param.dtype}")
logits = toy_model(text_input)
loss = (logits**2).sum()
logging.info(f"dtype of loss: {loss.dtype}")
loss.backward()
for param_name, param in toy_model.named_parameters():
    logging.info(f"dtype of gradient of {param_name:<25} | {param.grad.dtype}")

logging.info("##################### With AMP #####################")
with train_context:
    for param_name, param in toy_model.named_parameters():
        logging.info(f"dtype of {param_name:<25} | {param.dtype}")
    logits = toy_model(text_input)
    loss = (logits**2).sum()
    logging.info(f"dtype of loss: {loss.dtype}")
    loss.backward()
    for param_name, param in toy_model.named_parameters():
        logging.info(f"dtype of gradient of {param_name:<25} | {param.grad.dtype}")
