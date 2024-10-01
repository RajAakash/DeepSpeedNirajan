# Assume you have already loaded your model and dataset
from xturing.datasets import InstructionDataset  # or your specific dataset class
from xturing.models import BaseModel
import deepspeed
import os
import torch
# Load your dataset
test_dataset = InstructionDataset("./train_data_weights/test_test/")

# Load your model
#model_name = 'llama_lora_int8'  # or your model name
model = BaseModel.load("./model_weight_finals/model_weight1/")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move model to GPU
model.engine.model.to(device)

# Deepspeed use
#needs more parameters in torch.optim
optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
config_path = os.path.join(os.path.dirname(__file__), "deepSpeedConfig.json")
model_engine, optimizer, _, _ = deepspeed.initialize(
            args=None,  # Assuming no CLI args needed for DeepSpeed config
            model=model,
            optimizer=optimizer,
            config=config_path  
        )

# Evaluate the model on the test dataset
evaluation_results = model.evaluate(test_dataset)

# Print the results
print(f"Evaluation Results: {evaluation_results}")