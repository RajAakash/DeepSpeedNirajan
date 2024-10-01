from xturing.datasets import InstructionDataset  # or your specific dataset class
from xturing.models import BaseModel
import deepspeed
import os
import torch

# Load your dataset
test_dataset = InstructionDataset("./train_data_weights/test_test/")

# Load your model
model = BaseModel.load("./model_weight_finals/model_weight1/")

# Prepare DeepSpeed configuration
config_path = os.path.join(os.path.dirname(__file__), "deepSpeedConfig.json")

# Initialize DeepSpeed (handles optimizer and model placement)
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=None,  # Assuming no CLI args needed for DeepSpeed config
    model=model,
    model_parameters=model.parameters(),
    config=config_path
)

# Optionally put the model in evaluation mode if it's not handled automatically
model_engine.eval()

# Evaluate the model on the test dataset
evaluation_results = model.evaluate(test_dataset)

# Print the results
print(f"Evaluation Results: {evaluation_results}")
