from  data.dataset import load_multijail_dataset, explore_multijail_dataset

# Load the entire dataset
dataset = load_multijail_dataset()
print(f"Dataset size: {len(dataset['train'])}")

# Quick exploration
explore_multijail_dataset()