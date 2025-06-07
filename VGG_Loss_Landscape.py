import matplotlib as mpl
# Use Agg backend for non-interactive plotting (important for servers/no GUI)
# Keep this if running in an environment without a display
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
from torch import nn
from torch.optim import Adam,SGD # Explicitly import Adam
from tqdm import tqdm as tqdm
from IPython import display # Used for clearing output, mainly effective in notebooks

# --- Module Import ---
try:
    # Adjust paths as necessary for your project structure
    import sys
    # Try adding relative paths first
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    sys.path.append(os.path.join(script_dir)) # Current dir
    sys.path.append(os.path.join(script_dir, '..')) # Parent dir
    sys.path.append(os.path.join(script_dir, '..', '..')) # Grandparent dir

    # Ensure models and data are found
    if not any('models' in p for p in sys.path):
         print("Warning: 'models' directory might not be in Python path.")
    if not any('data' in p for p in sys.path):
         print("Warning: 'data' directory might not be in Python path.")

    from models.vgg import VGG_A
    from models.vgg import VGG_A_BN # Make sure this class is defined correctly in models/vgg.py
    from data.loaders import get_cifar_loader
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(sys.path)
    print("Please ensure 'models.vgg.VGG_A', 'models.vgg.VGG_A_BatchNorm', and 'data.loaders.get_cifar_loader' are accessible.")
    print("Make sure the 'models' and 'data' directories are in your Python path or relative to the script.")
    exit()
except NameError: # Handle case where __file__ is not defined (e.g. interactive)
     from models.vgg import VGG_A
     from models.vgg import VGG_A_BN
     from data.loaders import get_cifar_loader


# --- Constants and Parameters ---
# device_ids = [0] # Available GPU IDs (using target_gpu_id instead)
num_workers = 4
batch_size = 128
seed_value = 2020
epochs = 20 # Set desired number of epochs
# LEARNING RATES TO TEST (as per prompt example)
learning_rates_to_test = [1e-3, 1e-4, 5e-4]
print(f"Learning rates to test: {learning_rates_to_test}")

target_gpu_id = 0 # Choose your target GPU

# --- Paths Setup ---
# Determine project home path reliably
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd() # Fallback if __file__ is not defined

# Assume project root is one level up from script directory
project_home_path = os.path.dirname(script_dir)

# Define specific output paths within the project structure
reports_path = os.path.join(project_home_path, 'reports')
figures_path = os.path.join(reports_path, 'figures')
models_path = os.path.join(reports_path, 'models')
training_data_path = os.path.join(reports_path, 'training_data') # For losses

# Ensure directories exist
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)
os.makedirs(training_data_path, exist_ok=True)
print(f"Figures will be saved in: {figures_path}")
print(f"Model checkpoints will be saved in: {models_path}")
print(f"Training data (losses) will be saved in: {training_data_path}")

# --- Device Setup ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Ensure CUDA device order is consistent
device = torch.device(f"cuda:{target_gpu_id}" if torch.cuda.is_available() and target_gpu_id < torch.cuda.device_count() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(target_gpu_id)}")

# --- Data Loaders ---
print("Initializing data loaders...")
try:
    # Adjust root path if necessary, './data/' is relative to execution directory or project root
    data_root = os.path.join(project_home_path, 'data')
    if not os.path.exists(data_root):
        data_root = './data' # Fallback if project structure differs
        print(f"Data root '{os.path.join(project_home_path, 'data')}' not found, using '{data_root}'")

    train_loader = get_cifar_loader(root=data_root, train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = get_cifar_loader(root=data_root, train=False, batch_size=batch_size, num_workers=num_workers)
    print("Data loaders initialized.")

    # Quick check of one batch
    print("Checking one batch from train_loader...")
    for X, y in train_loader:
        print(f"  Input X shape: {X.shape}")
        print(f"  Target y shape: {y.shape}")
        break # Only check the first batch
except Exception as e:
    print(f"Error initializing or checking data loaders: {e}")
    print(f"Attempted data root: {data_root}")
    exit()

# --- Utility Functions ---

def get_accuracy(model, loader, device):
    """Calculates model accuracy on the given data loader."""
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculations
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    # Note: Not setting back to train mode here, caller should handle it.
    # model.train()
    return accuracy

def set_random_seeds(seed=0, device_type='cpu'):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU.
        # Comment out deterministic/benchmark settings if they cause issues or slow down significantly
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed}")


# --- Training Function (Modified to only return step losses) ---

def train_and_get_losses(model, optimizer, criterion, train_loader, device, epochs_n=100, model_name="model", current_lr=0.0):
    """Trains the model and returns a list of losses for every step."""
    model.to(device)
    all_step_losses = []
    total_steps = 0

    print(f"\n--- Training {model_name} with LR={current_lr} for {epochs_n} epochs on {device} ---")
    model.train() # Ensure model is in training mode initially

    for epoch in range(epochs_n):
        epoch_loss = 0.0
        # Use tqdm for progress bar per epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_n}", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at Epoch {epoch+1}, Step {i}. Stopping run for {model_name} with LR {current_lr}.")
                # Return partial losses, or handle as needed (e.g., return None or raise error)
                return all_step_losses # Return what we have so far

            all_step_losses.append(loss.item()) # Record step loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            total_steps += 1
            # Update progress bar description
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs_n} - Average Loss: {avg_epoch_loss:.4f}")

    print(f"--- Finished training {model_name} with LR={current_lr}. Total steps: {total_steps} ---")
    return all_step_losses


# --- Combined Plotting Function ---

def plot_loss_landscape_comparison(results_dict, figure_path):
    """
    Plots the min/max loss curves for different model types and fills the area.

    Args:
        results_dict (dict): A dictionary where keys are model names (e.g., 'VGG_A', 'VGG_A_BN')
                             and values are lists of lists, where each inner list contains
                             step losses from one training run (e.g., with a different LR).
        figure_path (str): Path to save the plot image.
    """
    print(f"\nGenerating loss landscape comparison plot to {figure_path}...")
    plt.style.use('seaborn-v0_8-darkgrid') # Use a style similar to the example
    plt.figure(figsize=(12, 7)) # Adjusted size slightly

    colors = {'VGG_A': 'mediumseagreen', 'VGG_A_BN': 'indianred'} # Colors similar to example
    labels = {'VGG_A': 'Standard VGG', 'VGG_A_BN': 'Standard VGG + BatchNorm'}

    min_steps = float('inf') # Find the minimum number of steps across all runs

    processed_data = {}

    for model_name, list_of_loss_runs in results_dict.items():
        if not list_of_loss_runs:
            print(f"Warning: No loss data found for model '{model_name}'. Skipping.")
            continue

        # Find min length for this model type
        current_min_len = min(len(run) for run in list_of_loss_runs if run) # Ensure run is not empty
        if current_min_len == 0:
             print(f"Warning: Found run with 0 length for model '{model_name}'. Skipping this model.")
             continue
        min_steps = min(min_steps, current_min_len)

        # Store truncated runs for processing
        processed_data[model_name] = [run[:current_min_len] for run in list_of_loss_runs if run]


    if min_steps == float('inf') or min_steps == 0:
        print("Error: Could not determine a valid number of steps from the training runs. No plot generated.")
        return

    print(f"Plotting based on the minimum common number of steps: {min_steps}")
    steps = np.arange(min_steps)

    for model_name, truncated_runs in processed_data.items():
        # Stack losses at each step across different runs (LRs)
        # Ensure all runs being stacked have the length 'min_steps'
        runs_for_stacking = [run[:min_steps] for run in truncated_runs]
        if not runs_for_stacking:
            print(f"Warning: No valid runs left for {model_name} after truncation. Skipping.")
            continue

        try:
            loss_array = np.array(runs_for_stacking) # Shape: (num_runs, min_steps)
        except ValueError as e:
            print(f"Error creating numpy array for {model_name}. Check consistency of run lengths. Details: {e}")
            # Print lengths for debugging
            for i, run in enumerate(runs_for_stacking):
                print(f" Run {i} length: {len(run)}")
            continue # Skip this model if array creation fails


        # Calculate min and max loss at each step
        min_curve = np.min(loss_array, axis=0)
        max_curve = np.max(loss_array, axis=0)

        # Get color and label
        color = colors.get(model_name, 'gray') # Default to gray if name not in dict
        label = labels.get(model_name, model_name) # Default to key name

        # Plot the min/max lines (optional, make them subtle)
        # plt.plot(steps, min_curve, color=color, alpha=0.6, linewidth=0.5)
        # plt.plot(steps, max_curve, color=color, alpha=0.6, linewidth=0.5)

        # Fill the area between min and max
        plt.fill_between(steps, min_curve, max_curve, color=color, alpha=0.4, label=label) # Slightly adjust alpha

    plt.title('Loss Landscape Comparison', fontsize=16)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12) # Changed Y-axis label
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(bottom=0) # Loss should be non-negative
    # Optional: set xlim if needed, e.g., plt.xlim(0, min_steps)
    plt.tight_layout()

    try:
        plt.savefig(figure_path)
        print(f"Loss landscape comparison plot saved to {figure_path}")
    except Exception as e:
        print(f"Error saving loss landscape plot: {e}")
    plt.close() # Close the figure to free memory


# --- Main Execution ---

if __name__ == '__main__':

    all_results = {
        'VGG_A': [],
        'VGG_A_BN': []
    }

    model_classes = {
        'VGG_A': VGG_A,
        'VGG_A_BN': VGG_A_BN
    }

    criterion = nn.CrossEntropyLoss()

    for model_name, ModelClass in model_classes.items():
        print(f"\n===== Processing Model Type: {model_name} =====")
        model_step_losses_across_lrs = []

        for lr in learning_rates_to_test:
            print(f"\n--- Starting run for {model_name} with LR = {lr} ---")

            # Set seed before each run for consistency IF desired,
            # otherwise set once outside the loop for more variability
            set_random_seeds(seed=seed_value + hash(f"{model_name}_{lr}") % 1000, device_type=device.type) # Vary seed slightly per run

            # Instantiate the model
            model = ModelClass(num_classes=10) # CIFAR-10 has 10 classes

            # Instantiate the optimizer
            optimizer = Adam(model.parameters(), lr=lr) # Use Adam

            # Train and get step losses
            step_losses = train_and_get_losses(
                model, optimizer, criterion, train_loader, device,
                epochs_n=epochs, model_name=model_name, current_lr=lr
            )

            if step_losses: # Only add if training didn't fail immediately
                model_step_losses_across_lrs.append(step_losses)

                # Save individual run losses (optional)
                loss_save_file = os.path.join(training_data_path, f'step_losses_{model_name}_lr{lr:.0e}.txt')
                try:
                    np.savetxt(loss_save_file, np.array(step_losses), fmt='%.6f')
                    print(f"Saved step losses for {model_name} (LR={lr}) to {loss_save_file}")
                except Exception as e:
                    print(f"Error saving step losses for {model_name} (LR={lr}): {e}")
            else:
                 print(f"Run for {model_name} with LR={lr} produced no loss data (possibly due to NaN).")


            # Clear memory (optional, might help on resource-constrained systems)
            del model
            del optimizer
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Store all loss lists for this model type
        all_results[model_name] = model_step_losses_across_lrs

    # --- Generate the Comparison Plot ---
    comparison_plot_file = os.path.join(figures_path, 'loss_landscape_comparison_vgg_vs_vggBN.png')
    plot_loss_landscape_comparison(all_results, comparison_plot_file)

    print("\nScript finished.")
