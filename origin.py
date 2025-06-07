import matplotlib as mpl
mpl.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display # Used for clearing output, mainly effective in notebooks
try:
    import sys
    sys.path.append('./')
    sys.path.append('../')
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add parent directory to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) 
    from models.vgg import VGG_A
    #from models.vgg import VGG_A_BatchNorm
    from data.loaders import get_cifar_loader
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure 'models' and 'data' packages are correctly placed or added to PYTHONPATH.")
    exit()

# --- Constants and Parameters ---
device_ids = [0] # Available GPU IDs
num_workers = 4
batch_size = 128
seed_value = 2020
epochs = 20
learning_rate = 0.001
target_gpu_id = 0

# --- Paths Setup ---
# Determine project home path reliably
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd() # Fallback if __file__ is not defined (e.g., in interactive session)
home_path = os.path.dirname(script_dir) # Assumes script is in a subdirectory of the project root

figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
training_data_path = os.path.join(home_path, 'reports', 'training_data') # For losses and grads

# Ensure directories exist
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)
os.makedirs(training_data_path, exist_ok=True)

# --- Device Setup ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Ensure CUDA device order is consistent
device = torch.device(f"cuda:{target_gpu_id}" if torch.cuda.is_available() and target_gpu_id < torch.cuda.device_count() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(target_gpu_id)}")

# --- Data Loaders ---
print("Initializing data loaders...")
try:
    # Adjust root path if necessary, './data/' is relative to execution directory
    train_loader = get_cifar_loader(root='./data', train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = get_cifar_loader(root='./data', train=False, batch_size=batch_size, num_workers=num_workers)
    print("Data loaders initialized.")

    # Quick check of one batch
    print("Checking one batch from train_loader...")
    for X, y in train_loader:
        print(f"  Input X shape: {X.shape}")
        print(f"  Target y shape: {y.shape}")
        break # Only check the first batch
except Exception as e:
    print(f"Error initializing or checking data loaders: {e}")
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
    model.train() # Set model back to train mode if used during training loop pause
    return accuracy

def set_random_seeds(seed=0, device_type='cpu'):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed}")


# --- Training Function ---

def train(model, optimizer, criterion, train_loader, val_loader, device, scheduler=None, epochs_n=100, best_model_path=None):
    """Trains the model, records metrics, and saves the best version."""
    model.to(device)
    epoch_avg_losses = [np.nan] * epochs_n
    train_accuracies = [np.nan] * epochs_n
    val_accuracies = [np.nan] * epochs_n
    max_val_accuracy = 0.0
    best_epoch = -1

    all_step_losses = []
    all_step_grad_norms = []
    batches_n = len(train_loader)

    print(f"Starting training for {epochs_n} epochs on {device}...")
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        model.train() # Ensure model is in training mode
        current_epoch_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            all_step_losses.append(loss.item()) # Record step loss
            current_epoch_loss += loss.item()

            loss.backward()

            # Record gradient norm for the specified layer (e.g., model.classifier[4])
            try:
                # Ensure the layer exists and has a gradient
                grad = model.classifier[4].weight.grad
                if grad is not None:
                    grad_norm = grad.norm().item()
                    all_step_grad_norms.append(grad_norm)
                else:
                    all_step_grad_norms.append(0.0) # Append 0 if no grad
            except (IndexError, AttributeError):
                 all_step_grad_norms.append(np.nan) # Append NaN if layer missing/doesn't have .weight

            optimizer.step()

        # End of Epoch
        epoch_avg_losses[epoch] = current_epoch_loss / batches_n

        # Validation
        train_acc = get_accuracy(model, train_loader, device) # Accuracy on training set
        val_acc = get_accuracy(model, val_loader, device)     # Accuracy on validation set
        train_accuracies[epoch] = train_acc
        val_accuracies[epoch] = val_acc

        if scheduler:
            scheduler.step() # Step the scheduler (if applicable)

        # Check for best model
        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            best_epoch = epoch
            if best_model_path:
                try:
                    torch.save(model.state_dict(), best_model_path)
                except Exception as e:
                    print(f"\nError saving model: {e}")

        # Plotting progress
        display.clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        # Loss plot
        axes[0].plot(epoch_avg_losses[:epoch+1], marker='o')
        axes[0].set_title(f'Epoch {epoch+1}/{epochs_n} - Avg Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        # Accuracy plot
        axes[1].plot(train_accuracies[:epoch+1], marker='o', label=f'Train Acc ({train_acc:.2f}%)')
        axes[1].plot(val_accuracies[:epoch+1], marker='s', label=f'Val Acc ({val_acc:.2f}%)')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        if best_epoch != -1 :
             axes[1].scatter(best_epoch, max_val_accuracy, color='red', s=100, label=f'Best Val Acc ({max_val_accuracy:.2f}%)', zorder=5)
        plt.tight_layout()
        plot_filename = os.path.join(figures_path, f'training_progress.png') # Overwrite plot each epoch
        plt.savefig(plot_filename)
        plt.close(fig)

        print(f"\nEpoch {epoch+1}/{epochs_n} - Avg Loss: {epoch_avg_losses[epoch]:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% (Best Val: {max_val_accuracy:.2f}% at Epoch {best_epoch+1})")

    print(f"\nTraining finished. Best validation accuracy: {max_val_accuracy:.2f}% at epoch {best_epoch+1}")
    return all_step_losses, all_step_grad_norms


# --- Plotting Function for Loss Landscape ---

def plot_loss_landscape(min_loss_curve, max_loss_curve, figure_path):
    """Plots the min/max loss curves (simulated or real) and fills the area."""
    print(f"Plotting loss landscape simulation to {figure_path}...")
    plt.figure(figsize=(12, 6))
    steps = np.arange(len(min_loss_curve))

    plt.plot(steps, min_loss_curve, label='Min Loss Curve', color='blue', alpha=0.7)
    plt.plot(steps, max_loss_curve, label='Max Loss Curve', color='red', alpha=0.7)
    plt.fill_between(steps, min_loss_curve, max_loss_curve, color='gray', alpha=0.3, label='Loss Range')

    plt.title('Loss Landscape Indicator (Min/Max Step Loss)')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0) # Loss should be non-negative
    plt.tight_layout()

    try:
        plt.savefig(figure_path)
        print(f"Loss landscape plot saved to {figure_path}")
    except Exception as e:
        print(f"Error saving loss landscape plot: {e}")
    plt.close()


# --- Main Execution ---

if __name__ == '__main__':
    print(f"Setting random seed to {seed_value}...")
    set_random_seeds(seed=seed_value, device_type=device.type)

    print("Initializing model VGG_A...")
    model = VGG_A(num_classes=10) # CIFAR-10 has 10 classes

    print(f"Initializing Adam optimizer with lr={learning_rate}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Initializing CrossEntropyLoss criterion...")
    criterion = nn.CrossEntropyLoss()

    # Define paths for saving outputs
    best_model_save_path = os.path.join(models_path, 'vgg_a_cifar10_best.pth')
    loss_save_file = os.path.join(training_data_path, 'step_losses.txt')
    grad_save_file = os.path.join(training_data_path, 'step_grad_norms.txt')
    landscape_plot_file = os.path.join(figures_path, 'loss_landscape_indicator.png')

    print("Starting training...")
    step_losses, step_grads = train(
        model, optimizer, criterion, train_loader, val_loader, device,
        epochs_n=epochs, best_model_path=best_model_save_path
    )

    print("Training complete.")

    # Save loss and gradient data
    try:
        print(f"Saving step losses to {loss_save_file}...")
        np.savetxt(loss_save_file, np.array(step_losses), fmt='%.6f')
        print(f"Saving step gradient norms to {grad_save_file}...")
        # Use nan_to_num to handle potential NaNs from missing layers/grads
        np.savetxt(grad_save_file, np.nan_to_num(np.array(step_grads)), fmt='%.6f')
        print("Training data saved.")
    except Exception as e:
        print(f"Error saving training data: {e}")

    # --- Loss Landscape Plotting (Simulation) ---
    # This part simulates min/max curves using only the single run data.
    # For a real plot, load data from multiple runs.
    print("Simulating min/max curves using single run data...")
    min_curve = np.array(step_losses)
    max_curve = np.array(step_losses)

    # Plot the simulated landscape
    plot_loss_landscape(min_curve, max_curve, landscape_plot_file)

    print("\nScript finished.")