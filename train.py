import matplotlib as mpl
mpl.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display  # Used for clearing output in notebooks
import yaml  # For loading configuration
import pandas as pd
import shutil

try:
    import sys
    sys.path.append('./')
    sys.path.append('../')
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from models.vgg import *
    # from models.vgg import VGG_A_BatchNorm
    from data.loaders import get_cifar_loader
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure 'models' and 'data' packages are correctly placed or added to PYTHONPATH.")
    exit()


def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 ** 2)
    return total_params, total_size_mb

def get_loss_and_accuracy(model, loader, device, criterion):
    """Calculates model loss and accuracy on the given data loader."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    model.train()  # Set model back to train mode
    return avg_loss, accuracy


def set_random_seeds(seed=0, device_type="cpu"):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed}")


def train(model, optimizer, criterion, train_loader, val_loader, device, scheduler=None,
          epochs_n=100, best_model_path=None, figures_path=None, training_data_path=None,
          step_csv_name=None, epoch_csv_name=None):
    """Trains the model, records metrics, and saves the best version."""
    model.to(device)

    # Prepare to record per-epoch metrics
    epoch_records = []  # Will hold dicts: {'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'}
    max_val_accuracy = 0.0
    best_epoch = -1

    # Prepare to record per-step metrics
    step_records = []  # Will hold dicts: {'step', 'epoch', 'loss', 'grad_norm'}
    step_count = 0
    batches_n = len(train_loader)

    print(f"Starting training for {epochs_n} epochs on {device}...")
    for epoch in tqdm(range(epochs_n), unit="epoch"):
        model.train()  # Ensure model is in training mode
        current_epoch_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            current_epoch_loss += loss.item()

            loss.backward()

            # Record gradient norm for a specific layer (e.g., model.classifier[4])
            try:
                grad = model.classifier[4].weight.grad
                if grad is not None:
                    grad_norm = grad.norm().item()
                else:
                    grad_norm = 0.0
            except (IndexError, AttributeError):
                grad_norm = np.nan

            optimizer.step()

            # Record this step's metrics
            step_records.append({
                'step': step_count,
                'epoch': epoch,
                'loss': loss.item(),
                'grad_norm': grad_norm
            })
            step_count += 1

        # End of epoch: compute average training loss over batches
        epoch_avg_train_loss = current_epoch_loss / batches_n

        # Compute training accuracy (and ignore training loss here since we already have avg)
        train_loss, train_acc = get_loss_and_accuracy(model, train_loader, device, criterion)
        # train_loss here is averaged over all train samples; to keep consistent, override
        train_loss = epoch_avg_train_loss

        # Compute validation loss and accuracy
        val_loss, val_acc = get_loss_and_accuracy(model, val_loader, device, criterion)

        epoch_records.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        if scheduler:
            scheduler.step()  # Step the scheduler

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
        axes[0].plot([r['train_loss'] for r in epoch_records], marker="o", label="Train Loss")
        axes[0].plot([r['val_loss'] for r in epoch_records], marker="s", label="Val Loss")
        axes[0].set_title(f"Epoch {epoch+1}/{epochs_n} - Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)
        # Accuracy plot
        axes[1].plot([r['train_acc'] for r in epoch_records], marker="o", label="Train Acc")
        axes[1].plot([r['val_acc'] for r in epoch_records], marker="s", label="Val Acc")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True)
        if best_epoch != -1:
            axes[1].scatter(
                best_epoch,
                max_val_accuracy,
                color="red",
                s=100,
                label=f"Best Val Acc ({max_val_accuracy:.2f}%)",
                zorder=5
            )
        plt.tight_layout()
        if figures_path:
            plot_filename = os.path.join(figures_path, "training_progress.png")
            plt.savefig(plot_filename)
        plt.close(fig)

        print(
            f"\nEpoch {epoch+1}/{epochs_n} - Avg Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% "
            f"(Best Val: {max_val_accuracy:.2f}% at Epoch {best_epoch+1})"
        )

    print(f"\nTraining finished. Best validation accuracy: {max_val_accuracy:.2f}% at epoch {best_epoch+1}")

    # Save per-step records to CSV
    if training_data_path and step_csv_name:
        step_df = pd.DataFrame(step_records)
        step_csv_path = os.path.join(training_data_path, step_csv_name)
        step_df.to_csv(step_csv_path, index=False)
        print(f"Saved step-wise metrics to {step_csv_path}")

    # Save per-epoch records to CSV
    if training_data_path and epoch_csv_name:
        epoch_df = pd.DataFrame(epoch_records)
        epoch_csv_path = os.path.join(training_data_path, epoch_csv_name)
        epoch_df.to_csv(epoch_csv_path, index=False)
        print(f"Saved epoch-wise metrics to {epoch_csv_path}")

    return step_records, epoch_records


def plot_loss_landscape(min_loss_curve, max_loss_curve, figure_path):
    """Plots the min/max loss curves (simulated or real) and fills the area."""
    print(f"Plotting loss landscape simulation to {figure_path}...")
    plt.figure(figsize=(12, 6))
    steps = np.arange(len(min_loss_curve))

    plt.plot(steps, min_loss_curve, label="Min Loss Curve", color="blue", alpha=0.7)
    plt.plot(steps, max_loss_curve, label="Max Loss Curve", color="red", alpha=0.7)
    plt.fill_between(steps, min_loss_curve, max_loss_curve, color="gray", alpha=0.3, label="Loss Range")

    plt.title("Loss Landscape Indicator (Min/Max Step Loss)")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)  # Loss should be non-negative
    plt.tight_layout()

    try:
        plt.savefig(figure_path)
        print(f"Loss landscape plot saved to {figure_path}")
    except Exception as e:
        print(f"Error saving loss landscape plot: {e}")
    plt.close()


if __name__ == "__main__":
    # --- Parse command-line arguments ---
    import argparse
    parser = argparse.ArgumentParser(description="Train a model with configurable parameters")
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # --- Load configuration from YAML ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Constants and Parameters from config ---
    training_cfg = config["training"]
    device_ids = training_cfg["device_ids"]
    num_workers = training_cfg["num_workers"]
    batch_size = training_cfg["batch_size"]
    seed_value = training_cfg["seed_value"]
    epochs = training_cfg["epochs"]
    target_gpu_id = training_cfg["target_gpu_id"]

    # --- Paths Setup from config ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    home_path = os.path.dirname(script_dir)

    figures_path = os.path.join(home_path, config["paths"]["figures"])
    models_path = os.path.join(home_path, config["paths"]["models"])
    training_data_path = os.path.join(home_path, config["paths"]["training_data"])

    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(training_data_path, exist_ok=True)

    # Copy the YAML config into the training_data directory for reproducibility
    try:
        shutil.copy(args.config, os.path.join(training_data_path, os.path.basename(args.config)))
        print(f"Copied config file to {training_data_path}")
    except Exception as e:
        print(f"Error copying config file: {e}")

    # --- Device Setup ---
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device(
        f"cuda:{target_gpu_id}"
        if torch.cuda.is_available() and target_gpu_id < torch.cuda.device_count()
        else "cpu"
    )
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(target_gpu_id)}")

    # --- Set random seeds ---
    print(f"Setting random seed to {seed_value}...")
    set_random_seeds(seed=seed_value, device_type=device.type)

    # --- Data Loaders ---
    print("Initializing data loaders...")
    try:
        data_root = config["data"]["root"]
        train_loader = get_cifar_loader(
            root=data_root,
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            augmentation_cfg=config["data"].get("augmentation", None)
        )
        val_loader = get_cifar_loader(
            root=data_root,
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            augmentation_cfg=None
        )
        print("Data loaders initialized.")

        print("Checking one batch from train_loader...")
        for X, y in train_loader:
            print(f"  Input X shape: {X.shape}")
            print(f"  Target y shape: {y.shape}")
            break
    except Exception as e:
        print(f"Error initializing or checking data loaders: {e}")
        exit()

    # --- Model Initialization based on config ---
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    num_classes = model_cfg.get("num_classes", 10)

    print("Initializing model...")
    if model_name == "VGG-base":
        model = VGG_A(num_classes=num_classes)
    elif model_name == "VGG-light":
        model = VGG_A_Light(num_classes=num_classes)
    elif model_name == "VGG-large":
        model = VGG_Large(num_classes=num_classes)
    elif model_name == "VGG-huge":
        model = VGG_Huge(num_classes=num_classes)
    elif model_name == "VGG-droupout":
        model = VGG_A_Dropout(num_classes=num_classes)
    elif model_name == "VGG-tanh":
        model = VGG_A_Tanh(num_classes=num_classes)
    elif model_name == "VGG-sigmoid":
        model = VGG_A_Sigmoid(num_classes=num_classes)
    elif model_name == "VGG-res":
        model = VGG_Res(num_classes=num_classes)
    elif model_name == "resnet50":
        model = ResNet50(num_classes=num_classes)
    elif model_name == "convnext":
        model = ConvNeXtLarge(num_classes=num_classes)
    elif model_name == "VGG-bn":
        model = VGG_A_BN(num_classes=num_classes)
    elif model_name=='resnet101':
        model=ResNet101(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name '{model_name}' in configuration.")
    total_params, total_size_mb = get_model_size(model)
    print(f"Model '{model_name}' total parameters: {total_params}")
    print(f"Approximate model size (float32): {total_size_mb:.2f} MB")
    # --- Optimizer and Scheduler Setup based on config ---
    optim_cfg = config["optimizer"]
    optim_name = optim_cfg["type"]
    learning_rate = optim_cfg.get("lr", 0.001)
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    momentum = optim_cfg.get("momentum", 0.9)

    print(f"Initializing {optim_name} optimizer with lr={learning_rate}...")
    if optim_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optim_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type '{optim_name}' in configuration.")

    # Scheduler (optional)
    scheduler = None
    if "scheduler" in config and config["scheduler"] is not None:
        sched_cfg = config["scheduler"]
        sched_name = sched_cfg["type"]
        sched_params = sched_cfg.get("params", {})

        print(f"Initializing {sched_name} scheduler with params={sched_params}...")
        if sched_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sched_params)
        elif sched_name == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **sched_params)
        elif sched_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_params)
        else:
            raise ValueError(f"Unsupported scheduler type '{sched_name}' in configuration.")

    criterion = nn.CrossEntropyLoss()
    print("Initialized CrossEntropyLoss criterion.")

    # --- Paths and filenames for saving outputs from config ---
    best_model_save_path = os.path.join(models_path, config["paths"]["best_model_filename"])
    landscape_plot_file = os.path.join(figures_path, config["paths"]["landscape_plot_filename"])
    step_csv_name = config["paths"]["step_records_filename"]
    epoch_csv_name = config["paths"]["epoch_metrics_filename"]

    # --- Start Training ---
    print("Starting training...")
    step_records, epoch_records = train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        scheduler=scheduler,
        epochs_n=epochs,
        best_model_path=best_model_save_path,
        figures_path=figures_path,
        training_data_path=training_data_path,
        step_csv_name=step_csv_name,
        epoch_csv_name=epoch_csv_name
    )

    print("Training complete.")

    # --- Loss Landscape Plotting (Simulation) ---
    print("Simulating min/max curves using single run data...")
    step_losses = [rec['loss'] for rec in step_records]
    min_curve = np.array(step_losses)
    max_curve = np.array(step_losses)

    # Plot the simulated landscape
    plot_loss_landscape(min_curve, max_curve, landscape_plot_file)

    print("\nScript finished.")
