import torch

def save_model(model, save_path):
    """
    Save the model's state dictionary to the specified path.

    Args:
        model: The model to save.
        save_path (str): The file path to save the model.
    """
    if save_path is not None:
        torch.save(model.state_dict(), save_path)  # Save model state dictionary


def save_train_process(train_loss, val_loss, val_acc, val_f1, epoch, save_path, label_metrics=None):
    """
    Save the training process metrics to a text file.

    Args:
        train_loss (float): The training loss for the current epoch.
        val_loss (float): The validation loss for the current epoch.
        val_acc (float): The validation accuracy for the current epoch.
        val_f1 (float): The validation F1 score for the current epoch.
        epoch (int): The current epoch number.
        save_path (str): The file path to save the training process metrics.
        label_metrics (dict, optional): Per-label metrics including accuracy, precision, recall, F1, and Jaccard score.
    """
    if save_path is not None:
        print("Saving training process to txt file...")

        with open(save_path, 'a') as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Train Loss: {train_loss:.4f}\n")
            f.write(f"Validation Loss: {val_loss:.4f}\n")
            f.write(f"Validation Accuracy: {val_acc:.4f}\n")
            f.write(f"Validation F1 Score: {val_f1:.4f}\n")

            if label_metrics:
                f.write("Per-Label Metrics:\n")
                for label, metrics in label_metrics.items():
                    f.write(f"  {label}:\n")
                    f.write(f"    Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"    Precision: {metrics['precision']:.4f}\n")
                    f.write(f"    Recall: {metrics['recall']:.4f}\n")
                    f.write(f"    F1 Score: {metrics['f1']:.4f}\n")
                    f.write(f"    Jaccard Score: {metrics['jaccard']:.4f}\n")
            f.write("\n")
