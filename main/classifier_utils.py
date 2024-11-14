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
    Save the training process metrics to a text file, including per-label metrics.

    Args:
        train_loss (float): The training loss for the current epoch.
        val_loss (float): The validation loss for the current epoch.
        val_acc (float): The overall validation accuracy for the current epoch.
        val_f1 (float): The weighted validation F1 score for the current epoch.
        epoch (int): The current epoch number.
        save_path (str): The file path to save the training process metrics.
        label_metrics (dict): Dictionary containing per-label metrics in the format:
                              {
                                  'label_name': {
                                      'accuracy': float,
                                      'precision': float,
                                      'recall': float,
                                      'f1': float
                                  },
                                  ...
                              }
    """
    if save_path is not None:
        print("Saving training process to txt file...")

        with open(save_path, 'a') as f:
            f.write("Epoch: {}\n".format(epoch))
            f.write("Train Loss: {:.4f}\n".format(train_loss))
            f.write("Validation Loss: {:.4f}\n".format(val_loss))
            f.write("Overall Validation Accuracy: {:.4f}\n".format(val_acc))
            f.write("Weighted Validation F1 Score: {:.4f}\n".format(val_f1))

            if label_metrics:
                f.write("Per-Label Metrics:\n")
                for label, metrics in label_metrics.items():
                    f.write("  {} - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}\n".format(
                        label, 
                        metrics['accuracy'], 
                        metrics['precision'], 
                        metrics['recall'], 
                        metrics['f1']
                    ))
            f.write("\n")
