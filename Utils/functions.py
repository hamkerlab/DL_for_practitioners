import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

import os
from tqdm.notebook import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any, Type
import pandas as pd
import numpy as np


def train_model(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        num_epochs: int = 10,
        checkpoint_path: Optional[str] = None,
        patience: Optional[int] = 5,  # Number of epochs to wait before early stopping. None for no early stopping
        min_delta: float = 0.001,  # Minimum change to qualify as improvement
        monitor: str = 'val_loss',  # Metric to monitor ('val_loss' or 'val_acc'), should be in history!
) -> Dict[str, List[float]]:
    """
    Train a model and return training history
    """

    # Initialize history dictionary to track metrics
    history: Dict[str, List[float]] = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    if checkpoint_path is not None:
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_path, exist_ok=True)
        best_value_checkpoint = float('inf') if 'loss' in monitor else 0.0

    # Move model to device
    model.to(device)

    # Early stopping
    if patience is not None:
        best_value_early_stop = float('inf') if 'loss' in monitor else 0.0
        no_improve_count = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for inputs, labels in train_pbar:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

        # Calculate epoch statistics
        epoch_train_loss: float = running_loss / len(train_loader.dataset)
        epoch_train_acc: float = 100 * correct / total

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Store metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch results
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save model checkpoint
        if checkpoint_path is not None:
            if history[monitor][-1] < best_value_checkpoint:
                best_value_checkpoint = history[monitor][-1]
                if isinstance(model, nn.DataParallel | nn.parallel.DistributedDataParallel):
                    torch.save(model.module.state_dict(), os.path.join(checkpoint_path, 'best_model.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'best_model.pth'))

        # Early stopping
        if patience is not None:
            if 'loss' in monitor:
                is_improved = history[monitor][-1] < (best_value_early_stop - min_delta)
            else:  # val_acc
                is_improved = history[monitor][-1] > (best_value_early_stop + min_delta)

            if is_improved:
                best_value_early_stop = history[monitor][-1]
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f'Early stopping triggered after epoch {epoch + 1}')
                    break

    return history


def evaluate_model(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate a model on a dataset
    """
    # Set model to evaluation mode
    model.eval()

    # Move model to device
    model.to(device)

    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        eval_pbar = tqdm(data_loader, desc='Evaluation')

        for inputs, labels in eval_pbar:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            eval_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

    # Calculate overall statistics
    avg_loss: float = running_loss / len(data_loader.dataset)
    accuracy: float = 100 * correct / total

    return avg_loss, accuracy


def test_model(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        class_names: list[str],
        print_per_class_summary: bool = True,
        collect_embeddings: bool = False,
):
    # Set model to evaluation mode
    model.eval()

    # Move model to device
    model.to(device)

    # Per-image results storage
    per_image_data = {
        'image_idx': [],
        'true_label': [],
        'true_class': [],
        'predicted_label': [],
        'predicted_class': [],
        'confidence': [],
        'correct': [],
        'probs': []
    }

    if collect_embeddings and hasattr(model, 'return_features'):
        all_embeddings = []
        embed_labels_idx = []
    else:
        collect_embeddings = False

    # Disable gradient calculation for inference
    with torch.no_grad():
        img_idx = 0  # Global image index counter. Only really useful for if the test_loader does not shuffle
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            batch_size = inputs.shape[0]

            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            if collect_embeddings:
                all_embeddings.append(outputs[1].cpu().numpy())
                outputs = outputs[0]
                embed_labels_idx.append(labels.cpu().numpy())
            else:
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence_values, predicted = torch.max(probabilities, dim=1)

            # Process each sample in the batch
            for i in range(batch_size):
                # Extract individual values
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                conf = confidence_values[i].item()
                probs = probabilities[i].cpu().numpy()

                # Store in per-image data
                per_image_data['image_idx'].append(img_idx + i)
                per_image_data['true_label'].append(true_label)
                per_image_data['true_class'].append(class_names[true_label])
                per_image_data['predicted_label'].append(pred_label)
                per_image_data['predicted_class'].append(class_names[pred_label])
                per_image_data['confidence'].append(conf)
                per_image_data['correct'].append(pred_label == true_label)
                per_image_data['probs'].append(probs)

            img_idx += batch_size

    # Create per-image DataFrame
    per_image_df = pd.DataFrame(per_image_data)

    # Calculate overall accuracy
    overall_accuracy = 100 * per_image_df['correct'].mean()

    # Create aggregated metrics
    aggregate_data = []
    for i, class_name in enumerate(class_names):
        # Filter for this class
        class_samples = per_image_df[per_image_df['true_label'] == i]

        class_accuracy = 100 * class_samples['correct'].mean()
        # Average confidence for correct predictions only
        correct_samples = class_samples[class_samples['correct']]
        avg_confidence = correct_samples['confidence'].mean() if len(correct_samples) > 0 else 0

        aggregate_data.append({
            'class_name': class_name,
            'accuracy': class_accuracy,
            'avg_confidence': avg_confidence,
            'support': len(class_samples)
        })

    # Create aggregate DataFrame
    aggregate_df = pd.DataFrame(aggregate_data)

    # Print summary
    print(f'Overall Test Accuracy: {overall_accuracy:.2f}%')
    if print_per_class_summary:
        print("\nPer-class Performance:")
        print(aggregate_df.to_string(index=False))

    if collect_embeddings:
        # Concatenate all batches (List[np.ndarray] to np.ndarray)
        embeddings = {
            'all_embeddings': np.concatenate(all_embeddings, axis=0),
            'all_labels': np.concatenate(embed_labels_idx, axis=0)
        }

        return aggregate_df, per_image_df, overall_accuracy, embeddings
    else:
        return aggregate_df, per_image_df, overall_accuracy


def get_model_predictions(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        class_info: dict,
        output_csv: str = None,
        return_probabilities: bool = False,
        top_k: int = 5,
):
    """
    Generate and return predictions for a model on the TinyImageNet test set.
    """
    # Set model to evaluation mode
    model.eval()

    # Move model to device
    model.to(device)

    # Prepare data structures for predictions
    predictions_data = {
        'image_id': [],
        'image_path': [],
        'predicted_class_idx': [],
        'predicted_class_id': [],
        'predicted_class_name': [],
        'confidence': []
    }

    # Add top-k predictions if requested
    if top_k > 1:
        for k in range(top_k):
            predictions_data[f'top_{k + 1}_class_idx'] = []
            predictions_data[f'top_{k + 1}_class_id'] = []
            predictions_data[f'top_{k + 1}_class_name'] = []
            predictions_data[f'top_{k + 1}_confidence'] = []

    # Add full probability distribution if requested
    if return_probabilities:
        num_classes = len(class_info['classes'])
        for i in range(num_classes):
            class_id = class_info['idx_to_class'][i]
            predictions_data[f'prob_{class_id}'] = []

    # Process each batch
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(test_loader, desc='Generating predictions')):
            # Move inputs to device
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Handle different model output formats (some return tuples with features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get top-k predictions
            if top_k > 1:
                topk_probs, topk_indices = torch.topk(probabilities, k=top_k, dim=1)
            else:
                # Get top-1 prediction
                confidence_values, predicted_indices = torch.max(probabilities, dim=1)

            # Process each sample in the batch
            for i in range(inputs.size(0)):
                # Get image ID from the test_loader (the file name without extension)
                img_path = test_loader.dataset.images[batch_idx * test_loader.batch_size + i]
                img_id = os.path.basename(img_path).split('.')[0]

                # Top-1 prediction info
                if top_k > 1:
                    pred_idx = topk_indices[i, 0].item()
                    confidence = topk_probs[i, 0].item()
                else:
                    pred_idx = predicted_indices[i].item()
                    confidence = confidence_values[i].item()

                pred_class_id = class_info['idx_to_class'][pred_idx]
                pred_class_name = class_info['class_names'].get(pred_class_id, f"class_{pred_idx}")

                # Store basic prediction info
                predictions_data['image_id'].append(img_id)
                predictions_data['image_path'].append(img_path)
                predictions_data['predicted_class_idx'].append(pred_idx)
                predictions_data['predicted_class_id'].append(pred_class_id)
                predictions_data['predicted_class_name'].append(pred_class_name)
                predictions_data['confidence'].append(confidence)

                # Store top-k predictions if requested
                if top_k > 1:
                    for k in range(top_k):
                        class_idx = topk_indices[i, k].item()
                        class_id = class_info['idx_to_class'][class_idx]
                        class_name = class_info['class_names'].get(class_id, f"class_{class_idx}")
                        class_prob = topk_probs[i, k].item()

                        predictions_data[f'top_{k + 1}_class_idx'].append(class_idx)
                        predictions_data[f'top_{k + 1}_class_id'].append(class_id)
                        predictions_data[f'top_{k + 1}_class_name'].append(class_name)
                        predictions_data[f'top_{k + 1}_confidence'].append(class_prob)

                # Store full probability distribution if requested
                if return_probabilities:
                    for j in range(num_classes):
                        class_id = class_info['idx_to_class'][j]
                        predictions_data[f'prob_{class_id}'].append(probabilities[i, j].item())

    # Create DataFrame from collected data
    predictions_df = pd.DataFrame(predictions_data)

    # Save to CSV if path is provided
    if output_csv:
        if os.path.dirname(output_csv) and not os.path.exists(os.path.dirname(output_csv)):
            os.makedirs(os.path.dirname(output_csv))
        predictions_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

    return predictions_df
