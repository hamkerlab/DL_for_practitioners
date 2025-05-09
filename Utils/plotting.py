import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms as T

import matplotlib.pyplot as plt
from PIL import ImageDraw
from typing import List, Tuple, Dict, Optional, Union
import pandas as pd

from Utils.little_helpers import get_overlap, get_bboxes


def visualize_training_results(train_losses: List[float],
                               train_accs: List[float],
                               test_losses: List[float],
                               test_accs: List[float],
                               output_dir: Optional[str]) -> None:
    # Plot loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Epoch')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs. Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'training_results.png'), dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_test_results(aggregate_df: pd.DataFrame,
                           per_image_df: pd.DataFrame,
                           overall_accuracy: float,
                           output_dir: Optional[str] = None,
                           class_subset: Optional[Union[List[str], Tuple[str]]] = None,
                           max_classes_display: int = 20,
                           ):
    """
    Visualize test results with support for class subsetting.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Filter data based on class_subset if provided, otherwise limit by max_classes_display
        if class_subset:
            # Ensure all classes in subset exist in the data
            valid_classes = set(aggregate_df['class_name'])
            invalid_classes = [c for c in class_subset if c not in valid_classes]
            if invalid_classes:
                print(f"Warning: Some specified classes not found in data: {invalid_classes}")

            # Filter to include only specified classes
            filtered_aggregate_df = aggregate_df[aggregate_df['class_name'].isin(class_subset)]
            filtered_per_image_df = per_image_df[
                (per_image_df['true_class'].isin(class_subset)) |
                (per_image_df['predicted_class'].isin(class_subset))
                ]

            # If no classes match, warn and use original data
            if len(filtered_aggregate_df) == 0:
                print("Warning: No classes from subset found in data. Using all available classes.")
                filtered_aggregate_df = aggregate_df
                filtered_per_image_df = per_image_df
        else:
            # If too many classes, select top classes by support count
            if len(aggregate_df) > max_classes_display:
                print(f"More than {max_classes_display} classes found. "
                      f"Displaying top {max_classes_display} by support count.")
                filtered_aggregate_df = aggregate_df.sort_values('support', ascending=False).head(max_classes_display)
                top_classes = filtered_aggregate_df['class_name'].tolist()
                filtered_per_image_df = per_image_df[
                    (per_image_df['true_class'].isin(top_classes)) |
                    (per_image_df['predicted_class'].isin(top_classes))
                    ]
            else:
                filtered_aggregate_df = aggregate_df
                filtered_per_image_df = per_image_df

        # Get number of classes for plot sizing
        num_classes = len(filtered_aggregate_df)

        # Set up figure resolution based on the number of classes
        if num_classes > 50:
            plt.figure(figsize=(40, 40))
            fontsize = 20
            rotation = 90
        elif num_classes > 10:
            plt.figure(figsize=(25, 25))
            fontsize = 12
            rotation = 90
        else:
            plt.figure(figsize=(15, 15))
            fontsize = 10
            rotation = 45

        # 1. Plot accuracy by class
        plt.subplot(2, 2, 1)
        bar_plot = sns.barplot(x='class_name', y='accuracy', data=filtered_aggregate_df)
        plt.axhline(y=overall_accuracy, color='r', linestyle='--',
                    label=f'Overall Accuracy: {overall_accuracy:.2f}%')
        plt.title('Per-class Accuracy', fontsize=fontsize)
        plt.xticks(rotation=rotation, ha='right')
        plt.ylabel('Accuracy (%)', fontsize=fontsize)
        plt.xlabel('', fontsize=fontsize)  # Hide x-label as it's obvious
        plt.legend()

        # Adjust x-tick label size if many classes
        if num_classes > 10:
            bar_plot.set_xticklabels(bar_plot.get_xticklabels(), fontsize=max(6, fontsize - 4))

        # 2. Plot confidence vs accuracy by class
        plt.subplot(2, 2, 2)
        plt.scatter(
            filtered_aggregate_df['accuracy'],
            filtered_aggregate_df['avg_confidence'] * 100,
            s=filtered_aggregate_df['support'] / filtered_aggregate_df['support'].max() * 300,
            alpha=0.6
        )

        # Add class labels to the scatter plot - adjust text size if many classes
        text_size = max(6, fontsize - 4) if num_classes > 15 else fontsize - 2
        for i, txt in enumerate(filtered_aggregate_df['class_name']):
            plt.annotate(txt,
                         (filtered_aggregate_df['accuracy'].iloc[i],
                          filtered_aggregate_df['avg_confidence'].iloc[i] * 100),
                         fontsize=text_size)

        plt.title('Confidence vs Accuracy (point size indicates class frequency)', fontsize=fontsize)
        plt.xlabel('Accuracy (%)', fontsize=fontsize)
        plt.ylabel('Average Confidence (%)', fontsize=fontsize)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 3. Plot confusion matrix - only if we have true labels (not for test set with unknown labels)
        plt.subplot(2, 2, 3)

        # Check if we have actual true labels (not dummy -1 values)
        if filtered_per_image_df['true_class'].nunique() > 1 and -1 not in filtered_per_image_df['true_class'].unique():
            # Create confusion matrix
            cm = pd.crosstab(
                filtered_per_image_df['true_class'],
                filtered_per_image_df['predicted_class'],
                normalize='index'
            )

            # Adjust annot and fontsize based on matrix size
            annot = True if len(cm) <= 30 else False
            annot_kws = {'fontsize': max(6, fontsize - 6)} if len(cm) > 10 else {}

            sns.heatmap(cm, annot=annot, fmt='.2f', cmap='Blues', annot_kws=annot_kws)
            plt.title('Confusion Matrix (Normalized)', fontsize=fontsize)
            plt.ylabel('True Class', fontsize=fontsize)
            plt.xlabel('Predicted Class', fontsize=fontsize)
        else:
            plt.text(0.5, 0.5, "True labels unavailable for confusion matrix",
                     ha='center', va='center', fontsize=fontsize)
            plt.axis('off')

        # 4. Plot confidence distribution by correctness
        plt.subplot(2, 2, 4)

        # Check if we have actual correctness data
        if 'correct' in filtered_per_image_df.columns and -1 not in filtered_per_image_df['true_class'].unique():
            sns.histplot(data=filtered_per_image_df, x='confidence', hue='correct',
                         bins=20, element='step', common_norm=False, stat='density')
            plt.title('Confidence Distribution', fontsize=fontsize)
            plt.xlabel('Confidence Score', fontsize=fontsize)
            plt.ylabel('Density', fontsize=fontsize)
        else:
            # For test set with unknown labels
            sns.histplot(data=filtered_per_image_df, x='confidence', bins=20)
            plt.title('Prediction Confidence Distribution (true labels unknown)', fontsize=fontsize)
            plt.xlabel('Confidence Score', fontsize=fontsize)
            plt.ylabel('Count', fontsize=fontsize)

        plt.tight_layout()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_file = 'test_results.png'
            if class_subset:
                # Create a filename based on subset
                subset_str = '_'.join(class_subset[:3])  # Use first 3 classes in name
                if len(class_subset) > 3:
                    subset_str += "_etc"
                output_file = f'test_results_{subset_str}.png'

            plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("Matplotlib and/or seaborn not available for visualization.")


def visualize_embeddings_tsne(embeddings: np.ndarray | torch.Tensor,
                              labels: np.ndarray | torch.Tensor,
                              output_dir: Optional[str],
                              class_names: List[str],
                              n_samples: int = 2000,
                              num_components: int = 2) -> None:
    try:
        from sklearn.manifold import TSNE

        # check if data is numpy
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        # Subsample if too many points
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]

        # Apply t-SNE
        tsne = TSNE(n_components=num_components, random_state=42, perplexity=30, max_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(12, 10))
        for i, class_name in enumerate(class_names):
            indices = labels == i
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=class_name, alpha=0.6)

        plt.title('t-SNE Visualization of Embeddings')
        plt.legend()
        plt.tight_layout()
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'embeddings_tsne.png'), dpi=300)
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("scikit-learn not installed, skipping embedding visualization")


def visualize_embeddings_pca(embeddings: np.ndarray | torch.Tensor,
                             labels: np.ndarray | torch.Tensor,
                             output_dir: Optional[str],
                             class_names: List[str],
                             n_samples: int = 2000,
                             n_components: int = 2) -> None:
    try:
        from sklearn.decomposition import PCA

        # check if data is numpy
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        # Subsample if too many points
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]

        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)

        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        explained_variance_sum = explained_variance.sum() * 100  # Convert to percentage

        # Plot
        plt.figure(figsize=(12, 10))
        for i, class_name in enumerate(class_names):
            indices = labels == i
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=class_name, alpha=0.6)

        plt.title(f'PCA Visualization of Embeddings\n(Explained variance: {explained_variance_sum:.2f}%)')
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
        plt.legend()
        plt.tight_layout()
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'embeddings_pca.png'), dpi=300)
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("scikit-learn not installed, skipping embedding visualization")


def visualize_misclassified(model: nn.Module,
                            test_loader: DataLoader,
                            device: torch.device,
                            output_dir: Optional[str],
                            class_names: List[str],
                            num_examples: int = 25, ) -> None:
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)

            # Find misclassified examples
            incorrect_mask = preds != labels
            if incorrect_mask.any():
                misclassified_idx = torch.where(incorrect_mask)[0]
                for idx in misclassified_idx:
                    misclassified_images.append(inputs[idx].cpu())
                    misclassified_labels.append(labels[idx].item())
                    misclassified_preds.append(preds[idx].item())

                    if len(misclassified_images) >= num_examples:
                        break

            if len(misclassified_images) >= num_examples:
                break

    # Plot misclassified examples
    num_examples = min(len(misclassified_images), num_examples)
    rows = int(np.ceil(num_examples / 5))

    fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes]

    for i in range(num_examples):
        # Plot the image and check if its grayscale or RGB. If the latter transpose 3ximg_sizeximg_size to img_sizeximg_sizex3
        if len(misclassified_images[i].shape) == 3:
            misclassified_images[i] = misclassified_images[i].permute(1, 2, 0)
        img = misclassified_images[i].squeeze().numpy()
        true_label = class_names[misclassified_labels[i]]
        pred_label = class_names[misclassified_preds[i]]

        axes[i].imshow(img)
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_examples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'misclassified_examples.png'), dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_patch_embeddings(model: nn.Module,
                               num_components: int = 21,  # Number of components to visualize, should be a multiple of 7
                               output_dir: Optional[str] = None) -> None:
    """
    Visualize the patch embedding filters or their principal components.
    Similar to Figure 7 (left) in the ViT paper.
    """
    # Extract the weights from the patch embedding layer
    weights = model.proj.weight.data.clone() if hasattr(model, 'proj') else model.patch_embed.proj.weight.data.clone()

    # Set up grid dimensions
    grid_cols = min(7, num_components)
    grid_rows = (num_components + grid_cols - 1) // grid_cols  # Ceiling division

    filters_to_display = weights[:num_components].cpu().numpy()
    title = f"Patch embedding filters (first {num_components} filters)"

    # Create a grid to visualize the filters
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
    fig.suptitle(title, fontsize=16)

    # Plot each filter
    for idx in range(num_components):
        i, j = idx // grid_cols, idx % grid_cols

        # Get the filter
        filt = filters_to_display[idx]

        # Normalize for better visualization
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-6)

        # Transpose to (H, W, C) for matplotlib
        axs[i, j].imshow(np.transpose(filt, (1, 2, 0)))
        axs[i, j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'patch_embeddings.png'), dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_position_embeddings(model: nn.Module,
                                  figsize: Tuple[int, int] = (10, 10),
                                  output_dir: Optional[str] = None) -> None:
    import torch.nn.functional as F

    # Extract position embeddings
    pos_embed = model.pos_embed.detach()

    # Get grid dimensions
    if hasattr(model.patch_embed, 'num_patches_h') and hasattr(model.patch_embed, 'num_patches_w'):
        h = model.patch_embed.num_patches_h
        w = model.patch_embed.num_patches_w
    else:
        # Assume square patches
        num_patches = model.patch_embed.num_patches
        h = w = int(np.sqrt(num_patches))

    # Create figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Cosine similarity of the position embedding", fontsize=24)

    # Create a subplot for each position embedding
    for i in range(1, pos_embed.shape[1]):
        sim = F.cosine_similarity(pos_embed[0, i:i + 1], pos_embed[0, 1:], dim=1)
        sim = sim.reshape((h, w)).detach().cpu().numpy()
        ax = fig.add_subplot(h, w, i)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(sim)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'pos_embeddings.png'), dpi=300)
        plt.close()
    else:
        plt.show()


def plot_boxes(data, labels, classes, color='orange', min_confidence=0.2, max_overlap=0.5,image_size = (448,448), file='test_img'):
    """Plots bounding boxes on the given image."""

    grid_size = 7
    num_classes = len(classes)

    grid_size_x = data.size(dim=2) / grid_size#S
    grid_size_y = data.size(dim=1) / grid_size#S
    m = labels.size(dim=0)
    n = labels.size(dim=1)

    bboxes = get_bboxes(m,n,labels, grid_size_x, grid_size_y,num_classes, min_confidence, image_size)

    # Sort by highest to lowest confidence
    bboxes = sorted(bboxes, key=lambda x: x[3], reverse=True)
    # Calculate IOUs between each pair of boxes
    num_boxes = len(bboxes)
    iou = [[0 for _ in range(num_boxes)] for _ in range(num_boxes)]
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou[i][j] = get_overlap(bboxes[i], bboxes[j])

    # Non-maximum suppression and render image
    data = data.numpy()
    data = np.moveaxis(data,0,-1)
    
    image = T.ToPILImage()(data)
    draw = ImageDraw.Draw(image)
    discarded = set()
    for i in range(num_boxes):
        if i not in discarded:
            tl, width, height, confidence, class_index = bboxes[i]

            # Decrease confidence of other conflicting bboxes
            for j in range(num_boxes):
                other_class = bboxes[j][4]
                if j != i and other_class == class_index and iou[i][j] > max_overlap:
                    discarded.add(j)

            # Annotate image
            draw.rectangle((tl, (tl[0] + width, tl[1] + height)), outline='orange')
            text_pos = (max(0, tl[0]), max(0, tl[1] - 11))
            text = f'{classes[class_index]} {round(confidence * 100, 1)}%'
            text_bbox = draw.textbbox(text_pos, text)
            draw.rectangle(text_bbox, fill='orange')
            draw.text(text_pos, text)
    if file is None:
        print('Show Image')
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        #image.show()
    else:
        output_dir = 'output' #os.path.dirname('output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not file.endswith('.png'):
            file += '.png'
        image.save(output_dir+'/'+file)

