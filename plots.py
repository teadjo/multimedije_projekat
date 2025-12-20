from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
import os
from scipy.stats import gaussian_kde, norm
from PIL import Image

from cfg import cfg, CFG



def create_heatmap(image_score_map: np.ndarray, patch_locations: List[Tuple[int, int]], patch_size: int, stride: int, image_resize: int):
    heatmap = np.zeros((image_resize, image_resize))
    count_map = np.zeros((image_resize, image_resize))
    
    for (top, left), score in zip(patch_locations, image_score_map):
        heatmap[top:top+patch_size, left:left+patch_size] += score
        count_map[top:top+patch_size, left:left+patch_size] += 1
    
    count_map[count_map == 0] = 1
    heatmap = heatmap / count_map
    
    return heatmap

def plot_evaluation_curves(y_true, y_scores, save_dir: str, category: str, model_name: str):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {category} ({model_name})', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'roc_curve_{model_name}.png'), dpi=300)
    plt.close()
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'g-', linewidth=2, label=f'PR (AP = {ap_score:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {category} ({model_name})', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pr_curve_{model_name}.png'), dpi=300)
    plt.close()
    
    return auc_score, ap_score, optimal_threshold

def plot_score_distributions(scores_good, scores_anomaly, optimal_threshold, save_path: str):
    plt.figure(figsize=(12, 6))
    
    plt.hist(scores_good, bins=50, alpha=0.5, color='green', 
             label=f'Good (n={len(scores_good)})', density=True)
    plt.hist(scores_anomaly, bins=50, alpha=0.5, color='red',
             label=f'Anomaly (n={len(scores_anomaly)})', density=True)
    
    plt.axvline(x=optimal_threshold, color='blue', linestyle='--', linewidth=2,
                label=f'Threshold = {optimal_threshold:.3f}')
    
    if len(scores_good) > 1:
        kde_good = gaussian_kde(scores_good)
        x_good = np.linspace(min(scores_good), max(scores_good), 1000)
        plt.plot(x_good, kde_good(x_good), 'darkgreen', linewidth=2)
    
    if len(scores_anomaly) > 1:
        kde_anomaly = gaussian_kde(scores_anomaly)
        x_anomaly = np.linspace(min(scores_anomaly), max(scores_anomaly), 1000)
        plt.plot(x_anomaly, kde_anomaly(x_anomaly), 'darkred', linewidth=2)
    
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Anomaly Scores', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_patch_heatmap(img_path: str, patch_scores: List[float], patch_locations: List[Tuple[int, int]],
                       image_score: float, optimal_threshold: float, save_path: str, cfg: CFG, model_name: str):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((cfg.image_resize, cfg.image_resize))
    img_np = np.array(img_resized) / 255.0
    
    heatmap = create_heatmap(np.array(patch_scores), patch_locations,
                            cfg.patch_size, cfg.patch_stride, cfg.image_resize)
    
    if heatmap.max() > heatmap.min():
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap_norm = heatmap
    
    cmap = plt.cm.jet
    heatmap_rgb = cmap(heatmap_norm)[:, :, :3]
    blended = 0.6 * img_np + 0.3 * heatmap_rgb
    
    is_anomaly = image_score > optimal_threshold
    anomaly_text = "ANOMALY" if is_anomaly else "NORMAL"
    text_color = 'red' if is_anomaly else 'green'
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    im = axes[1].imshow(heatmap_norm, cmap='jet')
    axes[1].set_title('Anomaly Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    axes[2].imshow(blended)
    axes[2].set_title('Blended Visualization', fontsize=12)
    axes[2].axis('off')
    
    axes[3].hist(patch_scores, bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[3].axvline(image_score, color='red', linestyle='--', linewidth=2, label=f'Image score: {image_score:.3f}')
    axes[3].axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {optimal_threshold:.3f}')
    axes[3].set_xlabel('Patch Score', fontsize=10)
    axes[3].set_ylabel('Count', fontsize=10)
    axes[3].set_title(f'Patch Score Distribution\n{anomaly_text}', fontsize=12, color=text_color)
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(f'{Path(img_path).name} | Image Score: {image_score:.3f} | Model: {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
