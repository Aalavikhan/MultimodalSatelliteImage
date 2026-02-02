# -*- coding: utf-8 -*-
"""
Extract Metrics from Your Trained Model
This script shows how to properly extract test metrics from your trained model
and use them with the visualization framework
"""

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def extract_metrics_from_checkpoint(checkpoint_path='best_spatiotemporal_model.pt'):
    """
    Extract metrics directly from your saved checkpoint
    
    Args:
        checkpoint_path: Path to your model checkpoint
        
    Returns:
        dict: Dictionary with all metrics
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract metrics that were saved during training
    metrics = {
        'r2': checkpoint.get('r2', None),
        'wape': checkpoint.get('wape', None),
        'mape': checkpoint.get('mape', None),
    }
    
    print("\n✅ Extracted metrics from checkpoint:")
    for key, value in metrics.items():
        if value is not None:
            print(f"   {key}: {value}")
    
    return metrics


def compute_all_metrics_from_predictions(y_true, y_pred):
    """
    Compute all metrics from predictions
    Use this if your checkpoint doesn't have all metrics
    
    Args:
        y_true: Ground truth values (numpy array)
        y_pred: Predicted values (numpy array)
        
    Returns:
        dict: Dictionary with all computed metrics
    """
    # Ensure arrays
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    # Ensure positive predictions
    y_pred = np.maximum(y_pred, 0)
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # WAPE (weighted absolute percentage error)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    # sMAPE (symmetric mean absolute percentage error)
    smape = np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))
    
    # Within tolerance
    pct_diff = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
    within_5 = np.mean(pct_diff <= 5) * 100
    within_10 = np.mean(pct_diff <= 10) * 100
    within_15 = np.mean(pct_diff <= 15) * 100
    within_20 = np.mean(pct_diff <= 20) * 100
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'wape': wape,
        'smape': smape,
        'within_5': within_5,
        'within_10': within_10,
        'within_15': within_15,
        'within_20': within_20
    }
    
    return metrics


def get_predictions_from_test_loader(model, test_loader, device, targ_scaler):
    """
    Get predictions from your trained model on test set
    
    Args:
        model: Your trained SpatioTemporalSwinModel
        test_loader: DataLoader for test set
        device: torch.device
        targ_scaler: Target scaler used during training
        
    Returns:
        y_true, y_pred: Ground truth and predictions (numpy arrays)
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    
    print("🔮 Generating predictions on test set...")
    
    with torch.no_grad():
        for batch_idx, (img_seqs, feats, labels) in enumerate(test_loader):
            img_seqs = img_seqs.to(device)
            feats = feats.to(device)
            
            outputs = model(img_seqs, feats)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels.numpy())
    
    # Concatenate all batches
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Inverse transform to original scale
    preds = targ_scaler.inverse_transform(preds)
    targets = targ_scaler.inverse_transform(targets)
    
    print(f"✅ Generated {len(preds)} predictions")
    
    return targets.ravel(), preds.ravel()


def save_predictions_for_analysis(y_true, y_pred, save_path='predictions.csv'):
    """
    Save predictions and ground truth for further analysis
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Path to save CSV
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'ground_truth': y_true,
        'prediction': y_pred,
        'absolute_error': np.abs(y_true - y_pred),
        'percentage_error': np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
    })
    
    df.to_csv(save_path, index=False)
    print(f"💾 Predictions saved to: {save_path}")
    
    # Print summary statistics
    print("\n📊 Prediction Summary:")
    print(f"   Mean absolute error: {df['absolute_error'].mean():.2f} kWh")
    print(f"   Mean percentage error: {df['percentage_error'].mean():.2f}%")
    print(f"   Median percentage error: {df['percentage_error'].median():.2f}%")
    print(f"   95th percentile error: {df['percentage_error'].quantile(0.95):.2f}%")


# ============================================================================
# COMPLETE EXAMPLE: Extract Metrics and Update Visualizations
# ============================================================================

def complete_example():
    """
    Complete example showing how to:
    1. Load your trained model
    2. Generate predictions on test set
    3. Compute all metrics
    4. Update visualizations
    """
    print("="*80)
    print("📊 COMPLETE EXAMPLE: EXTRACT METRICS AND UPDATE VISUALIZATIONS")
    print("="*80)
    
    # ========================================================================
    # METHOD 1: Load metrics directly from checkpoint
    # ========================================================================
    print("\n" + "="*80)
    print("METHOD 1: Extract from Checkpoint")
    print("="*80)
    
    checkpoint_path = 'best_spatiotemporal_model.pt'
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Metrics might be stored directly in checkpoint
        metrics_from_checkpoint = {
            'r2': checkpoint.get('r2'),
            'rmse': checkpoint.get('rmse'),
            'mape': checkpoint.get('mape'),
            'wape': checkpoint.get('wape'),
            'mae': checkpoint.get('mae'),
            'within_5': checkpoint.get('within_5'),
            'within_10': checkpoint.get('within_10')
        }
        
        print("✅ Loaded metrics from checkpoint:")
        for key, value in metrics_from_checkpoint.items():
            if value is not None:
                print(f"   {key}: {value}")
        
    except FileNotFoundError:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Run your training script first!")
        return
    
    # ========================================================================
    # METHOD 2: Recompute metrics from model predictions (MORE ACCURATE)
    # ========================================================================
    print("\n" + "="*80)
    print("METHOD 2: Recompute from Model (Recommended)")
    print("="*80)
    
    # This would require loading your model and test loader
    # Uncomment and modify if you want to recompute:
    
    """
    from spatiotemporal_swin import SpatioTemporalSwinModel, load_spatiotemporal_data
    from torch.utils.data import DataLoader
    
    # Load data
    csv_path = 'YOUR_PATH/data.csv'
    image_dir = 'YOUR_PATH/images_png_view'
    
    image_sequences, features, targets, years = load_spatiotemporal_data(
        csv_path, image_dir, sequence_length=6
    )
    
    # Create test loader (use same split as training)
    # ... (split data) ...
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalSwinModel()
    model.load_state_dict(checkpoint['model'])
    
    # Get predictions
    y_true, y_pred = get_predictions_from_test_loader(
        model, test_loader, device, targ_scaler
    )
    
    # Compute all metrics
    metrics_recomputed = compute_all_metrics_from_predictions(y_true, y_pred)
    
    print("✅ Recomputed metrics:")
    for key, value in metrics_recomputed.items():
        print(f"   {key}: {value:.4f}")
    
    # Save predictions for analysis
    save_predictions_for_analysis(y_true, y_pred, 'test_predictions.csv')
    """
    
    # ========================================================================
    # METHOD 3: Use your actual test results from training output
    # ========================================================================
    print("\n" + "="*80)
    print("METHOD 3: Manual Entry (Quick & Easy)")
    print("="*80)
    
    # Look at your training script output and copy the final test metrics here
    your_metrics = {
        'r2': 0.9876,      # ← Replace with your actual test R²
        'rmse': 6.12,      # ← Replace with your actual test RMSE
        'mape': 12.05,     # ← Replace with your actual test MAPE
        'wape': 9.80,      # ← Replace with your actual test WAPE
        'mae': 4.50,       # ← Replace with your actual test MAE
        'smape': 11.5,     # ← Replace with your actual test sMAPE
        'within_5': 45.2,  # ← Replace with your actual within ±5%
        'within_10': 68.5  # ← Replace with your actual within ±10%
    }
    
    print("📝 Using manually entered metrics:")
    for key, value in your_metrics.items():
        print(f"   {key}: {value}")
    
    # ========================================================================
    # Generate Visualizations with Your Metrics
    # ========================================================================
    print("\n" + "="*80)
    print("🎨 Generating Visualizations")
    print("="*80)
    
    from visualization_graphs import generate_all_visualizations
    
    generate_all_visualizations(
        comparison_results_path='comparison_results.json',
        your_model_metrics=your_metrics
    )
    
    print("\n✅ All done! Check the generated PNG files.")


# ============================================================================
# UTILITY: Pretty Print Metrics
# ============================================================================

def print_metrics_nicely(metrics, title="Model Performance"):
    """Pretty print metrics in a formatted table"""
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)
    
    # Group metrics
    main_metrics = ['r2', 'rmse', 'mae', 'mape', 'wape', 'smape']
    tolerance_metrics = ['within_5', 'within_10', 'within_15', 'within_20']
    
    print("\n🎯 Main Metrics:")
    print("-" * 80)
    for metric in main_metrics:
        if metric in metrics and metrics[metric] is not None:
            if metric == 'r2':
                print(f"   {metric.upper():10s}: {metrics[metric]:.4f}")
            else:
                print(f"   {metric.upper():10s}: {metrics[metric]:.2f}")
    
    print("\n✅ Accuracy Within Tolerance:")
    print("-" * 80)
    for metric in tolerance_metrics:
        if metric in metrics and metrics[metric] is not None:
            tolerance = metric.split('_')[1]
            print(f"   Within ±{tolerance}%: {metrics[metric]:.1f}%")
    
    print("="*80)


if __name__ == "__main__":
    # Run the complete example
    complete_example()
    
    # Or use individual functions:
    
    # Extract from checkpoint
    # metrics = extract_metrics_from_checkpoint('best_spatiotemporal_model.pt')
    
    # Compute from predictions
    # y_true = np.array([...])  # Your ground truth
    # y_pred = np.array([...])  # Your predictions
    # metrics = compute_all_metrics_from_predictions(y_true, y_pred)
    
    # Print nicely
    # print_metrics_nicely(metrics, title="Test Set Results")
