# -*- coding: utf-8 -*-
"""
Comparison Models for Energy Consumption Prediction
UPDATED: Replaced VGG16/ResNet50 with EfficientNet-B6/ResNet101 (>35M params)
Removed Random Forest (as requested)
Includes: Linear Regression, SVM, EfficientNet-B6+Tabular, ResNet-101+Tabular
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import cv2
import rasterio
import warnings
from PIL import Image

warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_single_frame_data(csv_path, image_dir):
    """
    Load data for single-frame models (no temporal sequences)
    Returns images and metadata for the current timestep only
    Matches the original spatiotemporal_swin.py data loading logic
    """
    print("\n📂 Loading single-frame data...")
    
    # Load and preprocess CSV (same as original)
    df = pd.read_csv(csv_path)
    df = df[(df['Energy Use per Capita (kWh)'] > 0) & 
            (df['Population'] > 0) & 
            (df['Area (Sq. Km)'] > 0)]
    
    df['date'] = pd.to_datetime(df['Date (month/year)'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df = df.sort_values(['Country', 'date']).reset_index(drop=True)
    
    print(f"   Total rows: {len(df)}")
    print(f"   Countries: {df['Country'].nunique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Feature engineering (same as original)
    for col in ['Population', 'Area (Sq. Km)']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['log_population'] = np.log1p(df['Population'].astype(float))
    df['log_area'] = np.log1p(df['Area (Sq. Km)'].astype(float))
    df['density'] = df['Population'].astype(float) / (df['Area (Sq. Km)'].astype(float) + 1)
    df['log_density'] = np.log1p(df['density'])
    df['month_sin'] = np.sin(2 * np.pi * df['month'].astype(float) / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'].astype(float) / 12)
    year_min, year_max = df['year'].min(), df['year'].max()
    df['year_normalized'] = (df['year'].astype(float) - year_min) / (year_max - year_min + 1e-8)
    
    df = df.fillna(0)
    
    feature_cols = [
        'log_population', 'log_area', 'log_density',
        'month_sin', 'month_cos', 'year_normalized'
    ]
    
    # Load images
    print(f"\n🖼️ Loading images...")
    images = []
    features = []
    targets = []
    years = []
    
    for idx, row in df.iterrows():
        # Image path format: {image_dir}/{Country}/{Country}_{year}_{month:02d}.png
        img_path = os.path.join(image_dir, row['Country'],
                                f"{row['Country']}_{row['year']}_{row['month']:02d}.png")
        
        if os.path.exists(img_path):
            try:
                # Use rasterio (same as original)
                with rasterio.open(img_path) as src:
                    image = src.read(1)
                    
                    if image is None or np.isnan(image).any() or np.isinf(image).any():
                        continue
                    
                    # Resize to 64x64 (same as original)
                    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
                    
                    # Normalize (simple min-max)
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                    image = np.clip(image, 0, 1).astype(np.float32)
                    
                    # Convert single channel to 3-channel for pretrained models
                    image = np.stack([image, image, image], axis=-1)
                    
                    images.append(image)
                    
                    # Extract features
                    feat = [float(row[col]) for col in feature_cols]
                    features.append(feat)
                    
                    # Target
                    targets.append(row['Energy Use per Capita (kWh)'])
                    years.append(row['year'])
                    
            except Exception as e:
                continue
    
    images = np.array(images)
    features = np.array(features)
    targets = np.array(targets)
    years = np.array(years)
    
    print(f"✅ Loaded {len(images)} samples")
    print(f"   Images shape: {images.shape}")
    print(f"   Features shape: {features.shape}")
    print(f"   Features: {', '.join(feature_cols)}")
    
    if len(images) == 0:
        raise ValueError("No images loaded! Check your image directory structure.")
    
    return images, features, targets, years


# ============================================================================
# DATASET CLASS
# ============================================================================

class EnergyDataset(Dataset):
    """Dataset for energy prediction with images and tabular features"""
    
    def __init__(self, images, features, targets, transform=None):
        self.images = images  # Already 3-channel from load_single_frame_data
        self.features = features
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        feature = self.features[idx]
        target = self.targets[idx]
        
        # Image is already in [0, 1] range and 3-channel from loading
        # Convert to PIL for transforms
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.FloatTensor(feature), torch.FloatTensor([target])


# ============================================================================
# HYBRID EFFICIENTNET-B6 MODEL (Images + Tabular Features)
# ============================================================================

class EfficientNetB6Hybrid(nn.Module):
    """
    EfficientNet-B6 + Tabular Features Hybrid Model
    EfficientNet-B6: 43M parameters (meets >35M requirement)
    Designed for efficient image processing with compound scaling
    """
    
    def __init__(self, num_tabular_features=6, pretrained=True):
        super(EfficientNetB6Hybrid, self).__init__()
        
        # Image branch: EfficientNet-B6 (43M params)
        if pretrained:
            efficientnet = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT)
        else:
            efficientnet = models.efficientnet_b6(weights=None)
        
        # Extract feature extractor (remove classifier)
        self.image_features = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # EfficientNet-B6 output: 2304 features
        self.image_fc = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        # Tabular branch: MLP for tabular features
        self.tabular_fc = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion: Concatenate image features (256) + tabular features (64)
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, tabular):
        # Image branch
        x_img = self.image_features(image)
        x_img = self.avgpool(x_img)
        x_img = torch.flatten(x_img, 1)
        x_img = self.image_fc(x_img)
        
        # Tabular branch
        x_tab = self.tabular_fc(tabular)
        
        # Fusion
        x = torch.cat([x_img, x_tab], dim=1)
        x = self.fusion_fc(x)
        
        return x


# ============================================================================
# HYBRID RESNET-101 MODEL (Images + Tabular Features)
# ============================================================================

class ResNet101Hybrid(nn.Module):
    """
    ResNet-101 + Tabular Features Hybrid Model
    ResNet-101: 44.5M parameters (meets >35M requirement)
    Standard deep residual learning baseline
    """
    
    def __init__(self, num_tabular_features=6, pretrained=True):
        super(ResNet101Hybrid, self).__init__()
        
        # Image branch: ResNet-101 feature extractor (44.5M params)
        if pretrained:
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            resnet = models.resnet101(weights=None)
        
        # Remove the avgpool and fc layers
        self.image_features = nn.Sequential(*list(resnet.children())[:-2])
        
        # ResNet output: 2048 channels
        # Use adaptive pooling to handle any input size
        self.image_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Additional FC layers for image features
        self.image_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Tabular branch: MLP for tabular features
        self.tabular_fc = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion: Concatenate image features (256) + tabular features (64)
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, tabular):
        # Image branch
        x_img = self.image_features(image)
        x_img = self.image_avgpool(x_img)
        x_img = torch.flatten(x_img, 1)
        x_img = self.image_fc(x_img)
        
        # Tabular branch
        x_tab = self.tabular_fc(tabular)
        
        # Fusion
        x = torch.cat([x_img, x_tab], dim=1)
        x = self.fusion_fc(x)
        
        return x
        
        # Tabular branch
        x_tab = self.tabular_fc(tabular)
        
        # Fusion
        x = torch.cat([x_img, x_tab], dim=1)
        x = self.fusion_fc(x)
        
        return x


# ============================================================================
# TRAINING FUNCTION FOR HYBRID MODELS
# ============================================================================

def train_hybrid_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4):
    """Train hybrid model (VGG or ResNet with tabular data)"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("🚂 Training for {} epochs...".format(epochs))
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, features, targets in train_loader:
            images = images.to(device)
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, features, targets in val_loader:
                images = images.to(device)
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(images, features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")
    
    return model


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, test_loader, device, target_scaler, is_ml_model=False):
    """
    Evaluate model and compute all metrics
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: torch.device
        target_scaler: Scaler for inverse transform
        is_ml_model: True for sklearn models, False for PyTorch models
    """
    all_preds = []
    all_targets = []
    
    if is_ml_model:
        # For sklearn models
        for images, features, targets in test_loader:
            # ML models use only tabular features
            # Handle both numpy arrays and torch tensors
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
            
            preds = model.predict(features)
            all_preds.extend(preds)
            all_targets.extend(targets.ravel())
    else:
        # For PyTorch models
        model.eval()
        model = model.to(device)
        
        with torch.no_grad():
            for images, features, targets in test_loader:
                images = images.to(device)
                features = features.to(device)
                
                outputs = model(images, features)
                all_preds.extend(outputs.cpu().numpy().ravel())
                all_targets.extend(targets.numpy().ravel())
    
    # Convert to numpy arrays
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    # Inverse transform
    preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    targets = target_scaler.inverse_transform(targets.reshape(-1, 1)).ravel()
    
    # Ensure non-negative predictions
    preds = np.maximum(preds, 0)
    
    # Compute metrics
    r2 = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    
    # MAPE (avoid division by zero)
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100
    
    # WAPE
    wape = np.sum(np.abs(targets - preds)) / np.sum(np.abs(targets)) * 100
    
    # sMAPE
    smape = np.mean(200 * np.abs(preds - targets) / (np.abs(preds) + np.abs(targets) + 1e-8))
    
    # Within tolerance
    pct_diff = np.abs((targets - preds) / (targets + 1e-8)) * 100
    within_5 = np.mean(pct_diff <= 5) * 100
    within_10 = np.mean(pct_diff <= 10) * 100
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'wape': wape,
        'smape': smape,
        'within_5': within_5,
        'within_10': within_10
    }


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_linear_regression(X_train, y_train, X_test, y_test, target_scaler):
    """Train Linear Regression baseline"""
    print("\n" + "="*80)
    print("📊 MODEL 1: Linear Regression")
    print("="*80)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Create dummy dataset for evaluation
    class DummyDataset:
        def __init__(self, X, y):
            self.X = X
            self.y = y
        def __iter__(self):
            for i in range(len(self.X)):
                yield None, self.X[i:i+1], self.y[i:i+1].reshape(-1, 1)
        def __len__(self):
            return len(self.X)
    
    test_dataset = DummyDataset(X_test, y_test)
    metrics = evaluate_model(model, test_dataset, None, target_scaler, is_ml_model=True)
    
    print("📊 Test Results:")
    print(f"   R² = {metrics['r2']:.4f}")
    print(f"   RMSE = {metrics['rmse']:.2f} kWh")
    print(f"   MAPE = {metrics['mape']:.2f}%")
    print(f"   WAPE = {metrics['wape']:.2f}%")
    
    return metrics


def train_svm(X_train, y_train, X_test, y_test, target_scaler):
    """Train SVM baseline"""
    print("\n" + "="*80)
    print("🎯 MODEL 2: Support Vector Machine")
    print("="*80)
    
    model = SVR(kernel='rbf', C=10.0, gamma='scale')
    print("🚂 Training SVM...")
    model.fit(X_train, y_train)
    
    class DummyDataset:
        def __init__(self, X, y):
            self.X = X
            self.y = y
        def __iter__(self):
            for i in range(len(self.X)):
                yield None, self.X[i:i+1], self.y[i:i+1].reshape(-1, 1)
        def __len__(self):
            return len(self.X)
    
    test_dataset = DummyDataset(X_test, y_test)
    metrics = evaluate_model(model, test_dataset, None, target_scaler, is_ml_model=True)
    
    print("📊 Test Results:")
    print(f"   R² = {metrics['r2']:.4f}")
    print(f"   RMSE = {metrics['rmse']:.2f} kWh")
    print(f"   MAPE = {metrics['mape']:.2f}%")
    print(f"   WAPE = {metrics['wape']:.2f}%")
    
    return metrics


def train_efficientnet_b6_hybrid(train_loader, val_loader, test_loader, device, target_scaler):
    """Train EfficientNet-B6 + Tabular Hybrid Model"""
    print("\n" + "="*80)
    print("⚡ MODEL 3: EfficientNet-B6 + Tabular (43M params)")
    print("="*80)
    
    model = EfficientNetB6Hybrid(num_tabular_features=6, pretrained=True)
    print("🔧 Initialized: EfficientNet-B6 + Tabular (pretrained, 43M params)")
    
    model = train_hybrid_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4)
    metrics = evaluate_model(model, test_loader, device, target_scaler, is_ml_model=False)
    
    print("📊 Test Results:")
    print(f"   R² = {metrics['r2']:.4f}")
    print(f"   RMSE = {metrics['rmse']:.2f} kWh")
    print(f"   MAPE = {metrics['mape']:.2f}%")
    print(f"   WAPE = {metrics['wape']:.2f}%")
    
    return metrics


def train_resnet101_hybrid(train_loader, val_loader, test_loader, device, target_scaler):
    """Train ResNet-101 + Tabular Hybrid Model"""
    print("\n" + "="*80)
    print("🏗️  MODEL 4: ResNet-101 + Tabular (44.5M params)")
    print("="*80)
    
    model = ResNet101Hybrid(num_tabular_features=6, pretrained=True)
    print("🔧 Initialized: ResNet-101 + Tabular (pretrained, 44.5M params)")
    
    model = train_hybrid_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4)
    metrics = evaluate_model(model, test_loader, device, target_scaler, is_ml_model=False)
    
    print("📊 Test Results:")
    print(f"   R² = {metrics['r2']:.4f}")
    print(f"   RMSE = {metrics['rmse']:.2f} kWh")
    print(f"   MAPE = {metrics['mape']:.2f}%")
    print(f"   WAPE = {metrics['wape']:.2f}%")
    
    return metrics


# ============================================================================
# MAIN COMPARISON FUNCTION
# ============================================================================

def run_all_comparisons(csv_path, image_dir):
    """
    Run all comparison models and return results
    """
    print("="*80)
    print("🔬 RUNNING COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # Load data
    images, features, targets, years = load_single_frame_data(csv_path, image_dir)
    
    # Split data (80/10/10)
    unique_years = np.unique(years)
    train_years = unique_years[:int(0.8 * len(unique_years))]
    val_years = unique_years[int(0.8 * len(unique_years)):int(0.9 * len(unique_years))]
    test_years = unique_years[int(0.9 * len(unique_years)):]
    
    train_mask = np.isin(years, train_years)
    val_mask = np.isin(years, val_years)
    test_mask = np.isin(years, test_years)
    
    X_train_img, X_train_feat, y_train = images[train_mask], features[train_mask], targets[train_mask]
    X_val_img, X_val_feat, y_val = images[val_mask], features[val_mask], targets[val_mask]
    X_test_img, X_test_feat, y_test = images[test_mask], features[test_mask], targets[test_mask]
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train_img)} samples")
    print(f"   Val: {len(X_val_img)} samples")
    print(f"   Test: {len(X_test_img)} samples")
    
    # Scale features and targets
    feat_scaler = RobustScaler()
    X_train_feat = feat_scaler.fit_transform(X_train_feat)
    X_val_feat = feat_scaler.transform(X_val_feat)
    X_test_feat = feat_scaler.transform(X_test_feat)
    
    target_scaler = RobustScaler()
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test = target_scaler.transform(y_test.reshape(-1, 1)).ravel()
    
    # Create datasets and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = EnergyDataset(X_train_img, X_train_feat, y_train, transform)
    val_dataset = EnergyDataset(X_val_img, X_val_feat, y_val, transform)
    test_dataset = EnergyDataset(X_test_img, X_test_feat, y_test, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Using device: {device}")
    
    # Run all models
    results = {}
    
    # 1. Linear Regression
    results['Linear Regression'] = train_linear_regression(
        X_train_feat, y_train, X_test_feat, y_test, target_scaler
    )
    
    # 2. SVM
    results['SVM'] = train_svm(
        X_train_feat, y_train, X_test_feat, y_test, target_scaler
    )
    
    # 3. EfficientNet-B6 + Tabular (43M params)
    results['EfficientNet-B6 + Tabular'] = train_efficientnet_b6_hybrid(
        train_loader, val_loader, test_loader, device, target_scaler
    )
    
    # 4. ResNet-101 + Tabular (44.5M params)
    results['ResNet-101 + Tabular'] = train_resnet101_hybrid(
        train_loader, val_loader, test_loader, device, target_scaler
    )
    
    print("\n" + "="*80)
    print("✅ ALL COMPARISONS COMPLETE!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Example usage - UPDATE THESE PATHS TO YOUR ACTUAL DATA LOCATIONS
    import sys
    
    if len(sys.argv) > 2:
        # Allow command line arguments: python comparison_models.py <csv_path> <image_dir>
        csv_path = sys.argv[1]
        image_dir = sys.argv[2]
    else:
        # Default paths - CHANGE THESE TO YOUR ACTUAL PATHS
        csv_path = r'C:\Users\FA004\Desktop\satimg2\data.csv'  # Update this path
        image_dir = r'C:\Users\FA004\Desktop\satimg2\images_png_view'  # Update this path
        
        print("⚠️  Using default paths. To specify custom paths, use:")
        print(f"   python {sys.argv[0]} <csv_path> <image_dir>")
        print(f"\n📂 Current paths:")
        print(f"   CSV: {csv_path}")
        print(f"   Images: {image_dir}\n")
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: CSV file not found at: {csv_path}")
        print("   Please update the csv_path variable in the script or provide it as a command line argument.")
        sys.exit(1)
    
    if not os.path.exists(image_dir):
        print(f"❌ ERROR: Image directory not found at: {image_dir}")
        print("   Please update the image_dir variable in the script or provide it as a command line argument.")
        sys.exit(1)
    
    results = run_all_comparisons(csv_path, image_dir)
    
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON SUMMARY")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"   R² = {metrics['r2']:.4f}")
        print(f"   RMSE = {metrics['rmse']:.2f} kWh")
        print(f"   MAPE = {metrics['mape']:.2f}%")
        print(f"   WAPE = {metrics['wape']:.2f}%")
