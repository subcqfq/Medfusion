# -*- coding: utf-8 -*-
# ===== (1) Environment-level reproducibility settings: before importing torch =====
import os
import random

SEED = 42  # You can change this to any other fixed number
os.environ["PYTHONHASHSEED"] = str(SEED)
# Enable determinism for some cuBLAS operations (CUDA >= 10.2; must be set before first CUDA use)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# ===== (2) General imports =====
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ===== (3) Unified global seed-setting function =====
def set_global_seed(seed: int = 20240518):
    # Python / NumPy / PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN / TF32: avoid nondeterminism or numerical drift
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    # Disable TF32 (some GPUs/drivers enable it by default and it may cause subtle differences)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    # Strictly use deterministic algorithms (will raise if an op is unsupported; use warn_only=True to only warn)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ===== (4) Device selection (doesn't affect determinism, but affects speed) =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ===== (5) Data loading and preprocessing =====
# - For other prediction tasks, switch files to:
#   mimicmodstrain.csv (MODS), mimichxtrain.csv (hypoxemia), mimichstrain.csv (hemorrhagic shock).
def load_and_preprocess_data():
    train_df = pd.read_csv('mimicmodstrain.csv')
    test_df  = pd.read_csv('mimicmodstest.csv')

    print(f"Train set size: {train_df.shape}")
    print(f"Test set size: {test_df.shape}")
    print(f"Train positive rate: {train_df['mods'].mean():.4f}")
    print(f"Test positive rate: {test_df['mods'].mean():.4f}")

    feature_columns = [c for c in train_df.columns if c != 'mods']

    X_train = train_df[feature_columns].values
    y_train = train_df['mods'].values.astype(np.float32)
    X_test  = test_df[feature_columns].values
    y_test  = test_df['mods'].values.astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Split out a validation set from the training set (avoid leakage from using the test set as validation)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    X_tr  = torch.tensor(X_tr,  dtype=torch.float32)
    y_tr  = torch.tensor(y_tr,  dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_te  = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_te  = torch.tensor(y_test, dtype=torch.float32)

    return X_tr, y_tr, X_val, y_val, X_te, y_te, len(feature_columns)


# ===== (6) Model definitions =====
# 6.1 MLP
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]  # No Sigmoid; use BCEWithLogitsLoss
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# 6.2 GRU + Attention
class GRUAttnModel(nn.Module):
    """
    Treat each numerical feature as a "token":
    token = value_proj(value) + feature_embedding[feature_id]
    Then feed into a BiGRU, use learnable attention for weighted pooling, and finally classify.
    """
    def __init__(self, input_dim, emb_dim=64, hidden_dim=128, num_layers=2,
                 bidirectional=True, dropout=0.3, attn_hidden=128):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Assign a learnable embedding to each "feature position"
        self.feature_emb = nn.Embedding(input_dim, emb_dim)
        # Scalar value -> vector
        self.val_proj = nn.Linear(1, emb_dim)
        self.token_ln = nn.LayerNorm(emb_dim)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Attention (additive / HAN-style)
        self.attn_fc = nn.Linear(hidden_dim * self.num_directions, attn_hidden)
        self.attn_tanh = nn.Tanh()
        self.attn_vec = nn.Linear(attn_hidden, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # No Sigmoid; pair with BCEWithLogitsLoss
        )

        # Initialization (controlled by global seed)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

    def forward(self, x):
        # x: (batch, seq_len)  Treat each feature as a time step
        B, S = x.shape
        feat_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)  # (B, S)

        value_tokens = self.val_proj(x.unsqueeze(-1))            # (B, S, emb_dim)
        feat_tokens  = self.feature_emb(feat_ids)                # (B, S, emb_dim)
        tokens = self.token_ln(value_tokens + feat_tokens)       # (B, S, emb_dim)
        tokens = self.dropout(tokens)

        H, _ = self.gru(tokens)  # (B, S, hidden*dirs)

        # Attention weights
        U = self.attn_tanh(self.attn_fc(H))          # (B, S, attn_hidden)
        scores = self.attn_vec(U).squeeze(-1)        # (B, S)
        alpha = torch.softmax(scores, dim=1)         # (B, S)
        context = torch.sum(H * alpha.unsqueeze(-1), dim=1)  # (B, hidden*dirs)

        context = self.dropout(context)
        logits = self.classifier(context).squeeze(-1)  # (B,)
        return logits  # logits (pre-sigmoid)


# 6.3 Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(input_dim, d_model))  # controlled by global seed
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)  # no Sigmoid
        )

    def forward(self, x):
        B, S = x.size(0), x.size(1)
        x = self.input_projection(x.unsqueeze(-1))              # (B, S, d_model)
        x = x + self.pos_encoding.unsqueeze(0).expand(B, -1, -1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)


# ===== (7) Training and evaluation =====
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, pos_weight=None):
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False, min_lr=1e-6
    )

    model.to(device)
    best_auc, patience_counter, patience = 0.0, 0, 30
    train_losses, val_aucs = [], []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.detach().cpu().numpy())
                val_labels.extend(yb.detach().cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(val_auc)

        train_losses.append(running / max(1, len(train_loader)))
        val_aucs.append(val_auc)

        improved = val_auc > best_auc + 1e-6
        if improved:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val AUC: {val_auc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    return model, best_auc, train_losses, val_aucs


def evaluate_model(model, test_loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy())
            labels.extend(yb.numpy())
    auc = roc_auc_score(labels, preds)
    return auc, preds, labels


# ===== (8) Main pipeline =====
def main():
    # Fix random seed before any stochastic operations
    set_global_seed(SEED)

    X_tr, y_tr, X_val, y_val, X_te, y_te, input_dim = load_and_preprocess_data()

    # Compute positive-class weight
    pos_weight = (y_tr == 0).sum().item() / max(1, (y_tr == 1).sum().item())
    print(f"Positive-class weight pos_weight = {pos_weight:.2f}")

    # Sampler: make each batch more balanced (reproducible: use a fixed generator)
    y_tr_np = y_tr.numpy().astype(int)
    class_count = np.bincount(y_tr_np)
    class_weight = 1.0 / np.maximum(class_count, 1)
    sample_weight = class_weight[y_tr_np]

    # Unified CPU random generator (shared by sampler & dataloader)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(SEED)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weight, dtype=torch.double),
        num_samples=len(sample_weight),
        replacement=True,
        generator=g_cpu
    )

    batch_size = 128
    # For strongest reproducibility: num_workers=0 and pass the same generator
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=0,
        generator=g_cpu,
        persistent_workers=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=g_cpu,
        persistent_workers=False
    )
    test_loader = DataLoader(
        TensorDataset(X_te, y_te),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=g_cpu,
        persistent_workers=False
    )

    # Instantiate models after setting randomness (parameter init is controlled by the global seed)
    models_config = {
        'MLP': MLPModel(input_dim, hidden_dims=[512, 256, 128, 64, 32, 16], dropout=0.3),
        'GRU': GRUAttnModel(input_dim, emb_dim=64, hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.3),
        'Transformer': TransformerModel(input_dim, d_model=32, nhead=8, num_layers=2, dropout=0.3)
    }

    results = {}
    for name, model in models_config.items():
        print("\n" + "="*50)
        print(f"Training {name} model")
        print("="*50)
        trained, best_val_auc, train_losses, val_aucs = train_model(
            model, train_loader, val_loader, epochs=120, lr=1e-3, pos_weight=pos_weight
        )
        test_auc, test_preds, test_labels = evaluate_model(trained, test_loader)
        pd.DataFrame({"y_proba": test_preds}).to_csv(f"mods_{name}_proba.csv", index=False)

        y_hat_bin = (np.array(test_preds) > 0.5).astype(int)
        print(f"\n{name} results:")
        print(f"Best validation AUC: {best_val_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print("\nClassification report:")
        print(classification_report(test_labels, y_hat_bin, digits=4))

        results[name] = {
            'model': trained, 'val_auc': best_val_auc, 'test_auc': test_auc,
            'train_losses': train_losses, 'val_aucs': val_aucs, 'predictions': test_preds
        }

    # Summary
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_auc'])
    best_auc = results[best_model_name]['test_auc']

    print("\n" + "="*60)
    print("Final results summary")
    print("="*60)
    for name in results:
        print(f"{name:12} - Val AUC: {results[name]['val_auc']:.4f}, Test AUC: {results[name]['test_auc']:.4f}")
    print(f"\n🏆 Best model: {best_model_name} (AUC = {best_auc:.4f})")

    # Visualization
    plt.figure(figsize=(15,5))
    # Training loss
    plt.subplot(1,3,1)
    for name in results:
        plt.plot(results[name]['train_losses'], label=name)
    plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()

    # Validation AUC
    plt.subplot(1,3,2)
    for name in results:
        plt.plot(results[name]['val_aucs'], label=name)
    plt.title('Validation AUC'); plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.grid(True); plt.legend()

    # Test AUC bar chart
    plt.subplot(1,3,3)
    names = list(results.keys())
    test_aucs = [results[n]['test_auc'] for n in names]
    bars = plt.bar(names, test_aucs)
    for bar, auc in zip(bars, test_aucs):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{auc:.4f}", ha='center', va='bottom')
    plt.ylim(0,1); plt.title('Test-set AUC'); plt.ylabel('AUC')

    plt.tight_layout(); plt.show()
    return results, best_model_name


if __name__ == "__main__":
    results, best_model = main()
