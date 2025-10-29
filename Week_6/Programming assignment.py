import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------
# 1) 讀資料與前處理
# -------------------------
path = "/content/classification_3x3.csv"  # 若路徑不同請修改
df = pd.read_csv(path)

# features and label
feature_cols = [f"f{i}" for i in range(9)] + ["lon", "lat"]
X = df[feature_cols].replace(-999.0, np.nan)
X = X.fillna(X.median())   # 中位數補值（和你之前一致）
y = df["label"].values

print("Data shape:", X.shape, "Labels:", np.bincount(y.astype(int)))

# split
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------
# 2) GDA implementation
# -------------------------
def fit_gda(X, y):
    """
    Fit Gaussian Discriminant Analysis assuming shared covariance.
    Returns: phi, mu0, mu1, Sigma (pooled), Sigma_inv, logdet
    """
    n, d = X.shape
    y = y.astype(int)
    phi = y.mean()  # P(y=1)
    X1 = X[y == 1]
    X0 = X[y == 0]
    mu1 = X1.mean(axis=0)
    mu0 = X0.mean(axis=0)
    # pooled covariance
    # unbiased pooled: sum (xi - mu_y)(xi - mu_y).T / n
    S1 = ((X1 - mu1).T @ (X1 - mu1))
    S0 = ((X0 - mu0).T @ (X0 - mu0))
    Sigma = (S1 + S0) / n  # divide by n (MLE for GDA uses n in denom)
    # for numerical stability, add small regularization
    eps = 1e-6
    Sigma += np.eye(d) * eps
    Sigma_inv = np.linalg.inv(Sigma)
    sign, logdet = np.linalg.slogdet(Sigma)
    return {"phi": phi, "mu0": mu0, "mu1": mu1, "Sigma": Sigma, "Sigma_inv": Sigma_inv, "logdet": logdet}

def log_gauss_pdf(X, mu, Sigma_inv, logdet):
    """
    Compute log N(x | mu, Sigma) up to constant:
    log p = -0.5 * ( (x-mu)^T Sigma_inv (x-mu) ) - 0.5 * logdet - D/2 * log(2π)
    We will return full log p (including constant) for comparisons.
    """
    d = mu.shape[0]
    const = -0.5 * d * np.log(2*np.pi) - 0.5 * logdet
    xm = X - mu
    # compute quadratic form efficiently for all rows
    q = np.einsum('ij,jk,ik->i', xm, Sigma_inv, xm)  # shape (n,)
    return const - 0.5 * q

def predict_gda(model, X):
    """
    Return predicted probabilities P(y=1 | x) and predicted class (0/1)
    """
    phi = model["phi"]
    mu0 = model["mu0"]
    mu1 = model["mu1"]
    Sigma_inv = model["Sigma_inv"]
    logdet = model["logdet"]
    # log p(x|y=1) and log p(x|y=0)
    logp1 = log_gauss_pdf(X, mu1, Sigma_inv, logdet)
    logp0 = log_gauss_pdf(X, mu0, Sigma_inv, logdet)
    # log posterior ratio: log p(y=1|x) / p(y=0|x) = log p(x|1) + log phi - [log p(x|0) + log(1-phi)]
    log_prior_ratio = np.log(phi + 1e-12) - np.log(1-phi + 1e-12)
    log_odds = logp1 - logp0 + log_prior_ratio
    # convert to probability via sigmoid
    probs = 1 / (1 + np.exp(-log_odds))
    preds = (probs >= 0.5).astype(int)
    return probs, preds

# fit on training data
gda_model = fit_gda(X_train, y_train)
probs_test, preds_test = predict_gda(gda_model, X_test)

# -------------------------
# 3) 評估
# -------------------------
acc = accuracy_score(y_test, preds_test)
prec = precision_score(y_test, preds_test)
rec = recall_score(y_test, preds_test)
f1 = f1_score(y_test, preds_test)
cm = confusion_matrix(y_test, preds_test)

print("GDA (full features) evaluation:")
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
print("Confusion Matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, preds_test, digits=4))

# -------------------------
# 4) Decision boundary visualization (PCA -> 2D)
#    We project features to 2D with PCA, then fit a 2D GDA (only for plotting)
# -------------------------
pca = PCA(n_components=2, random_state=42)
X_all = np.vstack([X_train, X_test])
pca.fit(X_all)  # fit on train+test for visualization consistency
X2 = pca.transform(X.values)
X2_train, X2_test = train_test_split(X2, test_size=0.2, random_state=42, stratify=y)

# Fit GDA in 2D (for plotting)
gda_2d = fit_gda(X2_train, y_train)  # uses corresponding y_train (split same random_state as earlier)

# Create grid for contour
x_min, x_max = X2[:,0].min() - 1.0, X2[:,0].max() + 1.0
y_min, y_max = X2[:,1].min() - 1.0, X2[:,1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

# predict on grid with 2D GDA
probs_grid, preds_grid = predict_gda(gda_2d, grid_points)
Z = preds_grid.reshape(xx.shape)

# Plot
plt.figure(figsize=(7,6))
# contour for decision region
plt.contourf(xx, yy, Z, alpha=0.2, levels=[-0.5,0.5,1.5], colors=['#d1e5f0','#fddbc7'])
# scatter the data (PCA 2D)
mask0 = (y==0)
mask1 = (y==1)
plt.scatter(X2[mask0,0], X2[mask0,1], s=10, label='class 0', alpha=0.8, edgecolors='k', linewidth=0.2)
plt.scatter(X2[mask1,0], X2[mask1,1], s=10, label='class 1', alpha=0.8, edgecolors='k', linewidth=0.2)
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.title("GDA decision boundary (visualized on PCA 2D)")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("/content/GDA_decision_boundary.png", dpi=300)
plt.show()

print("Saved visualization to /content/GDA_decision_boundary.png")
