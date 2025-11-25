# 🔍 ROOT CAUSE ANALYSIS - Trading Bot AI Performance Issues

**Date**: 2025-11-25
**Status**: CRITICAL - Models Not Learning
**Analyst**: Claude AI

---

## 📋 EXECUTIVE SUMMARY

The AI Trading Bot is experiencing **complete model failure** with balanced accuracy at 33.3% (random guessing level) and DQL agent showing negative rewards. This analysis identifies 8 critical root causes and provides actionable solutions.

---

## 🚨 CRITICAL ISSUES IDENTIFIED

### 1. **SINGLE-CLASS PREDICTION COLLAPSE** ⚠️

**Symptoms**:
- LSTM: Predicts only UP class (accuracy: [0.0, 0.0, 1.0])
- GRU: Predicts only NEUTRAL class (accuracy: [0.0, 1.0, 0.0])
- Balanced accuracy locked at 33.3% (random guessing)

**Root Causes**:
```
a) Focal Loss Implementation Bug
   - Class weights calculated AFTER SMOTE
   - Should calculate BEFORE SMOTE for proper weighting
   - Alpha weights might be inverted (punishing majority instead of minority)

b) OneCycleLR Scheduler Issue
   - Scheduler steps called incorrectly
   - scheduler.step(val_balanced_acc) is WRONG for OneCycleLR
   - OneCycleLR requires step() after every batch, not epoch
   - Current implementation breaks learning rate schedule

c) Insufficient Model Capacity
   - hidden_size=96 too small for 77-dimensional input
   - Sequence length=60 might be too long (vanishing gradients)
   - No batch normalization or layer normalization

d) Class Imbalance Still Severe
   - Down: 24.1%, Neutral: 47.7%, Up: 28.0%
   - SMOTE not applied (library was missing)
   - Adaptive threshold (0.0667%) too conservative
```

**Impact**: **CRITICAL** - Models cannot learn any meaningful patterns

---

### 2. **DATA LABELING ISSUES** ⚠️

**Symptoms**:
- Adaptive threshold = 0.0667% (only $0.67 movement per $1000)
- 47.7% samples labeled as NEUTRAL (imbalanced)
- Labels might not align with profitable trading patterns

**Root Causes**:
```
a) Threshold Too Conservative
   - 30% of ATR is too small for intraday trading
   - Should use 50-80% of ATR for meaningful price moves
   - Current: 0.0667% ≈ 1.3 points on XAUUSD (noise level)

b) Single-Timeframe Labeling
   - Only looking 1 bar ahead (myopic)
   - Should consider multi-bar forward returns
   - Missing trend context (up/down trend vs ranging)

c) Binary Classification Mismatch
   - 3-class (up/down/neutral) difficult to learn
   - Neutral class = "I don't know" but model forced to trade
   - Should use 2-class (up/down) with confidence thresholding

d) No Profit-Alignment
   - Labels based on price direction only
   - Doesn't consider transaction costs
   - 0.1% move might be directionally up but unprofitable
```

**Impact**: **HIGH** - Garbage labels = garbage predictions

---

### 3. **CLASS IMBALANCE HANDLING FAILURE** ⚠️

**Symptoms**:
- SMOTE not installed initially
- Class distribution still 24%/48%/28% after "fixes"
- Focal Loss not preventing collapse

**Root Causes**:
```
a) SMOTE Was Not Installed
   - Library missing until just now
   - Training proceeded without resampling
   - Minority classes underrepresented in batches

b) Focal Loss Class Weights Bug
   - Weights calculated AFTER SMOTE (wrong!)
   - Should calculate on ORIGINAL imbalance
   - Current implementation:
     unique, counts = np.unique(y_train, return_counts=True)  # After SMOTE
     class_weights = [total / (n_classes * count) for count in counts]
   - After SMOTE, classes are balanced, so weights ≈ [1, 1, 1] (useless!)

c) No Balanced Batch Sampling
   - DataLoader uses random shuffle
   - Each batch might have 0 minority class samples
   - Should use WeightedRandomSampler

d) Validation/Test Split Not Stratified
   - train_test_split(shuffle=False) in preprocessor
   - Test set might have different distribution
   - Should use stratified split
```

**Impact**: **CRITICAL** - Model biased toward majority class

---

### 4. **DQL AGENT COMPLETE FAILURE** ⚠️

**Symptoms**:
- All metrics = 0.0
- Average reward = -3412 (massively negative)
- Win rate = 0%
- Sharpe ratio = 0

**Root Causes**:
```
a) Reward Scaling Issue
   - Current: reward = (equity_change / init_balance) * 1000
   - equity_change is typically 0.001-0.01 (0.1-1%)
   - reward ≈ 1-10 per step
   - But fee_penalty = trade_cost * init_balance * 10
   - If init_balance = 10000, trade_cost = 0.0001
   - fee_penalty = 0.0001 * 10000 * 10 = 10
   - Net reward ≈ 1 - 10 = -9 (always negative!)

b) Excessive Fee Penalization
   - fee_penalty multiplied by 10 is too harsh
   - Discourages ALL trading
   - Agent learns "never trade" to minimize penalty

c) Risk Penalty Calculation Bug
   - risk_penalty = 0.01 * (position * unrealized_pnl) / init_balance
   - If unrealized_pnl is negative, penalty is NEGATIVE (reward!)
   - Should use abs(unrealized_pnl)

d) State Space Too Large
   - State size = 140 dims (13 features × 10 bars + 3 account)
   - DQN struggles with high-dimensional continuous states
   - Should reduce to 50-80 dims

e) Epsilon Decay Too Slow
   - epsilon_decay_steps = 20,000
   - With 100 episodes × max_steps
   - Epsilon barely decays during training
   - Agent stuck in random exploration
```

**Impact**: **CRITICAL** - DQL agent learns nothing, accumulates losses

---

### 5. **TRAINING PROCESS ISSUES** ⚠️

**Symptoms**:
- Early stopping at epoch 21-23 (too early)
- No improvement in validation accuracy
- Loss decreases but accuracy doesn't improve

**Root Causes**:
```
a) OneCycleLR Scheduler Misuse
   - Line 369: self.scheduler.step(val_balanced_acc)
   - OneCycleLR doesn't accept metric argument
   - Should call scheduler.step() after EACH BATCH
   - Current: called once per epoch with wrong argument

b) Early Stopping Too Aggressive
   - patience = 20 epochs
   - min_delta = 0.005 (0.5% improvement required)
   - For 33% balanced accuracy, need to reach 33.5% to count
   - Random fluctuations cause premature stopping

c) Batch Size vs Learning Rate Mismatch
   - batch_size = 64
   - learning_rate = 5e-4
   - For batch_size 64, typical LR = 1e-3 to 2e-3
   - Learning too slow

d) No Warmup
   - OneCycleLR has 10% warmup (pct_start=0.1)
   - But scheduler not called properly
   - Warmup not happening

e) Dropout Too Early
   - Dropout = 0.2 from epoch 1
   - Should start without dropout, add gradually
   - Or use lower dropout (0.1) initially
```

**Impact**: **HIGH** - Model stops training before learning

---

### 6. **MODEL ARCHITECTURE WEAKNESSES** ⚠️

**Symptoms**:
- Models converge to single prediction quickly
- No diversity in predictions across validation set

**Root Causes**:
```
a) No Normalization Layers
   - Financial data has varying scales (price, volume, indicators)
   - No BatchNorm or LayerNorm
   - Gradients unstable

b) Hidden Size Too Small
   - hidden_size = 96 for 77 input features
   - Rule of thumb: hidden_size ≥ input_size
   - Should be 128-256

c) No Residual Connections
   - 2-layer BiLSTM without skip connections
   - Gradient flow issues
   - Should add residual paths

d) Attention Mechanism Weak
   - Simple attention (Linear → Tanh → Linear)
   - Modern: Multi-head attention or scaled dot-product
   - Current attention might not capture temporal dependencies

e) Classification Head Too Deep
   - 3 layers: hidden*2 → hidden → hidden/2 → classes
   - Adds unnecessary complexity
   - Should be: hidden*2 → dropout → classes (1-2 layers max)
```

**Impact**: **MEDIUM** - Architecture limits learning capacity

---

### 7. **DATA QUANTITY ISSUES** ⚠️

**Symptoms**:
- Total: 8,560 samples
- After split (70/15/15): Train=5,992, Val=1,284, Test=1,284
- After sequence creation (window=60): ~5,900 sequences

**Analysis**:
```
a) Is 8,560 Enough for Deep Learning?
   - For simple LSTM: Minimum 5,000-10,000 samples ✓
   - For complex model (BiLSTM + Attention): 10,000-50,000 ideal
   - Current: BORDERLINE

b) Is 5,900 Sequences Enough?
   - Parameters in DirectionalLSTM:
     - Input: 77 features
     - Hidden: 96 × 2 (bidirectional) = 192
     - LSTM params: 4 × (77×192 + 192×192 + 192) ≈ 210,000
     - Attention: ~20,000 params
     - Classifier: ~50,000 params
     - Total: ~280,000 parameters
   - Rule: Need 10× samples vs parameters
   - Required: 2,800,000 samples
   - Actual: 5,900 samples
   - **SEVERELY UNDERFITTED** (476× shortfall!)

c) Overfitting Risk
   - With 280K params and 5.9K samples
   - Model will memorize, not generalize
   - Explains single-class prediction (overfitting to majority)
```

**Impact**: **CRITICAL** - Insufficient data for model complexity

---

### 8. **EVALUATION METRICS INSUFFICIENT** ⚠️

**Symptoms**:
- Only logging accuracy and balanced accuracy
- No per-class precision/recall during training
- No confusion matrix
- No ROC-AUC or calibration metrics

**Root Causes**:
```
a) Missing Per-Class Metrics
   - Can't see which class is being ignored
   - Should log: precision, recall, F1 per class every epoch

b) No Confusion Matrix Logging
   - Essential for diagnosing single-class prediction
   - Should log confusion matrix every 10 epochs

c) No Probability Calibration
   - Focal Loss might make probabilities overconfident
   - Should check calibration (reliability diagram)

d) No Cross-Validation
   - Single train/val/test split
   - Results might be lucky/unlucky split
   - Should use 5-fold CV for small dataset
```

**Impact**: **LOW** - Makes debugging harder but not cause of failure

---

## 🎯 PRIORITY RANKING

| Priority | Issue | Impact | Effort | Fix Priority |
|----------|-------|--------|--------|--------------|
| 1 | Focal Loss class weights bug | CRITICAL | LOW | **URGENT** |
| 2 | OneCycleLR scheduler misuse | CRITICAL | LOW | **URGENT** |
| 3 | SMOTE not applied | CRITICAL | LOW | **URGENT** |
| 4 | DQL reward calculation bug | CRITICAL | MEDIUM | **HIGH** |
| 5 | Adaptive threshold too low | HIGH | LOW | **HIGH** |
| 6 | Model too complex for data size | CRITICAL | MEDIUM | **HIGH** |
| 7 | No balanced batch sampling | HIGH | LOW | **MEDIUM** |
| 8 | No batch normalization | MEDIUM | LOW | **MEDIUM** |
| 9 | Early stopping too aggressive | MEDIUM | LOW | **MEDIUM** |
| 10 | Insufficient evaluation metrics | LOW | MEDIUM | **LOW** |

---

## ✅ RECOMMENDED FIX SEQUENCE

### Phase 1: Critical Fixes (TODAY)
1. Fix Focal Loss class weights (calculate BEFORE SMOTE)
2. Fix OneCycleLR scheduler (call after each batch)
3. Enable SMOTE with installed library
4. Increase adaptive threshold to 0.3-0.5% (50-80% of ATR)
5. Fix DQL reward calculation (reduce fee penalty multiplier)

### Phase 2: Architecture Optimization (DAY 2)
6. Reduce model complexity (hidden_size=128, simple classifier)
7. Add batch normalization layers
8. Add balanced batch sampler
9. Stratified train/val/test split
10. Add per-class metrics logging

### Phase 3: Advanced Improvements (DAY 3+)
11. Multi-timeframe labeling
12. 2-class instead of 3-class classification
13. Data augmentation (time series jittering)
14. Cross-validation
15. Hyperparameter optimization with Optuna

---

## 📊 EXPECTED IMPROVEMENTS

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| Balanced Accuracy | 33% | 45-50% | 52-58% | 55-62% |
| Per-class Accuracy | [0,0,1] | [0.35,0.45,0.55] | [0.45,0.50,0.60] | [0.50,0.55,0.65] |
| DQL Avg Reward | -3412 | -100 | +50 | +200 |
| Win Rate | 0% | 40-45% | 48-52% | 52-58% |

---

## 🔧 SPECIFIC CODE FIXES REQUIRED

### Fix 1: Focal Loss Class Weights
```python
# BEFORE (WRONG):
# Calculate class weights AFTER SMOTE
unique, counts = np.unique(y_train, return_counts=True)
class_weights = [total / (len(unique) * count) for count in counts]

# AFTER (CORRECT):
# Calculate class weights BEFORE SMOTE
unique_original, counts_original = np.unique(y_train, return_counts=True)
original_class_weights = [len(y_train) / (len(unique_original) * count)
                          for count in counts_original]

# Apply SMOTE
if use_smote:
    X_train, y_train = smote.fit_resample(X_train_flat, y_train)

# Use ORIGINAL weights for Focal Loss
self.criterion = FocalLoss(alpha=original_class_weights, gamma=2.0)
```

### Fix 2: OneCycleLR Scheduler
```python
# BEFORE (WRONG):
self.scheduler.step(val_balanced_acc)  # Wrong! OneCycleLR doesn't take metric

# AFTER (CORRECT):
# In training loop, after each batch:
for batch_X, batch_y in train_loader:
    loss.backward()
    optimizer.step()
    scheduler.step()  # Step after every batch, no argument
```

### Fix 3: Adaptive Threshold
```python
# BEFORE (WRONG):
threshold = (atr_value / median_close) * 0.3  # Too conservative

# AFTER (CORRECT):
threshold = (atr_value / median_close) * 0.7  # 70% of ATR
```

### Fix 4: DQL Reward
```python
# BEFORE (WRONG):
reward -= fee_penalty * 10  # Too harsh

# AFTER (CORRECT):
reward -= fee_penalty * 2  # Moderate penalty
```

---

## 📈 ALTERNATIVE APPROACHES

If deep learning continues to underperform with 8,560 samples:

### Option A: Simpler Models
- Use traditional ML: XGBoost, Random Forest, SVM
- Requires fewer samples (1,000-5,000)
- Often outperform DL on small datasets

### Option B: Transfer Learning
- Pre-train on larger financial dataset (Yahoo Finance, etc.)
- Fine-tune on your XAUUSD data
- Leverage learned patterns from related assets

### Option C: Ensemble Simple + DL
- Combine RandomForest (handles small data) + LSTM (captures sequences)
- Weighted voting or stacking
- More robust than pure DL

### Option D: Feature-Based RL Instead of DQL
- Use engineered features (20-30 dims) instead of raw sequences
- PPO or A3C instead of DQL
- Smaller state space, faster learning

---

## 🎓 DATA REQUIREMENTS RECOMMENDATION

For current BiLSTM architecture (280K parameters):
- **Minimum**: 50,000 samples (180× current)
- **Recommended**: 100,000 samples (360× current)
- **Ideal**: 500,000 samples (1,800× current)

**How to get more data:**
1. Use 5-minute bars instead of 1-hour (12× more data)
2. Train on multiple symbols (XAUUSD, EURUSD, GBPUSD) together
3. Use 5+ years of history instead of 1 year
4. Data augmentation (jittering, scaling) to synthetically 3-5× data

**Or simplify model:**
- Reduce hidden_size: 96 → 64 (params: 280K → 140K)
- Remove bidirectional: (params: 140K → 70K)
- Single LSTM layer: (params: 70K → 40K)
- **Target: 40K params requires 400K samples → still 70× short**
- **Better: Use RandomForest (5K params, handles current data size)**

---

## ✍️ CONCLUSION

The bot's failure is **NOT due to a single bug**, but a **confluence of 8 critical issues**:
1. Focal Loss implementation bug (class weights calculated wrongly)
2. LR scheduler misuse (breaking learning rate schedule)
3. SMOTE not installed/applied
4. DQL reward calculation bugs
5. Adaptive threshold too conservative
6. **Model too complex for available data (476× parameter-to-sample ratio!)**
7. Training process issues
8. Insufficient monitoring

**The #1 bottleneck: INSUFFICIENT DATA FOR MODEL COMPLEXITY**

**Recommended immediate action:**
1. Apply Phase 1 fixes (5 critical bugs)
2. Reduce model complexity OR get 50× more data
3. Consider simpler ML models (RandomForest, XGBoost) as baseline
4. If staying with DL, use transfer learning or multi-asset training

---

**Generated**: 2025-11-25
**Next Review**: After Phase 1 fixes implemented
