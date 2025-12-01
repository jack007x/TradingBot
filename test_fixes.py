#!/usr/bin/env python3
"""
Quick test for RegressionPredictor delegation methods and Scheduler fix
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from src.models.neural_networks.regression_predictor import RegressionPredictor
from src.learning.experience_buffer import ExperienceBuffer
from src.learning.online_trainer import AdaptiveLearningScheduler

print("=" * 80)
print("🧪 TESTING FIXES")
print("=" * 80)

# ============================================================================
# TEST 1: RegressionPredictor Delegation Methods
# ============================================================================
print("\n📊 TEST 1: RegressionPredictor delegation methods")
print("-" * 80)

try:
    # Create a RegressionPredictor
    predictor = RegressionPredictor(
        input_size=50,
        model_type='lstm',
        hidden_size=64,
        num_layers=2
    )

    print("✅ RegressionPredictor created")

    # Test state_dict()
    state = predictor.state_dict()
    print(f"✅ state_dict() works - got {len(state)} parameter tensors")

    # Test parameters()
    params = list(predictor.parameters())
    print(f"✅ parameters() works - got {len(params)} parameters")

    # Test named_parameters()
    named_params = list(predictor.named_parameters())
    print(f"✅ named_parameters() works - got {len(named_params)} named parameters")

    # Test __call__()
    dummy_input = torch.randn(1, 60, 50)  # (batch, seq_len, features)
    output = predictor(dummy_input)
    print(f"✅ __call__() works - output shape: {output.shape}")

    # Test load_state_dict()
    predictor.load_state_dict(state)
    print(f"✅ load_state_dict() works")

    # Test train/eval
    predictor.eval()
    print(f"✅ eval() works")
    predictor.train_mode()
    print(f"✅ train_mode() works")

    print("\n🎉 All RegressionPredictor delegation methods working!")

except Exception as e:
    print(f"\n❌ RegressionPredictor test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 2: AdaptiveLearningScheduler with 0 experiences
# ============================================================================
print("\n" + "=" * 80)
print("📊 TEST 2: AdaptiveLearningScheduler with 0 experiences")
print("-" * 80)

try:
    # Create empty buffer
    buffer = ExperienceBuffer(max_size=100)

    # Create scheduler
    scheduler = AdaptiveLearningScheduler(
        min_time_between_updates=10,  # 10 seconds
        min_experiences_for_update=50
    )

    print(f"✅ Scheduler created")
    print(f"   Min experiences required: {scheduler.min_experiences_for_update}")
    print(f"   Current experiences: {len(buffer.completed_experiences)}")

    # Test with 0 experiences
    should_update, reason = scheduler.should_update('LSTM', buffer)

    print(f"\n📋 Should update with 0 experiences?")
    print(f"   Result: {should_update}")
    print(f"   Reason: {reason}")

    if not should_update and "Insufficient total experiences" in reason:
        print(f"\n✅ CORRECT! Scheduler properly rejects 0 experiences")
    else:
        print(f"\n❌ WRONG! Scheduler should reject 0 experiences")
        sys.exit(1)

    # Add some experiences (but less than minimum)
    for i in range(25):  # Less than 50
        pred_id = f"test_{i}"
        buffer.add_prediction(pred_id, "XAUUSD", np.random.randn(100), 0.01, 'buy', 0.7, 'LSTM')
        buffer.link_trade_to_prediction(pred_id, 1000 + i, 2000.0)
        buffer.record_outcome(1000 + i, 2010.0, 100.0)

    print(f"\n📋 Should update with 25 experiences (< 50)?")
    should_update, reason = scheduler.should_update('LSTM', buffer)
    print(f"   Result: {should_update}")
    print(f"   Reason: {reason}")

    if not should_update and "Insufficient total experiences (25/50)" in reason:
        print(f"\n✅ CORRECT! Scheduler properly rejects 25 < 50 experiences")
    else:
        print(f"\n❌ WRONG! Expected rejection due to insufficient experiences")
        sys.exit(1)

    # Add more to reach minimum
    for i in range(25, 55):  # Now 55 total (> 50)
        pred_id = f"test_{i}"
        buffer.add_prediction(pred_id, "XAUUSD", np.random.randn(100), 0.01, 'buy', 0.7, 'LSTM')
        buffer.link_trade_to_prediction(pred_id, 1000 + i, 2000.0)
        buffer.record_outcome(1000 + i, 2010.0, 100.0)

    print(f"\n📋 Should update with 55 experiences (> 50)?")
    should_update, reason = scheduler.should_update('LSTM', buffer)
    print(f"   Result: {should_update}")
    print(f"   Reason: {reason}")

    if should_update:
        print(f"\n✅ CORRECT! Scheduler allows update with sufficient experiences")
    else:
        print(f"\n❌ WRONG! Scheduler should allow update with 55 experiences")
        sys.exit(1)

    print("\n🎉 AdaptiveLearningScheduler fix working correctly!")

except Exception as e:
    print(f"\n❌ Scheduler test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\n📊 Summary:")
print("   ✅ RegressionPredictor delegation methods - Working")
print("   ✅ AdaptiveLearningScheduler 0-experience bug - Fixed")
print("\n🎉 Fixes are ready for production!")
