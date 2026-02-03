#!/usr/bin/env python3
"""
Sample Usage of Memory-Native Neural Networks API
==================================================

Comprehensive examples demonstrating all features of the unified API
for AMRC (Adaptive Memory Recurrent Cell) and PMRC (Persistent Memory
Recurrent Cell) models.

Run this file to see all features in action:
    python sample.py
Licensed under GPL V3.
"""

import numpy as np
from pathlib import Path
import sys

# Import the unified API
try:
    from api import AMRC, PMRC, AMN, create_model, get_info
except ImportError:
    print("Error: Could not import api.py")
    print("Make sure api.py is in the same directory")
    sys.exit(1)


# ============================================================================
# EXAMPLE 1: Basic AMRC Usage
# ============================================================================

def example_01_amrc_basic():
    """Basic training and prediction with AMRC"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic AMRC Usage")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randn(100, 5).astype(np.float32)
    X_test = np.random.randn(20, 10).astype(np.float32)
    y_test = np.random.randn(20, 5).astype(np.float32)
    
    # Create AMRC model
    print("\nCreating AMRC model...")
    model = AMRC(
        input_size=10,
        hidden_size=20,
        output_size=5,
        beta=0.3,        # Memory preservation strength
        alpha=0.1,       # Memory update rate
        learning_rate=0.01
    )
    
    print(f"  Input size: {model.input_size}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Output size: {model.output_size}")
    print(f"  Beta (memory): {model.beta}")
    print(f"  Alpha (update): {model.alpha}")
    
    # Train
    print("\nTraining model...")
    model.fit(X_train, y_train, epochs=50, verbose=1)
    
    # Predict
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Evaluate
    score = model.score(X_test, y_test)
    print(f"\nTest R² Score: {score:.4f}")
    print(f"Training steps: {model.training_steps}")
    print(f"Last loss: {model.last_loss:.6f}")
    print(f"Prediction shape: {predictions.shape}")
    
    return model


# ============================================================================
# EXAMPLE 2: PMRC with Learnable Gates
# ============================================================================

def example_02_pmrc_learnable():
    """PMRC with learnable memory gates"""
    print("\n" + "="*70)
    print("EXAMPLE 2: PMRC with Learnable Memory Gates")
    print("="*70)
    
    np.random.seed(42)
    X_train = np.random.randn(150, 8).astype(np.float32)
    y_train = np.random.randn(150, 3).astype(np.float32)
    
    # Create PMRC with learnable gates
    print("\nCreating PMRC model with learnable gates...")
    model = PMRC(
        input_size=8,
        hidden_size=16,
        output_size=3,
        beta=0.3,
        alpha=0.1,
        use_recurrent=True,
        use_learnable_gates=True,  # Let network learn what to remember
        use_output_memory=True,
        learning_rate=0.01
    )
    
    print(f"  Features enabled:")
    print(f"    - Recurrent connections: {model.use_recurrent}")
    print(f"    - Learnable gates: {model.use_learnable_gates}")
    print(f"    - Output memory: {model.use_output_memory}")
    
    # Check initial gate state
    print(f"\n  Initial avg gate value: {model.avg_gate_value:.4f}")
    
    # Train
    print("\nTraining...")
    model.fit(X_train, y_train, epochs=50, verbose=1)
    
    # Check learned gate state
    print(f"\nAfter training:")
    print(f"  Avg gate value: {model.avg_gate_value:.4f}")
    print(f"  Avg memory magnitude: {model.avg_memory_magnitude:.6f}")
    
    gate_states = model.get_gate_state()
    print(f"  Gate state sample: {gate_states[:5]}")
    print(f"  Gate stats: min={gate_states.min():.3f}, max={gate_states.max():.3f}")
    
    print("\n✓ Network learned what to remember!")
    
    return model


# ============================================================================
# EXAMPLE 3: Memory State Management
# ============================================================================

def example_03_memory_management():
    """Demonstrating memory state manipulation"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Memory State Management")
    print("="*70)
    
    np.random.seed(42)
    model = AMRC(input_size=5, hidden_size=10, output_size=2, beta=0.5, alpha=0.2)
    
    # Initial state
    print("\nInitial memory state:")
    initial_memory = model.get_memory_state()
    print(f"  Memory shape: {initial_memory.shape}")
    print(f"  Memory sample: {initial_memory[:5]}")
    
    # Process some data
    X = np.random.randn(10, 5).astype(np.float32)
    predictions = model.predict(X)
    
    # Memory after processing
    print("\nMemory after processing 10 samples:")
    after_memory = model.get_memory_state()
    print(f"  Memory sample: {after_memory[:5]}")
    print(f"  Change from initial: {np.abs(after_memory - initial_memory).mean():.6f}")
    
    # Save memory snapshot
    memory_snapshot = model.get_memory_state()
    
    # Process more data
    X_more = np.random.randn(5, 5).astype(np.float32)
    model.predict(X_more)
    
    print("\nMemory after processing 5 more samples:")
    new_memory = model.get_memory_state()
    print(f"  Memory sample: {new_memory[:5]}")
    
    # Restore snapshot
    print("\nRestoring memory snapshot...")
    model.set_memory_state(memory_snapshot)
    restored_memory = model.get_memory_state()
    print(f"  Memory sample: {restored_memory[:5]}")
    print(f"  Difference from snapshot: {np.abs(restored_memory - memory_snapshot).max():.10f}")
    
    # Reset memory
    print("\nResetting memory to zero...")
    model.reset_memory()
    reset_memory = model.get_memory_state()
    print(f"  Memory sample: {reset_memory[:5]}")
    print(f"  Max absolute value: {np.abs(reset_memory).max():.10f}")
    
    print("\n✓ Memory state can be saved, restored, and reset!")


# ============================================================================
# EXAMPLE 4: Partial Training / Transfer Learning (AMRC)
# ============================================================================

def example_04_partial_training_amrc():
    """Transfer learning with layer freezing"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Partial Training with AMRC")
    print("="*70)
    
    np.random.seed(42)
    
    # Create model
    model = AMRC(input_size=8, hidden_size=16, output_size=4, learning_rate=0.02)
    
    # Generate initial training data
    X_general = np.random.randn(100, 8).astype(np.float32)
    y_general = np.random.randn(100, 4).astype(np.float32)
    
    # Phase 1: Train everything
    print("\n--- Phase 1: Training all layers ---")
    model.fit(X_general, y_general, epochs=30, verbose=0)
    print(f"Loss after full training: {model.last_loss:.6f}")
    
    # Generate new task data
    X_specific = np.random.randn(50, 8).astype(np.float32)
    y_specific = np.random.randn(50, 4).astype(np.float32)
    
    # Phase 2: Freeze hidden, train only output
    print("\n--- Phase 2: Freezing hidden layer ---")
    print("Only output layer will be updated...")
    model.freeze_hidden_layer()
    model.fit(X_specific, y_specific, epochs=20, verbose=0)
    print(f"Loss after partial training: {model.last_loss:.6f}")
    
    # Phase 3: Freeze output, train only hidden
    print("\n--- Phase 3: Freezing output layer ---")
    print("Only hidden layer will be updated...")
    model.unfreeze_hidden_layer()
    model.freeze_output_layer()
    model.fit(X_specific, y_specific, epochs=20, verbose=0)
    print(f"Loss: {model.last_loss:.6f}")
    
    # Phase 4: Unfreeze everything
    print("\n--- Phase 4: Unfreezing all layers ---")
    model.unfreeze_output_layer()
    model.fit(X_specific, y_specific, epochs=20, verbose=0)
    print(f"Final loss: {model.last_loss:.6f}")
    
    print("\n✓ Partial training enables selective learning!")


# ============================================================================
# EXAMPLE 5: Advanced Partial Training (PMRC)
# ============================================================================

def example_05_advanced_partial_training():
    """Advanced partial training techniques with PMRC"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Advanced Partial Training (PMRC)")
    print("="*70)
    
    np.random.seed(42)
    
    model = PMRC(
        input_size=10, 
        hidden_size=20, 
        output_size=5,
        use_learnable_gates=True,
        learning_rate=0.02
    )
    
    X_train = np.random.randn(80, 10).astype(np.float32)
    y_train = np.random.randn(80, 5).astype(np.float32)
    
    # Initial training
    print("\n--- Initial Training ---")
    model.fit(X_train, y_train, epochs=30, verbose=0)
    print(f"Initial loss: {model.last_loss:.6f}")
    
    # Technique 1: Freeze by percentage
    print("\n--- Technique 1: Freeze 50% of hidden neurons ---")
    model.freeze_hidden_percentage(0.5)
    model.fit(X_train, y_train, epochs=20, verbose=0)
    print(f"Loss after freezing 50%: {model.last_loss:.6f}")
    model.unfreeze_hidden_layer()
    
    # Technique 2: Freeze memory gates
    print("\n--- Technique 2: Freeze memory gates ---")
    print("Memory dynamics are now fixed...")
    model.freeze_memory_gates()
    model.fit(X_train, y_train, epochs=20, verbose=0)
    print(f"Loss with frozen gates: {model.last_loss:.6f}")
    model.unfreeze_memory_gates()
    
    # Technique 3: Freeze by magnitude
    print("\n--- Technique 3: Freeze large weights ---")
    print("Preserving strongly learned features...")
    model.freeze_by_magnitude(threshold=0.3, freeze_large=True)
    model.fit(X_train, y_train, epochs=20, verbose=0)
    print(f"Loss with frozen large weights: {model.last_loss:.6f}")
    
    print("\n✓ PMRC offers fine-grained control over training!")


# ============================================================================
# EXAMPLE 6: Model Persistence
# ============================================================================

def example_06_model_persistence():
    """Saving and loading models"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Model Persistence")
    print("="*70)
    
    np.random.seed(42)
    
    # Create and train AMRC model
    print("\n--- AMRC Model ---")
    amrc = AMRC(input_size=6, hidden_size=12, output_size=3, beta=0.4)
    
    X_train = np.random.randn(50, 6).astype(np.float32)
    y_train = np.random.randn(50, 3).astype(np.float32)
    
    amrc.fit(X_train, y_train, epochs=30, verbose=0)
    print(f"Trained AMRC - Loss: {amrc.last_loss:.6f}")
    
    # Get prediction before saving
    X_test = np.random.randn(1, 6).astype(np.float32)
    pred_before = amrc.predict(X_test)
    memory_before = amrc.get_memory_state()
    
    # Save
    print("\nSaving AMRC model...")
    amrc.save('amrc_model.bin')
    
    # Create new model and load
    print("Loading AMRC model...")
    amrc_loaded = AMRC(input_size=6, hidden_size=12, output_size=3)
    amrc_loaded.load('amrc_model.bin')
    
    # Verify
    pred_after = amrc_loaded.predict(X_test)
    memory_after = amrc_loaded.get_memory_state()
    
    print(f"Prediction difference: {np.abs(pred_before - pred_after).max():.10f}")
    print(f"Memory difference: {np.abs(memory_before - memory_after).max():.10f}")
    print("✓ AMRC model saved and loaded perfectly!")
    
    # Create and train PMRC model
    print("\n--- PMRC Model ---")
    pmrc = PMRC(input_size=6, hidden_size=12, output_size=3, use_learnable_gates=True)
    pmrc.fit(X_train, y_train, epochs=30, verbose=0)
    print(f"Trained PMRC - Loss: {pmrc.last_loss:.6f}")
    
    pred_before = pmrc.predict(X_test)
    
    # Save with different formats
    print("\nSaving PMRC with different formats...")
    pmrc.save('pmrc_custom.bin', method='custom')
    pmrc.save('pmrc_pickle.pkl', method='pickle')
    try:
        pmrc.save('pmrc_joblib.joblib', method='joblib')
        print("  ✓ Saved: custom, pickle, joblib")
    except:
        print("  ✓ Saved: custom, pickle (joblib not available)")
    
    # Load and verify
    pmrc_loaded = PMRC(input_size=6, hidden_size=12, output_size=3)
    pmrc_loaded.load('pmrc_custom.bin', method='custom')
    
    pred_after = pmrc_loaded.predict(X_test)
    print(f"Prediction difference: {np.abs(pred_before - pred_after).max():.10f}")
    print("✓ PMRC model saved and loaded perfectly!")


# ============================================================================
# EXAMPLE 7: Continuous Learning
# ============================================================================

def example_07_continuous_learning():
    """Continuous learning without catastrophic forgetting"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Continuous Learning")
    print("="*70)
    
    np.random.seed(42)
    
    model = PMRC(
        input_size=8,
        hidden_size=16,
        output_size=4,
        beta=0.4,  # Moderate memory for balance
        learning_rate=0.01
    )
    
    # Task 1: Initial learning
    print("\n--- Task 1: Initial Learning ---")
    X_task1 = np.random.randn(100, 8).astype(np.float32)
    y_task1 = np.random.randn(100, 4).astype(np.float32)
    
    model.fit(X_task1, y_task1, epochs=50, verbose=0, reset_memory=True)
    
    # Evaluate on task 1
    score_task1_before = model.score(X_task1[:20], y_task1[:20])
    print(f"Task 1 performance: {score_task1_before:.4f}")
    
    # Task 2: Continue learning (without reset)
    print("\n--- Task 2: Continuous Learning ---")
    X_task2 = np.random.randn(80, 8).astype(np.float32)
    y_task2 = np.random.randn(80, 4).astype(np.float32)
    
    # Learn task 2 without forgetting task 1
    model.freeze_hidden_percentage(0.3)  # Protect some knowledge
    model.fit(X_task2, y_task2, epochs=40, verbose=0, reset_memory=False)
    model.unfreeze_hidden_layer()
    
    # Evaluate on both tasks
    score_task1_after = model.score(X_task1[:20], y_task1[:20])
    score_task2 = model.score(X_task2[:20], y_task2[:20])
    
    print(f"Task 1 performance after Task 2: {score_task1_after:.4f}")
    print(f"Task 2 performance: {score_task2:.4f}")
    print(f"Task 1 knowledge retained: {score_task1_after/score_task1_before*100:.1f}%")
    
    print("\n✓ Model learned new task while retaining old knowledge!")


# ============================================================================
# EXAMPLE 8: Factory Function
# ============================================================================

def example_08_factory_function():
    """Using the create_model factory function"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Factory Function")
    print("="*70)
    
    np.random.seed(42)
    
    # Create models using factory function
    print("\nCreating models using create_model()...")
    
    model_amrc = create_model(
        'amrc',
        input_size=5,
        hidden_size=10,
        output_size=2,
        beta=0.3
    )
    print(f"Created AMRC: {model_amrc.model_type}")
    
    model_pmrc = create_model(
        'pmrc',
        input_size=5,
        hidden_size=10,
        output_size=2,
        use_learnable_gates=True
    )
    print(f"Created PMRC: {model_pmrc.model_type}")
    
    # Train both on same data
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randn(50, 2).astype(np.float32)
    
    print("\nTraining both models...")
    model_amrc.fit(X, y, epochs=30, verbose=0)
    model_pmrc.fit(X, y, epochs=30, verbose=0)
    
    print(f"AMRC final loss: {model_amrc.last_loss:.6f}")
    print(f"PMRC final loss: {model_pmrc.last_loss:.6f}")
    
    # Compare predictions
    X_test = np.random.randn(5, 5).astype(np.float32)
    y_test = np.random.randn(5, 2).astype(np.float32)
    
    score_amrc = model_amrc.score(X_test, y_test)
    score_pmrc = model_pmrc.score(X_test, y_test)
    
    print(f"\nAMRC test score: {score_amrc:.4f}")
    print(f"PMRC test score: {score_pmrc:.4f}")
    
    print("\n✓ Factory function simplifies model creation!")


# ============================================================================
# EXAMPLE 9: Memory Effect Visualization
# ============================================================================

def example_09_memory_effects():
    """Demonstrating effect of different memory parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Memory Parameter Effects")
    print("="*70)
    
    np.random.seed(42)
    
    configs = [
        ("No memory", 0.0, 0.0),
        ("Weak memory", 0.2, 0.1),
        ("Moderate memory", 0.5, 0.2),
        ("Strong memory", 0.8, 0.3),
    ]
    
    for name, beta, alpha in configs:
        print(f"\n--- {name} (beta={beta}, alpha={alpha}) ---")
        
        model = AMRC(input_size=3, hidden_size=6, output_size=2, beta=beta, alpha=alpha)
        
        # Strong initial input
        strong_input = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        pred = model.predict(strong_input)
        print(f"  After strong input: {pred[0]}")
        
        # Weak subsequent inputs
        weak_input = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        
        for step in range(3):
            pred = model.predict(weak_input)
            print(f"  After weak input {step+1}:  {pred[0]}")
        
        print(f"  → Higher beta = stronger retention of initial signal")


# ============================================================================
# EXAMPLE 10: Complete Workflow
# ============================================================================

def example_10_complete_workflow():
    """End-to-end workflow with validation and testing"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Complete Machine Learning Workflow")
    print("="*70)
    
    np.random.seed(42)
    
    # Generate dataset
    print("\n1. Generating dataset...")
    n_samples = 200
    X = np.random.randn(n_samples, 15).astype(np.float32)
    y = np.random.randn(n_samples, 8).astype(np.float32)
    
    # Split data: 60% train, 20% validation, 20% test
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Create model
    print("\n2. Creating PMRC model...")
    model = PMRC(
        input_size=15,
        hidden_size=30,
        output_size=8,
        beta=0.3,
        alpha=0.1,
        use_learnable_gates=True,
        learning_rate=0.01,
        random_state=42
    )
    
    # Training
    print("\n3. Training model...")
    model.fit(X_train, y_train, epochs=100, verbose=1)
    
    # Validation
    print("\n4. Validating model...")
    val_score = model.score(X_val, y_val)
    print(f"   Validation R² score: {val_score:.4f}")
    
    # If validation is good, continue; otherwise, retrain
    if val_score > 0.0:
        print("   ✓ Validation passed!")
    else:
        print("   ⚠ Low validation score, consider retraining")
    
    # Testing
    print("\n5. Testing on unseen data...")
    test_predictions = model.predict(X_test)
    test_score = model.score(X_test, y_test)
    print(f"   Test R² score: {test_score:.4f}")
    
    # Model introspection
    print("\n6. Model summary...")
    print(f"   Training steps: {model.training_steps}")
    print(f"   Final loss: {model.last_loss:.6f}")
    print(f"   Avg memory magnitude: {model.avg_memory_magnitude:.6f}")
    print(f"   Avg gate value: {model.avg_gate_value:.6f}")
    
    # Save model
    print("\n7. Saving trained model...")
    model.save('final_model.bin')
    print("   ✓ Model saved to final_model.bin")
    
    print("\n✓ Complete workflow finished successfully!")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """Run all examples"""
    
    # Print header
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*10 + "Memory-Native Neural Networks API" + " "*25 + "║")
    print("║" + " "*20 + "Sample Usage Examples" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    
    # Check API status
    info = get_info()
    print(f"\nAPI Version: {info['version']}")
    print(f"Available models:")
    print(f"  - AMRC: {info['amrc_available']}")
    print(f"  - PMRC: {info['pmrc_available']}")
    print(f"  - AMN: {info.get('amn_available', False)}")
    
    if not info['amrc_available'] and not info['pmrc_available'] and not info.get('amn_available', False):
        print("\n⚠ ERROR: No models available!")
        print("Please compile the C libraries first.")
        print("See README.md for instructions.")
        return
    
    # Run examples
    examples = [
        ("Basic AMRC Usage", example_01_amrc_basic, info['amrc_available']),
        ("PMRC Learnable Gates", example_02_pmrc_learnable, info['pmrc_available']),
        ("Memory Management", example_03_memory_management, info['amrc_available']),
        ("Partial Training (AMRC)", example_04_partial_training_amrc, info['amrc_available']),
        ("Advanced Partial Training (PMRC)", example_05_advanced_partial_training, info['pmrc_available']),
        ("Model Persistence", example_06_model_persistence, info['amrc_available'] or info['pmrc_available']),
        ("Continuous Learning", example_07_continuous_learning, info['pmrc_available']),
        ("Factory Function", example_08_factory_function, info['amrc_available'] and info['pmrc_available']),
        ("AMN Basic Usage (NEW!)", example_08a_amn_basic, info.get('amn_available', False)),
        ("AMN Manifold Memory (NEW!)", example_08b_amn_manifold, info.get('amn_available', False)),
        ("AMN vs AMRC Comparison (NEW!)", example_08c_amn_comparison, info.get('amn_available', False) and info['amrc_available']),
        ("Memory Effects", example_09_memory_effects, info['amrc_available']),
        ("Complete Workflow", example_10_complete_workflow, info['pmrc_available']),
    ]
    
    for i, (name, func, available) in enumerate(examples, 1):
        if available:
            try:
                func()
                input(f"\n[Press Enter to continue to next example...]")
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except Exception as e:
                print(f"\n⚠ Error in example {i}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n⚠ Skipping Example {i} ({name}): Required model not available")
    
    # Summary
    print("\n" + "="*70)
    print("All Examples Completed!")
    print("="*70)
    print("\nYou've seen:")
    print("  ✓ Basic model creation and training")
    print("  ✓ Memory state management")
    print("  ✓ Learnable memory dynamics (PMRC)")
    print("  ✓ Partial training and transfer learning")
    print("  ✓ Model persistence and serialization")
    print("  ✓ Continuous learning workflows")
    print("  ✓ Complete ML pipeline")
    
    print("\nFor more information:")
    print("  - Read USAGE.md for detailed documentation")
    print("  - Check api.py for full API reference")
    print("  - Explore the C implementations for theory")
    
    print("\n" + "="*70 + "\n")



# ============================================================================
# EXAMPLE 9: Memory Effect Visualization
# ============================================================================

def example_09_memory_effects():
    """Demonstrating effect of different memory parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Memory Parameter Effects")
    print("="*70)
    
    np.random.seed(42)
    
    configs = [
        ("No memory", 0.0, 0.0),
        ("Weak memory", 0.2, 0.1),
        ("Moderate memory", 0.5, 0.2),
        ("Strong memory", 0.8, 0.3),
    ]
    
    for name, beta, alpha in configs:
        print(f"\n--- {name} (beta={beta}, alpha={alpha}) ---")
        
        model = AMRC(input_size=3, hidden_size=6, output_size=2, beta=beta, alpha=alpha)
        
        # Strong initial input
        strong_input = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        pred = model.predict(strong_input)
        print(f"  After strong input: {pred[0]}")
        
        # Weak subsequent inputs
        weak_input = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        
        for step in range(3):
            pred = model.predict(weak_input)
            print(f"  After weak input {step+1}:  {pred[0]}")
        
        print(f"  → Higher beta = stronger retention of initial signal")


# ============================================================================
# EXAMPLE 10: Complete Workflow
# ============================================================================

def example_10_complete_workflow():
    """End-to-end workflow with validation and testing"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Complete Machine Learning Workflow")
    print("="*70)
    
    np.random.seed(42)
    
    # Generate dataset
    print("\n1. Generating dataset...")
    n_samples = 200
    X = np.random.randn(n_samples, 15).astype(np.float32)
    y = np.random.randn(n_samples, 8).astype(np.float32)
    
    # Split data: 60% train, 20% validation, 20% test
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Create model
    print("\n2. Creating PMRC model...")
    model = PMRC(
        input_size=15,
        hidden_size=30,
        output_size=8,
        beta=0.3,
        alpha=0.1,
        use_learnable_gates=True,
        learning_rate=0.01,
        random_state=42
    )
    
    # Training
    print("\n3. Training model...")
    model.fit(X_train, y_train, epochs=100, verbose=1)
    
    # Validation
    print("\n4. Validating model...")
    val_score = model.score(X_val, y_val)
    print(f"   Validation R² score: {val_score:.4f}")
    
    # If validation is good, continue; otherwise, retrain
    if val_score > 0.0:
        print("   ✓ Validation passed!")
    else:
        print("   ⚠ Low validation score, consider retraining")
    
    # Testing
    print("\n5. Testing on unseen data...")
    test_predictions = model.predict(X_test)
    test_score = model.score(X_test, y_test)
    print(f"   Test R² score: {test_score:.4f}")
    
    # Model introspection
    print("\n6. Model summary...")
    print(f"   Training steps: {model.training_steps}")
    print(f"   Final loss: {model.last_loss:.6f}")
    print(f"   Avg memory magnitude: {model.avg_memory_magnitude:.6f}")
    print(f"   Avg gate value: {model.avg_gate_value:.6f}")
    
    # Save model
    print("\n7. Saving trained model...")
    model.save('final_model.bin')
    print("   ✓ Model saved to final_model.bin")
    
    print("\n✓ Complete workflow finished successfully!")

def example_08a_amn_basic():
    """Basic training and prediction with AMN"""
    print("\n" + "="*70)
    print("EXAMPLE 8a: AMN Basic Usage")
    print("="*70)
    
    np.random.seed(42)
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randn(100, 5).astype(np.float32)
    
    print("\nCreating AMN model...")
    model = AMN(
        input_size=10,
        hidden_size=20,
        output_size=5,
        memory_manifold_size=64,
        learning_rate=0.005
    )
    
    print(f"  Input size: {model.input_size}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Manifold size: 64")
    
    print("\nTraining AMN...")
    model.fit(X_train, y_train, epochs=50, verbose=1)
    
    # Introspect AMN specific properties
    print(f"\nModel Diagnostics:")
    print(f"  Avg LC Timescale: {model.avg_lc_timescale:.4f}")
    print(f"  Initial Manifold Energy: {model.avg_manifold_energy:.6f}")
    
    X_test = np.random.randn(5, 10).astype(np.float32)
    preds = model.predict(X_test)
    print(f"\nPrediction shape: {preds.shape}")
    print("✓ AMN basic cycle complete!")
    
    return model

def example_08b_amn_manifold():
    """Demonstrating AMN Manifold Memory Dynamics"""
    print("\n" + "="*70)
    print("EXAMPLE 8b: AMN Manifold Memory")
    print("="*70)
    
    np.random.seed(42)
    model = AMN(input_size=4, hidden_size=16, output_size=2)
    
    print("\nProcessing sequence to observe Manifold Energy...")
    energies = []
    
    for i in range(5):
        X_step = np.random.randn(1, 4).astype(np.float32)
        model.predict(X_step)
        energy = model.avg_manifold_energy
        energies.append(energy)
        print(f"  Step {i+1}: Manifold Energy = {energy:.6f}")
    
    print("\nResetting memory...")
    model.reset_memory()
    print(f"  Energy after reset: {model.avg_manifold_energy:.6f}")
    
    print("\n✓ AMN Manifold tracks high-dimensional memory state!")

def example_08c_amn_comparison():
    """Comparison between AMRC and AMN architectures"""
    print("\n" + "="*70)
    print("EXAMPLE 8c: AMN vs AMRC Comparison")
    print("="*70)
    
    np.random.seed(42)
    X = np.random.randn(150, 12).astype(np.float32)
    y = np.random.randn(150, 4).astype(np.float32)
    
    # Setup models
    amrc = AMRC(input_size=12, hidden_size=24, output_size=4, beta=0.5)
    amn = AMN(input_size=12, hidden_size=24, output_size=4)
    
    print("\nTraining AMRC...")
    amrc.fit(X, y, epochs=40, verbose=0)
    amrc_loss = amrc.last_loss
    
    print("Training AMN...")
    amn.fit(X, y, epochs=40, verbose=0)
    # Note: AMN doesn't expose last_loss in api.py, we evaluate via score
    
    # Test performance
    X_test = np.random.randn(30, 12).astype(np.float32)
    y_test = np.random.randn(30, 4).astype(np.float32)
    
    score_amrc = amrc.score(X_test, y_test)
    score_amn = amn.score(X_test, y_test)
    
    print(f"\nResults Table:")
    print(f"{'Model':<10} | {'R² Score':<10}")
    print("-" * 25)
    print(f"{'AMRC':<10} | {score_amrc:<10.4f}")
    print(f"{'AMN':<10} | {score_amn:<10.4f}")
    
    print("\n✓ Comparison complete! AMN uses liquid state dynamics for memory.")
    
# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """Run all examples"""
    
    # Print header
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*10 + "Memory-Native Neural Networks API" + " "*25 + "║")
    print("║" + " "*20 + "Sample Usage Examples" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    
    # Check API status
    info = get_info()
    print(f"\nAPI Version: {info['version']}")
    print(f"Available models:")
    print(f"  - AMRC: {info['amrc_available']}")
    print(f"  - PMRC: {info['pmrc_available']}")
    print(f"  - AMN: {info.get('amn_available', False)}")
    
    if not info['amrc_available'] and not info['pmrc_available'] and not info.get('amn_available', False):
        print("\n⚠ ERROR: No models available!")
        print("Please compile the C libraries first.")
        print("See README.md for instructions.")
        return
    
    # Run examples
    examples = [
        ("Basic AMRC Usage", example_01_amrc_basic, info['amrc_available']),
        ("PMRC Learnable Gates", example_02_pmrc_learnable, info['pmrc_available']),
        ("Memory Management", example_03_memory_management, info['amrc_available']),
        ("Partial Training (AMRC)", example_04_partial_training_amrc, info['amrc_available']),
        ("Advanced Partial Training (PMRC)", example_05_advanced_partial_training, info['pmrc_available']),
        ("Model Persistence", example_06_model_persistence, info['amrc_available'] or info['pmrc_available']),
        ("Continuous Learning", example_07_continuous_learning, info['pmrc_available']),
        ("Factory Function", example_08_factory_function, info['amrc_available'] and info['pmrc_available']),
        ("AMN Basic Usage (NEW!)", example_08a_amn_basic, info.get('amn_available', False)),
        ("AMN Manifold Memory (NEW!)", example_08b_amn_manifold, info.get('amn_available', False)),
        ("AMN vs AMRC Comparison (NEW!)", example_08c_amn_comparison, info.get('amn_available', False) and info['amrc_available']),
        ("Memory Effects", example_09_memory_effects, info['amrc_available']),
        ("Complete Workflow", example_10_complete_workflow, info['pmrc_available']),
    ]
    
    for i, (name, func, available) in enumerate(examples, 1):
        if available:
            try:
                func()
                input(f"\n[Press Enter to continue to next example...]")
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except Exception as e:
                print(f"\n⚠ Error in example {i}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n⚠ Skipping Example {i} ({name}): Required model not available")
    
    # Summary
    print("\n" + "="*70)
    print("All Examples Completed!")
    print("="*70)
    print("\nYou've seen:")
    print("  ✓ Basic model creation and training")
    print("  ✓ Memory state management")
    print("  ✓ Learnable memory dynamics (PMRC)")
    print("  ✓ Partial training and transfer learning")
    print("  ✓ Model persistence and serialization")
    print("  ✓ Continuous learning workflows")
    print("  ✓ Complete ML pipeline")
    
    print("\nFor more information:")
    print("  - Read USAGE.md for detailed documentation")
    print("  - Check api.py for full API reference")
    print("  - Explore the C implementations for theory")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()