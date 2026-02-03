#!/usr/bin/env python3
"""
AMRC/PMRC PERSISTENCE & PARTIAL TRAINING TEST (FIXED)
========================================================
Validates bit-perfect state restoration and partial training
compatibility with api.py.
Licensed under GPL V3.
"""

import numpy as np
import os
import time
from api import AMRC, PMRC,AMN

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
INPUT_DIM = 4
HIDDEN_DIM = 512
OUTPUT_DIM = 1
BETA_VAL = 0.96
FILENAME_AMRC = "model_amrc_512.bin"
FILENAME_PMRC = "model_pmrc_512.bin"
FILENAME_AMN = "model_amn_512.bin"
MANIFOLD_SIZE = 32

def print_status(msg, success=True):
    symbol = "✅" if success else "❌"
    print(f"{symbol} {msg}")

def get_sample_data(n=50, input_size=4):
    """Generate consistent synthetic data for state checking."""
    np.random.seed(42) 
    X = np.random.uniform(-1, 1, (n, input_size)).astype(np.float32)
    y = np.sin(np.sum(X, axis=1)).reshape(-1, 1)
    return X, y

# ---------------------------------------------------------
# 2. AMRC PERSISTENCE TEST
# ---------------------------------------------------------
def run_amrc_test():
    print("\n" + "="*60)
    print("  PHASE 1: AMRC PERSISTENCE & PARTIAL TRAINING")
    print("="*60)
    
    X, y = get_sample_data(50, INPUT_DIM)
    
    print(f"[1/4] Initializing AMRC (Beta={BETA_VAL})...")
    m1 = AMRC(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, beta=BETA_VAL)
    
    print("Training m1...")
    m1.fit(X, y, epochs=5) 
    
    # Warm up internal memory state
    m1.predict(X) 
    orig_pred = m1.predict(X)
    
    print(f"[2/4] Saving model to {FILENAME_AMRC}...")
    m1.save(FILENAME_AMRC)

    print(f"[3/4] Loading into fresh instance...")
    m2 = AMRC(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    m2.load(FILENAME_AMRC)
    
    if abs(m2.beta - BETA_VAL) < 1e-4:
        print_status(f"AMRC Property Restore: Beta is correctly {m2.beta}")
    
    m2.predict(X)
    new_pred = m2.predict(X)
    
    diff = np.max(np.abs(orig_pred - new_pred))
    if diff < 1e-5:
        print_status(f"AMRC Load Success: Predictions match (Max Diff: {diff:.2e})")
    else:
        print_status(f"AMRC Load Error: Prediction mismatch (Diff: {diff:.2e})", False)

    print("[4/4] Testing Incremental Training (reset_memory=False)...")
    try:
        m2.fit(X[:10], y[:10], epochs=1, reset_memory=False)
        print_status("AMRC Incremental Training: Execution successful.")
    except Exception as e:
        print_status(f"AMRC Incremental Training Error: {e}", False)

# ---------------------------------------------------------
# 3. PMRC PERSISTENCE TEST (Learnable Gates)
# ---------------------------------------------------------
def run_pmrc_test():
    print("\n" + "="*60)
    print("  PHASE 2: PMRC PERSISTENCE (LEARNABLE GATES)")
    print("="*60)
    
    X, y = get_sample_data(50, INPUT_DIM)
    
    print("[1/4] Initializing PMRC with Learnable Gates...")
    p1 = PMRC(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, use_learnable_gates=True, beta=BETA_VAL)
    p1.fit(X, y, epochs=5)
    
    p1.predict(X)
    p_orig_pred = p1.predict(X)
    p1.save(FILENAME_PMRC)
    
    print(f"[2/4] Loading from {FILENAME_PMRC}...")
    p2 = PMRC(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    p2.load(FILENAME_PMRC)
    
    if p2.use_learnable_gates:
        print_status("PMRC Property Restore: use_learnable_gates is True")
    
    p2.predict(X)
    p_new_pred = p2.predict(X)
    
    diff = np.max(np.abs(p_orig_pred - p_new_pred))
    if diff < 1e-5:
        print_status(f"PMRC Load Success: State and Weights reloaded (Diff: {diff:.2e})")
    else:
        print_status(f"PMRC Load Error: Prediction mismatch (Diff: {diff:.2e})", False)

# ---------------------------------------------------------
# 4. AMN PERSISTENCE TEST (Liquid Constant & Manifolds)
# ---------------------------------------------------------
def run_amn_test():
    print("\n" + "="*60)
    print("  PHASE 3: AMN PERSISTENCE & ADAPTIVE DYNAMICS")
    print("="*60)
    
    X, y = get_sample_data(50, INPUT_DIM)
    
    print(f"[1/4] Initializing AMN (Manifold={MANIFOLD_SIZE})...")
    # Note: AMN uses dt and memory_decay parameters
    a1 = AMN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 
             memory_manifold_size=MANIFOLD_SIZE, 
             dt=0.05)
    
    print("Training a1...")
    a1.fit(X, y, epochs=5, verbose=0)
    
    # CRITICAL: Reset memory before capturing 'gold' prediction
    # This ensures we are testing the WEIGHTS preservation
    a1.reset_memory() 
    orig_pred = a1.predict(X)
    orig_energy = a1.avg_manifold_energy
    
    print(f"[2/4] Saving AMN model to {FILENAME_AMN}...")
    a1.save(FILENAME_AMN)

    print(f"[3/4] Loading into fresh AMN instance...")
    a2 = AMN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    a2.load(FILENAME_AMN)
    
    # Ensure a2 starts with same clean state as a1 did
    a2.reset_memory() 
    
    # Verify Manifold state restoration
    if abs(a2.avg_manifold_energy - orig_energy) < 1e-6:
        print_status(f"AMN Manifold Restore: Energy level match ({a2.avg_manifold_energy:.4f})")
    
    new_pred = a2.predict(X)
    diff = np.max(np.abs(orig_pred - new_pred))
    
    if diff < 1e-7:
        print_status(f"AMN Load Success: Predictions match (Max Diff: {diff:.2e})")
    else:
        # If this still fails, check that api.py uses: self._model = AMN_Base.load(path)
        print_status(f"AMN Load Error: Prediction mismatch (Diff: {diff:.2e})", False)

    print("[4/4] Testing Manifold Reset...")
    a2.reset_memory() 
    if a2.avg_manifold_energy < 0.01:
        print_status("AMN Memory Reset: Manifold energy cleared successfully.")
# ---------------------------------------------------------
# 5. ERROR HANDLING TEST (Mismatch Guards)
# ---------------------------------------------------------
def run_error_handling_test():
    print("\n" + "="*60)
    print("  PHASE 3: ERROR & MISMATCH HANDLING")
    print("="*60)
    
    # 1. AMRC Mismatch
    print("[Scenario: Loading 512-dim AMRC file into 256-dim model]")
    m_wrong = AMRC(INPUT_DIM, 256, OUTPUT_DIM)
    try:
        m_wrong.load(FILENAME_AMRC)
        print_status("Failed to catch AMRC size mismatch!", False)
    except Exception as e:
        print_status(f"Caught Expected AMRC Size Mismatch: {str(e)[:50]}...")

    # 2. PMRC Mismatch
    print("[Scenario: Loading 512-dim PMRC file into 256-dim model]")
    p_wrong = PMRC(INPUT_DIM, 5, OUTPUT_DIM)
    try:
        # After you manually edit memory_net_extended.py, this will raise ValueError
        p_wrong.load(FILENAME_PMRC)
        print_status("Failed to catch PMRC size mismatch!", False)
    except Exception as e:
        # We catch the exception raised by your manual edit in the underlying library
        print_status(f"Caught Expected PMRC Size Mismatch: {str(e)[:50]}...")

    # 3. Missing File
    print("[Scenario: Missing File]")
    m_ghost = AMRC(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    try:
        m_ghost.load("non_existent_file.bin")
        print_status("Failed to catch missing file!", False)
    except Exception:
        print_status("Caught Expected IO Error.")
        print("[Scenario: Missing File]")
    m_ghost = PMRC(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    try:
        m_ghost.load("non_existent_file.bin")
        print_status("Failed to catch missing file!", False)
    except Exception:
        print_status("Caught Expected IO Error.")


    # 4. AMN Mismatch
    print("[Scenario: Loading 512-dim AMN file into 128-dim model]")
    a_wrong = AMN(INPUT_DIM, 128, OUTPUT_DIM)
    try:
        a_wrong.load(FILENAME_AMN)
        print_status("Failed to catch AMN size mismatch!", False)
    except Exception as e:
        print_status(f"Caught Expected AMN Size Mismatch: {str(e)[:50]}...")
# ---------------------------------------------------------
# 5. EXECUTION
# ---------------------------------------------------------
def cleanup():
    for f in [FILENAME_AMRC, FILENAME_PMRC, FILENAME_AMN]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

if __name__ == "__main__":
    start_time = time.time()
    try:
        run_amrc_test()
        run_pmrc_test()
        run_amn_test()
        run_error_handling_test()
    except Exception as e:
        print(f"\nFATAL TEST ERROR: {e}")
    finally:
        cleanup()
        print(f"\nTotal Test Time: {time.time() - start_time:.2f}s")