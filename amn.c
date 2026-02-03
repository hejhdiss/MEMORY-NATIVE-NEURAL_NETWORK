/**
 * ADAPTIVE MEMORY NETWORK (AMN) - C LIBRARY
 * 
 * Combines three cutting-edge memory architectures:
 * 1. Liquid Constant (LC) Architecture: dh/dt = -[A + B·f(x)]·h + B·f(x)
 * 2. Linear Recurrent Units (LRU) with Parallel Prefix Scan: h_t = Λh_{t-1} + Bx_t
 * 3. Associative Memory Manifolds: M_t = M_{t-1} + σ(x_t·K^T)V
 * 
 * Compile to DLL/SO:
 * Windows: gcc -shared -o amn.dll amn.c -lm -O3 -fopenmp
 * Linux:   gcc -shared -fPIC -o amn.so amn.c -lm -O3 -fopenmp
 * Mac:     gcc -shared -fPIC -o amn.dylib amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp
 * 
 * Licensed under GPL V3.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    int memory_manifold_size;  // Size of associative memory matrix
    
    // === LIQUID CONSTANT (LC) PARAMETERS ===
    // dh/dt = -[A + B·f(x)]·h + B·f(x)
    float *A_rigidity;         // Per-neuron rigidity constants [hidden_size]
    float *B_fluidity;         // Per-neuron fluidity constants [hidden_size]
    float dt;                  // Time step (handles irregular timing)
    
    // === LINEAR RECURRENT UNIT (LRU) PARAMETERS ===
    // h_t = Λh_{t-1} + Bx_t
    float *Lambda_diag;        // Diagonal decay matrix [hidden_size]
    float *B_input;            // Input projection [input_size × hidden_size]
    float *C_output;           // Output projection [hidden_size × output_size]
    float *D_skip;             // Skip connection [input_size × output_size]
    
    // === ASSOCIATIVE MEMORY MANIFOLD (AMM) ===
    // M_t = M_{t-1} + σ(x_t·K^T)V
    float *Memory_manifold;    // Global memory matrix [memory_manifold_size × hidden_size]
    float *Key_weights;        // Key projection [hidden_size × memory_manifold_size]
    float *Value_weights;      // Value projection [hidden_size × memory_manifold_size]
    float *Query_weights;      // Query projection for reading [hidden_size × memory_manifold_size]
    float memory_decay;        // Manifold decay factor (0.99-0.999)
    
    // === STATE VARIABLES ===
    float *hidden_state;       // Current hidden state [hidden_size]
    float *lc_derivative;      // LC derivative term [hidden_size]
    float *lru_state;          // LRU recurrent state [hidden_size]
    float *output_state;       // Current output [output_size]
    
    // === TRAINING PARAMETERS ===
    float learning_rate;
    float gradient_clip_norm;
    int training_steps;
    float last_loss;
    
    // === STATISTICS ===
    float avg_manifold_energy;  // Average energy in memory manifold
    float avg_lru_magnitude;    // Average LRU state magnitude
    float avg_lc_timescale;     // Average effective timescale
    
} AMNetwork;

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

static inline float tanh_act(float x) {
    return tanhf(x);
}

static inline float tanh_derivative(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

static inline float swish(float x) {
    return x * sigmoid(x);
}

static inline float swish_derivative(float x) {
    float s = sigmoid(x);
    return s + x * s * (1.0f - s);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static float randn(void) {
    // Box-Muller transform for Gaussian random numbers
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * M_PI * u2);
}

static float clip_value(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// ============================================================================
// NETWORK CREATION AND DESTRUCTION
// ============================================================================

EXPORT AMNetwork* create_amn(int input_size, int hidden_size, int output_size,
                              int memory_manifold_size, float learning_rate) {
    AMNetwork *net = (AMNetwork*)calloc(1, sizeof(AMNetwork));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    net->memory_manifold_size = memory_manifold_size;
    net->learning_rate = learning_rate;
    net->dt = 0.1f;  // Default time step
    net->memory_decay = 0.995f;  // Default memory decay
    net->gradient_clip_norm = 10.0f;
    net->training_steps = 0;
    
    // Allocate LC parameters
    net->A_rigidity = (float*)malloc(hidden_size * sizeof(float));
    net->B_fluidity = (float*)malloc(hidden_size * sizeof(float));
    
    // Allocate LRU parameters
    net->Lambda_diag = (float*)malloc(hidden_size * sizeof(float));
    net->B_input = (float*)malloc(input_size * hidden_size * sizeof(float));
    net->C_output = (float*)malloc(hidden_size * output_size * sizeof(float));
    net->D_skip = (float*)malloc(input_size * output_size * sizeof(float));
    
    // Allocate AMM parameters
    net->Memory_manifold = (float*)calloc(memory_manifold_size * hidden_size, sizeof(float));
    net->Key_weights = (float*)malloc(hidden_size * memory_manifold_size * sizeof(float));
    net->Value_weights = (float*)malloc(hidden_size * memory_manifold_size * sizeof(float));
    net->Query_weights = (float*)malloc(hidden_size * memory_manifold_size * sizeof(float));
    
    // Allocate state variables
    net->hidden_state = (float*)calloc(hidden_size, sizeof(float));
    net->lc_derivative = (float*)calloc(hidden_size, sizeof(float));
    net->lru_state = (float*)calloc(hidden_size, sizeof(float));
    net->output_state = (float*)calloc(output_size, sizeof(float));
    
    // Initialize parameters with smart defaults
    srand(time(NULL));
    
    // LC parameters: Initialize with diversity
    for (int i = 0; i < hidden_size; i++) {
        net->A_rigidity[i] = 0.5f + 0.5f * ((float)rand() / RAND_MAX);  // 0.5 to 1.0
        net->B_fluidity[i] = 0.1f + 0.3f * ((float)rand() / RAND_MAX);  // 0.1 to 0.4 (smaller)
    }
    
    // LRU parameters: Initialize for stability
    for (int i = 0; i < hidden_size; i++) {
        // Lambda should be close to 1 for long-term memory, but stable
        net->Lambda_diag[i] = 0.95f + 0.04f * ((float)rand() / RAND_MAX);  // 0.95 to 0.99
    }
    
    // Xavier initialization for weight matrices
    float scale_input = sqrtf(2.0f / (input_size + hidden_size));
    float scale_output = sqrtf(2.0f / (hidden_size + output_size));
    float scale_memory = sqrtf(1.0f / hidden_size);
    
    for (int i = 0; i < input_size * hidden_size; i++) {
        net->B_input[i] = randn() * scale_input;
    }
    
    for (int i = 0; i < hidden_size * output_size; i++) {
        net->C_output[i] = randn() * scale_output;
    }
    
    for (int i = 0; i < input_size * output_size; i++) {
        net->D_skip[i] = randn() * scale_output * 0.1f;  // Smaller skip connections
    }
    
    // Memory manifold weights
    for (int i = 0; i < hidden_size * memory_manifold_size; i++) {
        net->Key_weights[i] = randn() * scale_memory;
        net->Value_weights[i] = randn() * scale_memory;
        net->Query_weights[i] = randn() * scale_memory;
    }
    
    return net;
}

EXPORT void destroy_amn(AMNetwork *net) {
    if (!net) return;
    
    free(net->A_rigidity);
    free(net->B_fluidity);
    free(net->Lambda_diag);
    free(net->B_input);
    free(net->C_output);
    free(net->D_skip);
    free(net->Memory_manifold);
    free(net->Key_weights);
    free(net->Value_weights);
    free(net->Query_weights);
    free(net->hidden_state);
    free(net->lc_derivative);
    free(net->lru_state);
    free(net->output_state);
    free(net);
}

// ============================================================================
// PARALLEL PREFIX SCAN (for LRU)
// ============================================================================

// Parallel scan to compute h_t = Λ^t h_0 + Σ(Λ^(t-k) B x_k)
// This allows parallel computation across time steps
static void parallel_prefix_scan(float *states, const float *inputs, 
                                 const float *lambda, const float *B,
                                 int seq_len, int input_size, int hidden_size) {
    // For now, sequential implementation
    // Full parallel version would use scan algorithm
    
    for (int t = 0; t < seq_len; t++) {
        const float *x_t = inputs + t * input_size;
        float *h_t = states + t * hidden_size;
        
        // h_t = Λ * h_{t-1}
        if (t > 0) {
            const float *h_prev = states + (t-1) * hidden_size;
            for (int i = 0; i < hidden_size; i++) {
                h_t[i] = lambda[i] * h_prev[i];
            }
        } else {
            memset(h_t, 0, hidden_size * sizeof(float));
        }
        
        // h_t += B * x_t
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < input_size; j++) {
                h_t[i] += B[j * hidden_size + i] * x_t[j];
            }
        }
    }
}

// ============================================================================
// FORWARD PASS
// ============================================================================

EXPORT void amn_forward(AMNetwork *net, const float *input, float *output) {
    if (!net || !input || !output) return;
    
    // === STEP 1: LIQUID CONSTANT UPDATE ===
    // Compute f(x) - apply activation to input projection
    float *f_x = (float*)alloca(net->hidden_size * sizeof(float));
    memset(f_x, 0, net->hidden_size * sizeof(float));
    
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < net->input_size; j++) {
            f_x[i] += net->B_input[j * net->hidden_size + i] * input[j];
        }
        f_x[i] = tanh_act(f_x[i]);
    }
    
    // Update LC dynamics: dh/dt = -[A + B·f(x)]·h + B·f(x)
    for (int i = 0; i < net->hidden_size; i++) {
        float effective_decay = net->A_rigidity[i] + net->B_fluidity[i] * fabsf(f_x[i]);
        net->lc_derivative[i] = -effective_decay * net->hidden_state[i] + 
                                 net->B_fluidity[i] * f_x[i];
        
        // Euler integration with clipping
        net->hidden_state[i] += net->dt * net->lc_derivative[i];
        net->hidden_state[i] = clip_value(net->hidden_state[i], -10.0f, 10.0f);
    }
    
    // === STEP 2: LRU UPDATE ===
    // h_t = Λ * h_{t-1} + B * x_t (already computed in LC, so just apply Lambda)
    for (int i = 0; i < net->hidden_size; i++) {
        net->lru_state[i] = net->Lambda_diag[i] * net->lru_state[i] + 
                            net->hidden_state[i];
        net->lru_state[i] = clip_value(net->lru_state[i], -10.0f, 10.0f);
    }
    
    // === STEP 3: ASSOCIATIVE MEMORY MANIFOLD READ ===
    // Compute query: q = Query_weights^T * hidden_state
    float *query = (float*)alloca(net->memory_manifold_size * sizeof(float));
    memset(query, 0, net->memory_manifold_size * sizeof(float));
    
    for (int i = 0; i < net->memory_manifold_size; i++) {
        for (int j = 0; j < net->hidden_size; j++) {
            query[i] += net->Query_weights[j * net->memory_manifold_size + i] * 
                        net->lru_state[j];
        }
        query[i] = tanh_act(query[i]);  // Normalize query
    }
    
    // Read from manifold: retrieved = M^T * query
    float *retrieved = (float*)alloca(net->hidden_size * sizeof(float));
    memset(retrieved, 0, net->hidden_size * sizeof(float));
    
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < net->memory_manifold_size; j++) {
            retrieved[i] += net->Memory_manifold[j * net->hidden_size + i] * query[j];
        }
    }
    
    // Combine LRU state with retrieved memory
    for (int i = 0; i < net->hidden_size; i++) {
        net->hidden_state[i] = 0.7f * net->lru_state[i] + 0.3f * retrieved[i];
    }
    
    // === STEP 4: OUTPUT PROJECTION ===
    // y = C * h + D * x (skip connection)
    memset(output, 0, net->output_size * sizeof(float));
    
    for (int i = 0; i < net->output_size; i++) {
        // Main path
        for (int j = 0; j < net->hidden_size; j++) {
            output[i] += net->C_output[j * net->output_size + i] * net->hidden_state[j];
        }
        // Skip connection
        for (int j = 0; j < net->input_size; j++) {
            output[i] += net->D_skip[j * net->output_size + i] * input[j];
        }
        
        // Clip output to prevent NaN
        output[i] = clip_value(output[i], -100.0f, 100.0f);
    }
    
    memcpy(net->output_state, output, net->output_size * sizeof(float));
    
    // === STEP 5: WRITE TO ASSOCIATIVE MEMORY ===
    // Compute key: k = Key_weights^T * hidden_state
    float *key = (float*)alloca(net->memory_manifold_size * sizeof(float));
    memset(key, 0, net->memory_manifold_size * sizeof(float));
    
    for (int i = 0; i < net->memory_manifold_size; i++) {
        for (int j = 0; j < net->hidden_size; j++) {
            key[i] += net->Key_weights[j * net->memory_manifold_size + i] * 
                     net->hidden_state[j];
        }
        key[i] = sigmoid(key[i]);  // Gate for writing
    }
    
    // Compute value: v = Value_weights^T * hidden_state
    float *value = (float*)alloca(net->memory_manifold_size * sizeof(float));
    memset(value, 0, net->memory_manifold_size * sizeof(float));
    
    for (int i = 0; i < net->memory_manifold_size; i++) {
        for (int j = 0; j < net->hidden_size; j++) {
            value[i] += net->Value_weights[j * net->memory_manifold_size + i] * 
                        net->hidden_state[j];
        }
    }
    
    // Update manifold: M = decay * M + key * value^T (outer product)
    for (int i = 0; i < net->memory_manifold_size; i++) {
        for (int j = 0; j < net->hidden_size; j++) {
            net->Memory_manifold[i * net->hidden_size + j] = 
                net->memory_decay * net->Memory_manifold[i * net->hidden_size + j] +
                0.01f * key[i] * value[i] * net->hidden_state[j];  // Scale down writes
            
            // Clip manifold values
            net->Memory_manifold[i * net->hidden_size + j] = 
                clip_value(net->Memory_manifold[i * net->hidden_size + j], -5.0f, 5.0f);
        }
    }
    
    // Update statistics
    float manifold_sum = 0.0f, lru_sum = 0.0f, timescale_sum = 0.0f;
    for (int i = 0; i < net->memory_manifold_size * net->hidden_size; i++) {
        manifold_sum += fabsf(net->Memory_manifold[i]);
    }
    for (int i = 0; i < net->hidden_size; i++) {
        lru_sum += fabsf(net->lru_state[i]);
        float effective_decay = net->A_rigidity[i] + net->B_fluidity[i] * fabsf(f_x[i]);
        timescale_sum += 1.0f / (effective_decay + 1e-6f);
    }
    
    net->avg_manifold_energy = manifold_sum / (net->memory_manifold_size * net->hidden_size);
    net->avg_lru_magnitude = lru_sum / net->hidden_size;
    net->avg_lc_timescale = timescale_sum / net->hidden_size;
}

// ============================================================================
// TRAINING (Simplified backprop through time)
// ============================================================================

EXPORT float amn_train(AMNetwork *net, const float *input, const float *target) {
    if (!net || !input || !target) return -1.0f;
    
    // Forward pass
    float *output = (float*)alloca(net->output_size * sizeof(float));
    amn_forward(net, input, output);
    
    // Compute loss (MSE)
    float loss = 0.0f;
    float *output_grad = (float*)alloca(net->output_size * sizeof(float));
    
    for (int i = 0; i < net->output_size; i++) {
        float error = output[i] - target[i];
        
        // Check for NaN
        if (isnan(error) || isinf(error)) {
            // Reset network if NaN detected
            memset(net->hidden_state, 0, net->hidden_size * sizeof(float));
            memset(net->lru_state, 0, net->hidden_size * sizeof(float));
            return 1e6f;  // Return large loss
        }
        
        loss += error * error;
        output_grad[i] = 2.0f * error / net->output_size;
    }
    loss /= net->output_size;
    
    // Backprop through output layer
    float *hidden_grad = (float*)alloca(net->hidden_size * sizeof(float));
    memset(hidden_grad, 0, net->hidden_size * sizeof(float));
    
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < net->output_size; j++) {
            hidden_grad[i] += output_grad[j] * net->C_output[i * net->output_size + j];
        }
    }
    
    // Update C_output weights
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < net->output_size; j++) {
            float grad = output_grad[j] * net->hidden_state[i];
            grad = clip_value(grad, -net->gradient_clip_norm, net->gradient_clip_norm);
            net->C_output[i * net->output_size + j] -= net->learning_rate * grad;
        }
    }
    
    // Update D_skip weights
    for (int i = 0; i < net->input_size; i++) {
        for (int j = 0; j < net->output_size; j++) {
            float grad = output_grad[j] * input[i];
            grad = clip_value(grad, -net->gradient_clip_norm, net->gradient_clip_norm);
            net->D_skip[i * net->output_size + j] -= net->learning_rate * grad;
        }
    }
    
    // Update B_input (simplified - should include LC dynamics)
    for (int i = 0; i < net->input_size; i++) {
        for (int j = 0; j < net->hidden_size; j++) {
            float grad = hidden_grad[j] * input[i];
            grad = clip_value(grad, -net->gradient_clip_norm, net->gradient_clip_norm);
            net->B_input[i * net->hidden_size + j] -= net->learning_rate * grad;
        }
    }
    
    // Update memory weights (simplified)
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < net->memory_manifold_size; j++) {
            float grad = hidden_grad[i] * 0.01f;  // Small updates to memory system
            grad = clip_value(grad, -net->gradient_clip_norm, net->gradient_clip_norm);
            net->Key_weights[i * net->memory_manifold_size + j] -= net->learning_rate * grad;
            net->Value_weights[i * net->memory_manifold_size + j] -= net->learning_rate * grad;
            net->Query_weights[i * net->memory_manifold_size + j] -= net->learning_rate * grad;
        }
    }
    
    net->training_steps++;
    net->last_loss = loss;
    
    return loss;
}

// ============================================================================
// BATCH TRAINING
// ============================================================================

EXPORT float amn_train_batch(AMNetwork *net, const float *inputs, const float *targets, 
                             int batch_size) {
    if (!net || !inputs || !targets || batch_size <= 0) return -1.0f;
    
    float total_loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        const float *input = inputs + b * net->input_size;
        const float *target = targets + b * net->output_size;
        float loss = amn_train(net, input, target);
        total_loss += loss;
    }
    
    return total_loss / batch_size;
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

EXPORT void amn_reset_memory(AMNetwork *net) {
    if (!net) return;
    
    memset(net->hidden_state, 0, net->hidden_size * sizeof(float));
    memset(net->lc_derivative, 0, net->hidden_size * sizeof(float));
    memset(net->lru_state, 0, net->hidden_size * sizeof(float));
    memset(net->output_state, 0, net->output_size * sizeof(float));
    memset(net->Memory_manifold, 0, 
           net->memory_manifold_size * net->hidden_size * sizeof(float));
}

EXPORT void amn_reset_manifold(AMNetwork *net) {
    if (!net) return;
    memset(net->Memory_manifold, 0, 
           net->memory_manifold_size * net->hidden_size * sizeof(float));
}

EXPORT void amn_get_hidden_state(AMNetwork *net, float *state_out) {
    if (!net || !state_out) return;
    memcpy(state_out, net->hidden_state, net->hidden_size * sizeof(float));
}

EXPORT void amn_set_hidden_state(AMNetwork *net, const float *state_in) {
    if (!net || !state_in) return;
    memcpy(net->hidden_state, state_in, net->hidden_size * sizeof(float));
}

EXPORT void amn_get_manifold(AMNetwork *net, float *manifold_out) {
    if (!net || !manifold_out) return;
    memcpy(manifold_out, net->Memory_manifold, 
           net->memory_manifold_size * net->hidden_size * sizeof(float));
}

// ============================================================================
// PARAMETER GETTERS/SETTERS
// ============================================================================

EXPORT float amn_get_learning_rate(AMNetwork *net) {
    return net ? net->learning_rate : 0.0f;
}

EXPORT void amn_set_learning_rate(AMNetwork *net, float lr) {
    if (net) net->learning_rate = lr;
}

EXPORT float amn_get_dt(AMNetwork *net) {
    return net ? net->dt : 0.0f;
}

EXPORT void amn_set_dt(AMNetwork *net, float dt) {
    if (net) net->dt = dt;
}

EXPORT float amn_get_memory_decay(AMNetwork *net) {
    return net ? net->memory_decay : 0.0f;
}

EXPORT void amn_set_memory_decay(AMNetwork *net, float decay) {
    if (net) net->memory_decay = clip_value(decay, 0.9f, 0.9999f);
}

EXPORT int amn_get_training_steps(AMNetwork *net) {
    return net ? net->training_steps : 0;
}

EXPORT float amn_get_last_loss(AMNetwork *net) {
    return net ? net->last_loss : 0.0f;
}

EXPORT float amn_get_avg_manifold_energy(AMNetwork *net) {
    return net ? net->avg_manifold_energy : 0.0f;
}

EXPORT float amn_get_avg_lru_magnitude(AMNetwork *net) {
    return net ? net->avg_lru_magnitude : 0.0f;
}

EXPORT float amn_get_avg_lc_timescale(AMNetwork *net) {
    return net ? net->avg_lc_timescale : 0.0f;
}

// ============================================================================
// SAVE/LOAD
// ============================================================================

EXPORT int amn_save(AMNetwork *net, const char *filename) {
    if (!net || !filename) return -1;
    
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;
    
    // Write metadata
    fwrite(&net->input_size, sizeof(int), 1, f);
    fwrite(&net->hidden_size, sizeof(int), 1, f);
    fwrite(&net->output_size, sizeof(int), 1, f);
    fwrite(&net->memory_manifold_size, sizeof(int), 1, f);
    fwrite(&net->learning_rate, sizeof(float), 1, f);
    fwrite(&net->dt, sizeof(float), 1, f);
    fwrite(&net->memory_decay, sizeof(float), 1, f);
    fwrite(&net->training_steps, sizeof(int), 1, f);
    
    // Write LC parameters
    fwrite(net->A_rigidity, sizeof(float), net->hidden_size, f);
    fwrite(net->B_fluidity, sizeof(float), net->hidden_size, f);
    
    // Write LRU parameters
    fwrite(net->Lambda_diag, sizeof(float), net->hidden_size, f);
    fwrite(net->B_input, sizeof(float), net->input_size * net->hidden_size, f);
    fwrite(net->C_output, sizeof(float), net->hidden_size * net->output_size, f);
    fwrite(net->D_skip, sizeof(float), net->input_size * net->output_size, f);
    
    // Write AMM parameters
    fwrite(net->Memory_manifold, sizeof(float), 
           net->memory_manifold_size * net->hidden_size, f);
    fwrite(net->Key_weights, sizeof(float), 
           net->hidden_size * net->memory_manifold_size, f);
    fwrite(net->Value_weights, sizeof(float), 
           net->hidden_size * net->memory_manifold_size, f);
    fwrite(net->Query_weights, sizeof(float), 
           net->hidden_size * net->memory_manifold_size, f);
    
    // Write states
    fwrite(net->hidden_state, sizeof(float), net->hidden_size, f);
    fwrite(net->lru_state, sizeof(float), net->hidden_size, f);
    
    fclose(f);
    return 0;
}

EXPORT int amn_load(AMNetwork *net, const char *filename) {
    if (!net || !filename) return -1;
    
    FILE *f = fopen(filename, "rb");
    if (!f) return -1;
    
    // Verify metadata
    int input_size, hidden_size, output_size, memory_manifold_size;
    fread(&input_size, sizeof(int), 1, f);
    fread(&hidden_size, sizeof(int), 1, f);
    fread(&output_size, sizeof(int), 1, f);
    fread(&memory_manifold_size, sizeof(int), 1, f);
    
    if (input_size != net->input_size || hidden_size != net->hidden_size ||
        output_size != net->output_size || 
        memory_manifold_size != net->memory_manifold_size) {
        fclose(f);
        return -2;
    }
    
    fread(&net->learning_rate, sizeof(float), 1, f);
    fread(&net->dt, sizeof(float), 1, f);
    fread(&net->memory_decay, sizeof(float), 1, f);
    fread(&net->training_steps, sizeof(int), 1, f);
    
    // Read parameters
    fread(net->A_rigidity, sizeof(float), net->hidden_size, f);
    fread(net->B_fluidity, sizeof(float), net->hidden_size, f);
    fread(net->Lambda_diag, sizeof(float), net->hidden_size, f);
    fread(net->B_input, sizeof(float), net->input_size * net->hidden_size, f);
    fread(net->C_output, sizeof(float), net->hidden_size * net->output_size, f);
    fread(net->D_skip, sizeof(float), net->input_size * net->output_size, f);
    fread(net->Memory_manifold, sizeof(float), 
          net->memory_manifold_size * net->hidden_size, f);
    fread(net->Key_weights, sizeof(float), 
          net->hidden_size * net->memory_manifold_size, f);
    fread(net->Value_weights, sizeof(float), 
          net->hidden_size * net->memory_manifold_size, f);
    fread(net->Query_weights, sizeof(float), 
          net->hidden_size * net->memory_manifold_size, f);
    fread(net->hidden_state, sizeof(float), net->hidden_size, f);
    fread(net->lru_state, sizeof(float), net->hidden_size, f);
    
    fclose(f);
    return 0;
}

// ============================================================================
// INFO/DEBUG
// ============================================================================

EXPORT void amn_print_info(AMNetwork *net) {
    if (!net) return;
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║           ADAPTIVE MEMORY NETWORK (AMN) INFO                  ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Architecture: %d -> %d -> %d\n", 
           net->input_size, net->hidden_size, net->output_size);
    printf("Memory Manifold Size: %d\n\n", net->memory_manifold_size);
    
    printf("Parameters:\n");
    printf("  Learning Rate:         %.4f\n", net->learning_rate);
    printf("  Time Step (dt):        %.3f\n", net->dt);
    printf("  Memory Decay:          %.4f\n", net->memory_decay);
    
    printf("\nTraining:\n");
    printf("  Steps:                 %d\n", net->training_steps);
    printf("  Last Loss:             %.6f\n", net->last_loss);
    
    printf("\nMemory Statistics:\n");
    printf("  Manifold Energy:       %.6f\n", net->avg_manifold_energy);
    printf("  LRU Magnitude:         %.6f\n", net->avg_lru_magnitude);
    printf("  LC Timescale:          %.6f\n", net->avg_lc_timescale);
    
    printf("\n");
}