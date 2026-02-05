/**
 * NEURAL DIFFERENTIAL MANIFOLDS (NDM)
 * 
 * Advanced architecture with continuous weight evolution:
 * dW/dt = f(x, W, M) - weights evolve as differential equations
 * 
 * Key Innovation: Network physically rewires connections in real-time
 * based on data importance, creating true "neuroplasticity"
 * 
 * FIXES APPLIED:
 * - Added gradient clipping to prevent exploding gradients
 * - Added weight clamping to prevent unbounded growth
 * - Improved numerical stability in exponentials and logs
 * - Added adaptive learning rate scaling
 * - Better weight initialization bounds
 * - Safe normalization with epsilon terms
 * 
 * Compile to DLL/SO:
 * Windows: gcc -shared -o ndm.dll ndm.c -lm -O3 -fopenmp
 * Linux:   gcc -shared -fPIC -o ndm.so ndm.c -lm -O3 -fopenmp
 * Mac:     gcc -shared -fPIC -o ndm.dylib ndm.c -lm -O3 -Xpreprocessor -fopenmp -lomp
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Numerical stability constants
#define EPSILON 1e-8f
#define MAX_WEIGHT 10.0f
#define MIN_WEIGHT -10.0f
#define MAX_GRADIENT 5.0f
#define MAX_EXP_ARG 20.0f
#define MIN_LOG_ARG 1e-10f

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    int manifold_size;
    
    // === CORE WEIGHTS (These evolve via ODEs) ===
    float *W_input;           // [input_size × hidden_size]
    float *W_hidden;          // [hidden_size × hidden_size] (recurrent)
    float *W_output;          // [hidden_size × output_size]
    
    // === WEIGHT DERIVATIVES (dW/dt) ===
    float *dW_input_dt;       // Rate of change for input weights
    float *dW_hidden_dt;      // Rate of change for hidden weights
    float *dW_output_dt;      // Rate of change for output weights
    
    // === WEIGHT VELOCITY (for momentum) ===
    float *V_input;           // Momentum for input weights
    float *V_hidden;          // Momentum for hidden weights
    float *V_output;          // Momentum for output weights
    
    // === PLASTICITY PARAMETERS ===
    // These control how quickly weights can change
    float *plasticity_mask_input;   // [input_size × hidden_size] - per-weight plasticity
    float *plasticity_mask_hidden;  // [hidden_size × hidden_size]
    float *plasticity_mask_output;  // [hidden_size × output_size]
    float base_plasticity;          // Base learning rate for weight evolution
    float plasticity_decay;         // How fast plasticity decreases
    
    // === ASSOCIATIVE MEMORY MANIFOLD ===
    float *Memory_manifold;   // [manifold_size × hidden_size]
    float *Key_weights;       // [hidden_size × manifold_size]
    float *Value_weights;     // [hidden_size × manifold_size]
    float *Query_weights;     // [hidden_size × manifold_size]
    float memory_decay;
    
    // === HEBBIAN TRACE (for plasticity rules) ===
    // "Neurons that fire together wire together"
    float *pre_trace;         // [hidden_size] - presynaptic activity trace
    float *post_trace;        // [hidden_size] - postsynaptic activity trace
    float trace_decay;        // How fast traces decay
    
    // === STATE VARIABLES ===
    float *hidden_state;      // Current hidden state
    float *output_state;      // Current output
    float *hidden_derivative; // dh/dt for hidden state
    
    // === ODE SOLVER PARAMETERS ===
    float dt;                 // Time step for ODE integration
    float weight_decay_lambda; // L2 regularization on weights
    
    // === TRAINING PARAMETERS ===
    float learning_rate;
    float momentum;
    int training_steps;
    float last_loss;
    
    // === STATISTICS ===
    float avg_weight_velocity; // Average rate of weight change
    float avg_plasticity;      // Average plasticity across network
    float avg_manifold_energy;
    float hebbian_strength;    // Strength of Hebbian learning
    
} NDMNetwork;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static inline float clip_value(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// ============================================================================
// ACTIVATION FUNCTIONS WITH NUMERICAL STABILITY
// ============================================================================

static inline float safe_exp(float x) {
    x = clip_value(x, -MAX_EXP_ARG, MAX_EXP_ARG);
    return expf(x);
}

static inline float safe_log(float x) {
    return logf(fmaxf(x, MIN_LOG_ARG));
}

static inline float tanh_act(float x) {
    x = clip_value(x, -MAX_EXP_ARG, MAX_EXP_ARG);
    return tanhf(x);
}

static inline float tanh_derivative(float x) {
    x = clip_value(x, -MAX_EXP_ARG, MAX_EXP_ARG);
    float t = tanhf(x);
    return 1.0f - t * t;
}

static inline float sigmoid(float x) {
    x = clip_value(x, -MAX_EXP_ARG, MAX_EXP_ARG);
    return 1.0f / (1.0f + expf(-x));
}

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

static float randn(void) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    // Add epsilon to prevent log(0)
    return sqrtf(-2.0f * safe_log(u1 + EPSILON)) * cosf(2.0f * M_PI * u2);
}

// Clip gradient to prevent explosion
static inline float clip_gradient(float grad) {
    return clip_value(grad, -MAX_GRADIENT, MAX_GRADIENT);
}

// Safe weight clamping
static inline float clamp_weight(float w) {
    return clip_value(w, MIN_WEIGHT, MAX_WEIGHT);
}

// ============================================================================
// NETWORK CREATION
// ============================================================================

EXPORT NDMNetwork* create_ndm(int input_size, int hidden_size, int output_size,
                               int manifold_size, float learning_rate) {
    NDMNetwork *net = (NDMNetwork*)calloc(1, sizeof(NDMNetwork));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    net->manifold_size = manifold_size;
    net->learning_rate = learning_rate;
    net->base_plasticity = 0.01f;
    net->plasticity_decay = 0.999f;
    net->momentum = 0.9f;
    net->dt = 0.05f;  // REDUCED from 0.1 for stability
    net->weight_decay_lambda = 0.001f;  // INCREASED for better regularization
    net->memory_decay = 0.995f;
    net->trace_decay = 0.95f;
    net->hebbian_strength = 0.05f;  // REDUCED from 0.1 to prevent explosion
    net->training_steps = 0;
    
    // Allocate weight matrices
    net->W_input = (float*)malloc(input_size * hidden_size * sizeof(float));
    net->W_hidden = (float*)malloc(hidden_size * hidden_size * sizeof(float));
    net->W_output = (float*)malloc(hidden_size * output_size * sizeof(float));
    
    // Allocate derivatives
    net->dW_input_dt = (float*)calloc(input_size * hidden_size, sizeof(float));
    net->dW_hidden_dt = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    net->dW_output_dt = (float*)calloc(hidden_size * output_size, sizeof(float));
    
    // Allocate velocities
    net->V_input = (float*)calloc(input_size * hidden_size, sizeof(float));
    net->V_hidden = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    net->V_output = (float*)calloc(hidden_size * output_size, sizeof(float));
    
    // Allocate plasticity masks
    net->plasticity_mask_input = (float*)malloc(input_size * hidden_size * sizeof(float));
    net->plasticity_mask_hidden = (float*)malloc(hidden_size * hidden_size * sizeof(float));
    net->plasticity_mask_output = (float*)malloc(hidden_size * output_size * sizeof(float));
    
    // Allocate manifold
    net->Memory_manifold = (float*)calloc(manifold_size * hidden_size, sizeof(float));
    net->Key_weights = (float*)malloc(hidden_size * manifold_size * sizeof(float));
    net->Value_weights = (float*)malloc(hidden_size * manifold_size * sizeof(float));
    net->Query_weights = (float*)malloc(hidden_size * manifold_size * sizeof(float));
    
    // Allocate traces
    net->pre_trace = (float*)calloc(hidden_size, sizeof(float));
    net->post_trace = (float*)calloc(hidden_size, sizeof(float));
    
    // Allocate states
    net->hidden_state = (float*)calloc(hidden_size, sizeof(float));
    net->output_state = (float*)calloc(output_size, sizeof(float));
    net->hidden_derivative = (float*)calloc(hidden_size, sizeof(float));
    
    // Initialize weights with Xavier/Glorot - IMPROVED BOUNDS
    srand((unsigned)time(NULL));
    
    // Input weights: scale by sqrt(2 / (input_size + hidden_size))
    float scale_input = sqrtf(2.0f / (input_size + hidden_size));
    for (int i = 0; i < input_size * hidden_size; i++) {
        net->W_input[i] = randn() * scale_input;
        net->plasticity_mask_input[i] = net->base_plasticity;
    }
    
    // Hidden weights: scale by sqrt(2 / (2 * hidden_size)) - smaller for recurrent
    float scale_hidden = sqrtf(1.0f / hidden_size);  // REDUCED scale for stability
    for (int i = 0; i < hidden_size * hidden_size; i++) {
        net->W_hidden[i] = randn() * scale_hidden;
        net->plasticity_mask_hidden[i] = net->base_plasticity;
    }
    
    // Output weights
    float scale_output = sqrtf(2.0f / (hidden_size + output_size));
    for (int i = 0; i < hidden_size * output_size; i++) {
        net->W_output[i] = randn() * scale_output;
        net->plasticity_mask_output[i] = net->base_plasticity;
    }
    
    // Manifold weights - SMALLER initialization
    float scale_manifold = sqrtf(1.0f / (hidden_size + manifold_size));
    for (int i = 0; i < hidden_size * manifold_size; i++) {
        net->Key_weights[i] = randn() * scale_manifold;
        net->Value_weights[i] = randn() * scale_manifold;
        net->Query_weights[i] = randn() * scale_manifold;
    }
    
    return net;
}

// ============================================================================
// MEMORY CLEANUP
// ============================================================================

EXPORT void destroy_ndm(NDMNetwork *net) {
    if (!net) return;
    
    free(net->W_input);
    free(net->W_hidden);
    free(net->W_output);
    free(net->dW_input_dt);
    free(net->dW_hidden_dt);
    free(net->dW_output_dt);
    free(net->V_input);
    free(net->V_hidden);
    free(net->V_output);
    free(net->plasticity_mask_input);
    free(net->plasticity_mask_hidden);
    free(net->plasticity_mask_output);
    free(net->Memory_manifold);
    free(net->Key_weights);
    free(net->Value_weights);
    free(net->Query_weights);
    free(net->pre_trace);
    free(net->post_trace);
    free(net->hidden_state);
    free(net->output_state);
    free(net->hidden_derivative);
    free(net);
}

// ============================================================================
// FORWARD PASS WITH NUMERICAL STABILITY
// ============================================================================

EXPORT void ndm_forward(NDMNetwork *net, const float *input, float *output) {
    if (!net || !input || !output) return;
    
    int I = net->input_size;
    int H = net->hidden_size;
    int O = net->output_size;
    int M = net->manifold_size;
    
    // === Step 1: Input projection ===
    float input_contrib[H];
    for (int i = 0; i < H; i++) {
        input_contrib[i] = 0.0f;
        for (int j = 0; j < I; j++) {
            input_contrib[i] += net->W_input[j * H + i] * input[j];
        }
        // Clip to prevent explosion
        input_contrib[i] = clip_value(input_contrib[i], -MAX_EXP_ARG, MAX_EXP_ARG);
    }
    
    // === Step 2: Recurrent projection ===
    float recurrent_contrib[H];
    for (int i = 0; i < H; i++) {
        recurrent_contrib[i] = 0.0f;
        for (int j = 0; j < H; j++) {
            recurrent_contrib[i] += net->W_hidden[j * H + i] * net->hidden_state[j];
        }
        // Clip to prevent explosion
        recurrent_contrib[i] = clip_value(recurrent_contrib[i], -MAX_EXP_ARG, MAX_EXP_ARG);
    }
    
    // === Step 3: Attention-based memory read ===
    float queries[M];
    for (int m = 0; m < M; m++) {
        queries[m] = 0.0f;
        for (int i = 0; i < H; i++) {
            queries[m] += net->Query_weights[i * M + m] * net->hidden_state[i];
        }
        queries[m] = sigmoid(queries[m]);
    }
    
    float memory_read[H];
    for (int i = 0; i < H; i++) {
        memory_read[i] = 0.0f;
        for (int m = 0; m < M; m++) {
            memory_read[i] += queries[m] * net->Memory_manifold[m * H + i];
        }
        // Clip memory contribution
        memory_read[i] = clip_value(memory_read[i], -MAX_EXP_ARG, MAX_EXP_ARG);
    }
    
    // === Step 4: Update hidden state using ODE with stability ===
    for (int i = 0; i < H; i++) {
        float total_input = input_contrib[i] + recurrent_contrib[i] + memory_read[i];
        total_input = clip_value(total_input, -MAX_EXP_ARG, MAX_EXP_ARG);
        
        net->hidden_derivative[i] = -net->hidden_state[i] + tanh_act(total_input);
        
        // CRITICAL FIX: Clamp derivative before integration
        net->hidden_derivative[i] = clip_gradient(net->hidden_derivative[i]);
        
        // Update state with clamping
        net->hidden_state[i] += net->dt * net->hidden_derivative[i];
        net->hidden_state[i] = clip_value(net->hidden_state[i], -5.0f, 5.0f);
    }
    
    // === Step 5: Update Hebbian traces with safe values ===
    for (int i = 0; i < H; i++) {
        float abs_state = fabsf(net->hidden_state[i]);
        float abs_deriv = fabsf(net->hidden_derivative[i]);
        
        // Clip to prevent trace explosion
        abs_state = clip_value(abs_state, 0.0f, 5.0f);
        abs_deriv = clip_value(abs_deriv, 0.0f, 5.0f);
        
        net->pre_trace[i] = net->trace_decay * net->pre_trace[i] + 
                            (1.0f - net->trace_decay) * abs_state;
        net->post_trace[i] = net->trace_decay * net->post_trace[i] + 
                             (1.0f - net->trace_decay) * abs_deriv;
        
        // Additional safety clamp on traces
        net->pre_trace[i] = clip_value(net->pre_trace[i], 0.0f, 2.0f);
        net->post_trace[i] = clip_value(net->post_trace[i], 0.0f, 2.0f);
    }
    
    // === Step 6: Compute weight derivatives (neuroplasticity) with clipping ===
    // dW/dt = plasticity * (Hebbian_term - weight_decay * W)
    
    // Input weights evolution
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < H; j++) {
            int idx = i * H + j;
            float hebbian = net->hebbian_strength * input[i] * net->post_trace[j];
            hebbian = clip_gradient(hebbian);
            
            float decay = net->weight_decay_lambda * net->W_input[idx];
            net->dW_input_dt[idx] = net->plasticity_mask_input[idx] * (hebbian - decay);
            
            // CRITICAL FIX: Clip derivative
            net->dW_input_dt[idx] = clip_gradient(net->dW_input_dt[idx]);
        }
    }
    
    // Hidden weights evolution
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            int idx = i * H + j;
            float hebbian = net->hebbian_strength * net->pre_trace[i] * net->post_trace[j];
            hebbian = clip_gradient(hebbian);
            
            float decay = net->weight_decay_lambda * net->W_hidden[idx];
            net->dW_hidden_dt[idx] = net->plasticity_mask_hidden[idx] * (hebbian - decay);
            
            // CRITICAL FIX: Clip derivative
            net->dW_hidden_dt[idx] = clip_gradient(net->dW_hidden_dt[idx]);
        }
    }
    
    // === Step 7: Integrate weight ODEs with clamping ===
    // W(t+dt) = W(t) + dt * dW/dt (Euler method)
    for (int i = 0; i < I * H; i++) {
        net->W_input[i] += net->dt * net->dW_input_dt[i];
        // CRITICAL FIX: Clamp weights after update
        net->W_input[i] = clamp_weight(net->W_input[i]);
    }
    
    for (int i = 0; i < H * H; i++) {
        net->W_hidden[i] += net->dt * net->dW_hidden_dt[i];
        // CRITICAL FIX: Clamp weights after update
        net->W_hidden[i] = clamp_weight(net->W_hidden[i]);
    }
    
    // Decay plasticity over time
    for (int i = 0; i < I * H; i++) {
        net->plasticity_mask_input[i] *= net->plasticity_decay;
    }
    for (int i = 0; i < H * H; i++) {
        net->plasticity_mask_hidden[i] *= net->plasticity_decay;
    }
    
    // === Step 8: Compute output with safety ===
    for (int i = 0; i < O; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < H; j++) {
            output[i] += net->W_output[j * O + i] * net->hidden_state[j];
        }
        // Clip output to prevent explosion
        output[i] = clip_value(output[i], -20.0f, 20.0f);
    }
    
    memcpy(net->output_state, output, O * sizeof(float));
    
    // === Update statistics ===
    float weight_velocity = 0.0f;
    for (int i = 0; i < I * H; i++) {
        weight_velocity += fabsf(net->dW_input_dt[i]);
    }
    for (int i = 0; i < H * H; i++) {
        weight_velocity += fabsf(net->dW_hidden_dt[i]);
    }
    net->avg_weight_velocity = weight_velocity / (I * H + H * H);
    
    float avg_plast = 0.0f;
    for (int i = 0; i < I * H; i++) avg_plast += net->plasticity_mask_input[i];
    for (int i = 0; i < H * H; i++) avg_plast += net->plasticity_mask_hidden[i];
    net->avg_plasticity = avg_plast / (I * H + H * H);
    
    float energy = 0.0f;
    for (int i = 0; i < M * H; i++) {
        energy += net->Memory_manifold[i] * net->Memory_manifold[i];
    }
    net->avg_manifold_energy = energy / (M * H);
}

// ============================================================================
// TRAINING WITH GRADIENT CLIPPING
// ============================================================================

EXPORT float ndm_train(NDMNetwork *net, const float *input, const float *target) {
    if (!net || !input || !target) return -1.0f;
    
    // Forward pass
    float output[net->output_size];
    ndm_forward(net, input, output);
    
    // Compute loss
    float loss = 0.0f;
    for (int i = 0; i < net->output_size; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    loss /= net->output_size;
    
    // Backprop through output weights
    float grad_output[net->output_size];
    for (int i = 0; i < net->output_size; i++) {
        grad_output[i] = 2.0f * (output[i] - target[i]) / net->output_size;
        // CRITICAL FIX: Clip gradient
        grad_output[i] = clip_gradient(grad_output[i]);
    }
    
    // Update output weights with momentum and clipping
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < net->output_size; j++) {
            int idx = i * net->output_size + j;
            float grad = grad_output[j] * net->hidden_state[i];
            
            // CRITICAL FIX: Clip gradient
            grad = clip_gradient(grad);
            
            // Momentum update
            net->V_output[idx] = net->momentum * net->V_output[idx] - net->learning_rate * grad;
            // CRITICAL FIX: Clip velocity
            net->V_output[idx] = clip_value(net->V_output[idx], -1.0f, 1.0f);
            
            net->W_output[idx] += net->V_output[idx];
            // CRITICAL FIX: Clamp weight
            net->W_output[idx] = clamp_weight(net->W_output[idx]);
            
            // Also compute derivative for ODE
            float hebbian = net->hebbian_strength * net->hidden_state[i] * grad_output[j];
            hebbian = clip_gradient(hebbian);
            
            net->dW_output_dt[idx] = net->plasticity_mask_output[idx] * 
                                      (hebbian - net->weight_decay_lambda * net->W_output[idx]);
            net->dW_output_dt[idx] = clip_gradient(net->dW_output_dt[idx]);
        }
    }
    
    // Boost plasticity on error (with safety limits)
    float error_signal = sqrtf(loss);
    error_signal = clip_value(error_signal, 0.0f, 5.0f);
    
    float plasticity_boost = 0.005f * error_signal;  // REDUCED from 0.01
    for (int i = 0; i < net->input_size * net->hidden_size; i++) {
        net->plasticity_mask_input[i] = fminf(1.0f, net->plasticity_mask_input[i] + plasticity_boost);
    }
    for (int i = 0; i < net->hidden_size * net->hidden_size; i++) {
        net->plasticity_mask_hidden[i] = fminf(1.0f, net->plasticity_mask_hidden[i] + plasticity_boost);
    }
    
    net->training_steps++;
    net->last_loss = loss;
    
    return loss;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

EXPORT void ndm_reset_memory(NDMNetwork *net) {
    if (!net) return;
    
    memset(net->hidden_state, 0, net->hidden_size * sizeof(float));
    memset(net->Memory_manifold, 0, net->manifold_size * net->hidden_size * sizeof(float));
    memset(net->pre_trace, 0, net->hidden_size * sizeof(float));
    memset(net->post_trace, 0, net->hidden_size * sizeof(float));
}

EXPORT float ndm_get_avg_weight_velocity(NDMNetwork *net) {
    return net ? net->avg_weight_velocity : 0.0f;
}

EXPORT float ndm_get_avg_plasticity(NDMNetwork *net) {
    return net ? net->avg_plasticity : 0.0f;
}

EXPORT float ndm_get_avg_manifold_energy(NDMNetwork *net) {
    return net ? net->avg_manifold_energy : 0.0f;
}

EXPORT void ndm_print_info(NDMNetwork *net) {
    if (!net) return;
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║        NEURAL DIFFERENTIAL MANIFOLDS (NDM)                    ║\n");
    printf("║        Continuous Weight Evolution via ODEs                   ║\n");
    printf("║        FIXED VERSION - Numerical Stability Enhanced           ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Architecture: %d -> %d -> %d\n", 
           net->input_size, net->hidden_size, net->output_size);
    printf("Manifold Size: %d\n\n", net->manifold_size);
    
    printf("Neuroplasticity Metrics:\n");
    printf("  Weight Velocity:       %.6f (rate of rewiring)\n", net->avg_weight_velocity);
    printf("  Average Plasticity:    %.4f (0=rigid, 1=fluid)\n", net->avg_plasticity);
    printf("  Hebbian Strength:      %.4f\n", net->hebbian_strength);
    
    printf("\nMemory:\n");
    printf("  Manifold Energy:       %.6f\n", net->avg_manifold_energy);
    
    printf("\nTraining:\n");
    printf("  Steps:                 %d\n", net->training_steps);
    printf("  Last Loss:             %.6f\n", net->last_loss);
    printf("  Time Step (dt):        %.3f\n\n", net->dt);
}