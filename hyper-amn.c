/**
 * HYPER-AMN: MULTI-HEAD ASSOCIATIVE MANIFOLD NETWORK
 * 
 * Advanced architecture with specialized manifolds:
 * - Spatial Manifold: Tracks positional/structural patterns
 * - Emotional Manifold: Tracks sentiment/tone patterns
 * - Logical Manifold: Tracks reasoning/causal patterns
 * 
 * Key Innovation: Prevents memory interference by domain separation
 * 
 * Compile to DLL/SO:
 * Windows: gcc -shared -o hyper-amn.dll hyper-amn.c -lm -O3 -fopenmp
 * Linux:   gcc -shared -fPIC -o hyper-amn.so hyper-amn.c -lm -O3 -fopenmp
 * Mac:     gcc -shared -fPIC -o hyper-amn.dylib hyper-amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp
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

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef enum {
    MANIFOLD_SPATIAL = 0,    // Tracks spatial/structural patterns
    MANIFOLD_EMOTIONAL = 1,  // Tracks sentiment/emotional patterns
    MANIFOLD_LOGICAL = 2,    // Tracks reasoning/logical patterns
    NUM_MANIFOLD_HEADS = 3
} ManifoldType;

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    int manifold_size;         // Size per manifold head
    int num_heads;             // Number of manifold heads (default 3)
    
    // === LIQUID CONSTANT (LC) PARAMETERS (per head) ===
    float **A_rigidity;        // [num_heads][hidden_size]
    float **B_fluidity;        // [num_heads][hidden_size]
    float dt;
    
    // === LINEAR RECURRENT UNIT (LRU) PARAMETERS ===
    float *Lambda_diag;        // [hidden_size]
    float *B_input;            // [input_size × hidden_size]
    float *C_output;           // [hidden_size × output_size]
    float *D_skip;             // [input_size × output_size]
    
    // === MULTI-HEAD ASSOCIATIVE MEMORY MANIFOLDS ===
    // Each head specializes in different pattern types
    float **Memory_manifolds;  // [num_heads][manifold_size × hidden_size]
    float **Key_weights;       // [num_heads][hidden_size × manifold_size]
    float **Value_weights;     // [num_heads][hidden_size × manifold_size]
    float **Query_weights;     // [num_heads][hidden_size × manifold_size]
    float *manifold_decay;     // [num_heads] - different decay per head
    
    // === HEAD GATING MECHANISM ===
    // Determines which head to route information to
    float *gate_weights;       // [hidden_size × num_heads]
    float *head_activations;   // [num_heads] - current activation per head
    
    // === CROSS-HEAD COMMUNICATION ===
    // Allows manifolds to exchange information
    float **cross_head_weights; // [num_heads][num_heads] - interaction matrix
    
    // === STATE VARIABLES ===
    float **hidden_states;     // [num_heads][hidden_size]
    float *merged_hidden;      // [hidden_size] - merged across heads
    float *lru_state;          // [hidden_size]
    float *output_state;       // [output_size]
    
    // === TRAINING PARAMETERS ===
    float learning_rate;
    float gradient_clip_norm;
    int training_steps;
    float last_loss;
    
    // === STATISTICS ===
    float *avg_manifold_energy; // [num_heads]
    float avg_lru_magnitude;
    float avg_cross_head_flow;  // Average information flow between heads
    
} HyperAMN;

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

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float swish(float x) {
    return x * sigmoid(x);
}

static inline float softmax_single(float x, float *logits, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += expf(logits[i] - max_val);
    }
    
    return expf(x - max_val) / sum;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static float randn(void) {
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

EXPORT HyperAMN* create_hyper_amn(int input_size, int hidden_size, int output_size,
                                   int manifold_size, float learning_rate) {
    HyperAMN *net = (HyperAMN*)calloc(1, sizeof(HyperAMN));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    net->manifold_size = manifold_size;
    net->num_heads = NUM_MANIFOLD_HEADS;
    net->learning_rate = learning_rate;
    net->dt = 0.1f;
    net->gradient_clip_norm = 10.0f;
    net->training_steps = 0;
    
    // Allocate arrays of arrays for multi-head
    net->A_rigidity = (float**)malloc(net->num_heads * sizeof(float*));
    net->B_fluidity = (float**)malloc(net->num_heads * sizeof(float*));
    net->Memory_manifolds = (float**)malloc(net->num_heads * sizeof(float*));
    net->Key_weights = (float**)malloc(net->num_heads * sizeof(float*));
    net->Value_weights = (float**)malloc(net->num_heads * sizeof(float*));
    net->Query_weights = (float**)malloc(net->num_heads * sizeof(float*));
    net->hidden_states = (float**)malloc(net->num_heads * sizeof(float*));
    
    for (int h = 0; h < net->num_heads; h++) {
        net->A_rigidity[h] = (float*)malloc(hidden_size * sizeof(float));
        net->B_fluidity[h] = (float*)malloc(hidden_size * sizeof(float));
        net->Memory_manifolds[h] = (float*)calloc(manifold_size * hidden_size, sizeof(float));
        net->Key_weights[h] = (float*)malloc(hidden_size * manifold_size * sizeof(float));
        net->Value_weights[h] = (float*)malloc(hidden_size * manifold_size * sizeof(float));
        net->Query_weights[h] = (float*)malloc(hidden_size * manifold_size * sizeof(float));
        net->hidden_states[h] = (float*)calloc(hidden_size, sizeof(float));
    }
    
    // Allocate single arrays
    net->manifold_decay = (float*)malloc(net->num_heads * sizeof(float));
    net->Lambda_diag = (float*)malloc(hidden_size * sizeof(float));
    net->B_input = (float*)malloc(input_size * hidden_size * sizeof(float));
    net->C_output = (float*)malloc(hidden_size * output_size * sizeof(float));
    net->D_skip = (float*)malloc(input_size * output_size * sizeof(float));
    net->gate_weights = (float*)malloc(hidden_size * net->num_heads * sizeof(float));
    net->head_activations = (float*)calloc(net->num_heads, sizeof(float));
    
    // Cross-head communication
    net->cross_head_weights = (float**)malloc(net->num_heads * sizeof(float*));
    for (int i = 0; i < net->num_heads; i++) {
        net->cross_head_weights[i] = (float*)malloc(net->num_heads * sizeof(float));
    }
    
    net->merged_hidden = (float*)calloc(hidden_size, sizeof(float));
    net->lru_state = (float*)calloc(hidden_size, sizeof(float));
    net->output_state = (float*)calloc(output_size, sizeof(float));
    net->avg_manifold_energy = (float*)calloc(net->num_heads, sizeof(float));
    
    // Initialize parameters
    srand(time(NULL));
    
    // Different characteristics per head
    for (int h = 0; h < net->num_heads; h++) {
        for (int i = 0; i < hidden_size; i++) {
            if (h == MANIFOLD_SPATIAL) {
                // Spatial: Fast adaptation, short memory
                net->A_rigidity[h][i] = 0.3f + 0.3f * ((float)rand() / RAND_MAX);
                net->B_fluidity[h][i] = 0.3f + 0.4f * ((float)rand() / RAND_MAX);
            } else if (h == MANIFOLD_EMOTIONAL) {
                // Emotional: Medium adaptation, medium memory
                net->A_rigidity[h][i] = 0.5f + 0.3f * ((float)rand() / RAND_MAX);
                net->B_fluidity[h][i] = 0.2f + 0.3f * ((float)rand() / RAND_MAX);
            } else { // MANIFOLD_LOGICAL
                // Logical: Slow adaptation, long memory
                net->A_rigidity[h][i] = 0.7f + 0.3f * ((float)rand() / RAND_MAX);
                net->B_fluidity[h][i] = 0.1f + 0.2f * ((float)rand() / RAND_MAX);
            }
        }
        
        // Different decay rates
        if (h == MANIFOLD_SPATIAL) {
            net->manifold_decay[h] = 0.990f; // Faster decay
        } else if (h == MANIFOLD_EMOTIONAL) {
            net->manifold_decay[h] = 0.995f; // Medium decay
        } else {
            net->manifold_decay[h] = 0.999f; // Slower decay
        }
    }
    
    // Initialize LRU
    for (int i = 0; i < hidden_size; i++) {
        net->Lambda_diag[i] = 0.95f + 0.04f * ((float)rand() / RAND_MAX);
    }
    
    // Xavier initialization
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
        net->D_skip[i] = randn() * scale_output * 0.1f;
    }
    
    // Initialize manifold weights
    for (int h = 0; h < net->num_heads; h++) {
        for (int i = 0; i < hidden_size * manifold_size; i++) {
            net->Key_weights[h][i] = randn() * scale_memory;
            net->Value_weights[h][i] = randn() * scale_memory;
            net->Query_weights[h][i] = randn() * scale_memory;
        }
    }
    
    // Initialize gating
    for (int i = 0; i < hidden_size * net->num_heads; i++) {
        net->gate_weights[i] = randn() * 0.1f;
    }
    
    // Initialize cross-head weights (identity-like)
    for (int i = 0; i < net->num_heads; i++) {
        for (int j = 0; j < net->num_heads; j++) {
            if (i == j) {
                net->cross_head_weights[i][j] = 1.0f;
            } else {
                net->cross_head_weights[i][j] = 0.1f * randn();
            }
        }
    }
    
    return net;
}

EXPORT void destroy_hyper_amn(HyperAMN *net) {
    if (!net) return;
    
    for (int h = 0; h < net->num_heads; h++) {
        free(net->A_rigidity[h]);
        free(net->B_fluidity[h]);
        free(net->Memory_manifolds[h]);
        free(net->Key_weights[h]);
        free(net->Value_weights[h]);
        free(net->Query_weights[h]);
        free(net->hidden_states[h]);
        free(net->cross_head_weights[h]);
    }
    
    free(net->A_rigidity);
    free(net->B_fluidity);
    free(net->Memory_manifolds);
    free(net->Key_weights);
    free(net->Value_weights);
    free(net->Query_weights);
    free(net->hidden_states);
    free(net->cross_head_weights);
    
    free(net->manifold_decay);
    free(net->Lambda_diag);
    free(net->B_input);
    free(net->C_output);
    free(net->D_skip);
    free(net->gate_weights);
    free(net->head_activations);
    free(net->merged_hidden);
    free(net->lru_state);
    free(net->output_state);
    free(net->avg_manifold_energy);
    
    free(net);
}

// ============================================================================
// FORWARD PASS
// ============================================================================

EXPORT void hyper_amn_forward(HyperAMN *net, const float *input, float *output) {
    if (!net || !input || !output) return;
    
    int H = net->hidden_size;
    int M = net->manifold_size;
    int O = net->output_size;
    int I = net->input_size;
    
    // === Step 1: Compute head gating ===
    float gate_logits[NUM_MANIFOLD_HEADS];
    for (int h = 0; h < net->num_heads; h++) {
        gate_logits[h] = 0.0f;
        for (int i = 0; i < H; i++) {
            gate_logits[h] += net->merged_hidden[i] * net->gate_weights[i * net->num_heads + h];
        }
    }
    
    // Softmax for gate activations
    for (int h = 0; h < net->num_heads; h++) {
        net->head_activations[h] = softmax_single(gate_logits[h], gate_logits, net->num_heads);
    }
    
    // === Step 2: Process each head independently ===
    #pragma omp parallel for if(net->num_heads > 1)
    for (int h = 0; h < net->num_heads; h++) {
        // LRU update for this head
        for (int i = 0; i < H; i++) {
            float input_contrib = 0.0f;
            for (int j = 0; j < I; j++) {
                input_contrib += input[j] * net->B_input[j * H + i];
            }
            net->hidden_states[h][i] = net->Lambda_diag[i] * net->hidden_states[h][i] + input_contrib;
        }
        
        // Liquid Constant dynamics
        for (int i = 0; i < H; i++) {
            float f_x = tanh_act(net->hidden_states[h][i]);
            float adaptive_tau = net->A_rigidity[h][i] + net->B_fluidity[h][i] * fabsf(f_x);
            float target = net->B_fluidity[h][i] * f_x;
            net->hidden_states[h][i] += net->dt * (-adaptive_tau * net->hidden_states[h][i] + target);
        }
        
        // Associative memory write
        float keys[M];
        for (int m = 0; m < M; m++) {
            keys[m] = 0.0f;
            for (int i = 0; i < H; i++) {
                keys[m] += net->hidden_states[h][i] * net->Key_weights[h][i * M + m];
            }
            keys[m] = sigmoid(keys[m]);
        }
        
        // Update manifold
        for (int m = 0; m < M; m++) {
            for (int i = 0; i < H; i++) {
                float value = net->hidden_states[h][i] * net->Value_weights[h][i * M + m];
                net->Memory_manifolds[h][m * H + i] = 
                    net->manifold_decay[h] * net->Memory_manifolds[h][m * H + i] + 
                    keys[m] * value;
            }
        }
        
        // Read from manifold
        float queries[M];
        for (int m = 0; m < M; m++) {
            queries[m] = 0.0f;
            for (int i = 0; i < H; i++) {
                queries[m] += net->hidden_states[h][i] * net->Query_weights[h][i * M + m];
            }
            queries[m] = sigmoid(queries[m]);
        }
        
        // Retrieve and add to hidden state
        for (int i = 0; i < H; i++) {
            float retrieved = 0.0f;
            for (int m = 0; m < M; m++) {
                retrieved += queries[m] * net->Memory_manifolds[h][m * H + i];
            }
            net->hidden_states[h][i] = 0.7f * net->hidden_states[h][i] + 0.3f * tanh_act(retrieved);
        }
    }
    
    // === Step 3: Cross-head communication ===
    float temp_hidden[NUM_MANIFOLD_HEADS][H];
    for (int h = 0; h < net->num_heads; h++) {
        for (int i = 0; i < H; i++) {
            temp_hidden[h][i] = 0.0f;
            for (int h2 = 0; h2 < net->num_heads; h2++) {
                temp_hidden[h][i] += net->cross_head_weights[h][h2] * net->hidden_states[h2][i];
            }
        }
    }
    
    // Copy back
    for (int h = 0; h < net->num_heads; h++) {
        for (int i = 0; i < H; i++) {
            net->hidden_states[h][i] = temp_hidden[h][i];
        }
    }
    
    // === Step 4: Merge heads with gating ===
    memset(net->merged_hidden, 0, H * sizeof(float));
    for (int h = 0; h < net->num_heads; h++) {
        for (int i = 0; i < H; i++) {
            net->merged_hidden[i] += net->head_activations[h] * net->hidden_states[h][i];
        }
    }
    
    // === Step 5: Output projection ===
    for (int i = 0; i < O; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < H; j++) {
            output[i] += net->merged_hidden[j] * net->C_output[j * O + i];
        }
        // Skip connection
        for (int j = 0; j < I; j++) {
            output[i] += input[j] * net->D_skip[j * O + i];
        }
    }
    
    memcpy(net->output_state, output, O * sizeof(float));
    
    // === Update statistics ===
    for (int h = 0; h < net->num_heads; h++) {
        float energy = 0.0f;
        for (int m = 0; m < M; m++) {
            for (int i = 0; i < H; i++) {
                float val = net->Memory_manifolds[h][m * H + i];
                energy += val * val;
            }
        }
        net->avg_manifold_energy[h] = energy / (M * H);
    }
    
    float cross_flow = 0.0f;
    for (int i = 0; i < net->num_heads; i++) {
        for (int j = 0; j < net->num_heads; j++) {
            if (i != j) {
                cross_flow += fabsf(net->cross_head_weights[i][j]);
            }
        }
    }
    net->avg_cross_head_flow = cross_flow / (net->num_heads * (net->num_heads - 1));
}

// ============================================================================
// TRAINING
// ============================================================================

EXPORT float hyper_amn_train(HyperAMN *net, const float *input, const float *target) {
    if (!net || !input || !target) return -1.0f;
    
    // Forward pass
    float output[net->output_size];
    hyper_amn_forward(net, input, output);
    
    // Compute loss (MSE)
    float loss = 0.0f;
    for (int i = 0; i < net->output_size; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    loss /= net->output_size;
    
    // Simple gradient descent (placeholder for full backprop)
    // In a real implementation, this would use proper backpropagation
    float gradient[net->output_size];
    for (int i = 0; i < net->output_size; i++) {
        gradient[i] = 2.0f * (output[i] - target[i]) / net->output_size;
    }
    
    // Update output weights
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < net->output_size; j++) {
            float grad = gradient[j] * net->merged_hidden[i];
            grad = clip_value(grad, -net->gradient_clip_norm, net->gradient_clip_norm);
            net->C_output[i * net->output_size + j] -= net->learning_rate * grad;
        }
    }
    
    net->training_steps++;
    net->last_loss = loss;
    
    return loss;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

EXPORT void hyper_amn_reset_memory(HyperAMN *net) {
    if (!net) return;
    
    for (int h = 0; h < net->num_heads; h++) {
        memset(net->hidden_states[h], 0, net->hidden_size * sizeof(float));
        memset(net->Memory_manifolds[h], 0, net->manifold_size * net->hidden_size * sizeof(float));
    }
    
    memset(net->merged_hidden, 0, net->hidden_size * sizeof(float));
    memset(net->lru_state, 0, net->hidden_size * sizeof(float));
    memset(net->head_activations, 0, net->num_heads * sizeof(float));
}

EXPORT void hyper_amn_get_head_activations(HyperAMN *net, float *activations) {
    if (!net || !activations) return;
    memcpy(activations, net->head_activations, net->num_heads * sizeof(float));
}

EXPORT float hyper_amn_get_manifold_energy(HyperAMN *net, int head_idx) {
    if (!net || head_idx < 0 || head_idx >= net->num_heads) return 0.0f;
    return net->avg_manifold_energy[head_idx];
}

EXPORT float hyper_amn_get_cross_head_flow(HyperAMN *net) {
    return net ? net->avg_cross_head_flow : 0.0f;
}

EXPORT void hyper_amn_print_info(HyperAMN *net) {
    if (!net) return;
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║           HYPER-AMN: MULTI-HEAD MANIFOLDS                     ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Architecture: %d -> %d -> %d\n", 
           net->input_size, net->hidden_size, net->output_size);
    printf("Manifold Heads: %d (size %d each)\n\n", net->num_heads, net->manifold_size);
    
    printf("Head Activations:\n");
    for (int h = 0; h < net->num_heads; h++) {
        const char *head_name = (h == 0) ? "Spatial" : 
                                (h == 1) ? "Emotional" : "Logical";
        printf("  %s: %.4f (Energy: %.6f)\n", 
               head_name, net->head_activations[h], net->avg_manifold_energy[h]);
    }
    
    printf("\nCross-Head Flow: %.6f\n", net->avg_cross_head_flow);
    printf("Training Steps: %d\n", net->training_steps);
    printf("Last Loss: %.6f\n\n", net->last_loss);
}