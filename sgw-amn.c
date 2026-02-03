/**
 * SPARSE GLOBAL WORKSPACE - ADAPTIVE MEMORY NETWORK (SGW-AMN)
 * 
 * Inspired by Global Workspace Theory of consciousness:
 * - Thousands of neurons compete to access a tiny "conscious" bottleneck
 * - Only the most vital information survives compression
 * - Creates high-level "conscious" summary of data
 * 
 * Key Innovation: Attention by Compression - forces the model to extract
 * only the most essential features through competitive routing
 * 
 * Compile to DLL/SO:
 * Windows: gcc -shared -o sgw-amn.dll sgw-amn.c -lm -O3 -fopenmp
 * Linux:   gcc -shared -fPIC -o sgw-amn.so sgw-amn.c -lm -O3 -fopenmp
 * Mac:     gcc -shared -fPIC -o sgw-amn.dylib sgw-amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp
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

typedef struct {
    int input_size;
    int pre_bottleneck_size;   // Large hidden layer (e.g., 256)
    int workspace_size;        // Tiny bottleneck (e.g., 16)
    int post_bottleneck_size;  // Large hidden layer after bottleneck
    int output_size;
    int manifold_size;
    
    // === PRE-BOTTLENECK LAYER ===
    // This layer processes input richly
    float *W_input_to_pre;     // [input_size × pre_bottleneck_size]
    float *pre_hidden_state;   // [pre_bottleneck_size]
    
    // === COMPETITIVE ROUTING TO WORKSPACE ===
    // Neurons compete to access the sparse workspace
    float *competition_weights; // [pre_bottleneck_size × workspace_size]
    float *competition_scores;  // [pre_bottleneck_size] - how badly each wants access
    float *routing_probs;       // [pre_bottleneck_size] - probability of being routed
    int *winners;               // [workspace_size] - indices of winning neurons
    float competition_temp;     // Temperature for competitive softmax
    
    // === GLOBAL WORKSPACE (BOTTLENECK) ===
    // Only the most important information passes through
    float *workspace_state;     // [workspace_size] - the "conscious" state
    float *workspace_importance; // [workspace_size] - importance of each slot
    
    // === WORKSPACE MEMORY ===
    // Small but crucial memory in the bottleneck
    float *workspace_memory;    // [workspace_size × workspace_size] - internal workspace memory
    float workspace_mem_decay;
    
    // === POST-BOTTLENECK EXPANSION ===
    float *W_workspace_to_post; // [workspace_size × post_bottleneck_size]
    float *post_hidden_state;   // [post_bottleneck_size]
    
    // === ASSOCIATIVE MEMORY MANIFOLD ===
    float *Memory_manifold;     // [manifold_size × post_bottleneck_size]
    float *Key_weights;         // [post_bottleneck_size × manifold_size]
    float *Value_weights;       // [post_bottleneck_size × manifold_size]
    float *Query_weights;       // [post_bottleneck_size × manifold_size]
    float memory_decay;
    
    // === OUTPUT LAYER ===
    float *W_output;            // [post_bottleneck_size × output_size]
    float *output_state;        // [output_size]
    
    // === LIQUID CONSTANT FOR WORKSPACE ===
    float *A_rigidity;          // [workspace_size]
    float *B_fluidity;          // [workspace_size]
    float dt;
    
    // === TRAINING PARAMETERS ===
    float learning_rate;
    float sparsity_penalty;     // Penalty for using too many workspace slots
    int training_steps;
    float last_loss;
    
    // === STATISTICS ===
    float avg_workspace_sparsity; // How many workspace slots are active
    float avg_competition_entropy; // How competitive the routing is
    float avg_manifold_energy;
    float information_bottleneck_rate; // Compression ratio
    
} SGWNetwork;

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

static inline float tanh_act(float x) {
    return tanhf(x);
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float gelu(float x) {
    // Approximation of GELU
    return 0.5f * x * (1.0f + tanhf(0.797884560803f * (x + 0.044715f * x * x * x)));
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

// Compute softmax with temperature
static void softmax_with_temp(float *out, const float *in, int n, float temperature) {
    float max_val = in[0];
    for (int i = 1; i < n; i++) {
        if (in[i] > max_val) max_val = in[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf((in[i] - max_val) / temperature);
        sum += out[i];
    }
    
    for (int i = 0; i < n; i++) {
        out[i] /= sum;
    }
}

// Compute entropy of a probability distribution
static float compute_entropy(const float *probs, int n) {
    float entropy = 0.0f;
    for (int i = 0; i < n; i++) {
        if (probs[i] > 1e-10f) {
            entropy -= probs[i] * logf(probs[i]);
        }
    }
    return entropy;
}

// ============================================================================
// NETWORK CREATION
// ============================================================================

EXPORT SGWNetwork* create_sgw_amn(int input_size, int pre_bottleneck_size,
                                   int workspace_size, int post_bottleneck_size,
                                   int output_size, int manifold_size,
                                   float learning_rate) {
    SGWNetwork *net = (SGWNetwork*)calloc(1, sizeof(SGWNetwork));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->pre_bottleneck_size = pre_bottleneck_size;
    net->workspace_size = workspace_size;
    net->post_bottleneck_size = post_bottleneck_size;
    net->output_size = output_size;
    net->manifold_size = manifold_size;
    net->learning_rate = learning_rate;
    net->competition_temp = 0.5f;  // Low temp = more competitive
    net->sparsity_penalty = 0.01f;
    net->dt = 0.1f;
    net->memory_decay = 0.995f;
    net->workspace_mem_decay = 0.98f;
    net->training_steps = 0;
    
    // Allocate layers
    net->W_input_to_pre = (float*)malloc(input_size * pre_bottleneck_size * sizeof(float));
    net->pre_hidden_state = (float*)calloc(pre_bottleneck_size, sizeof(float));
    
    net->competition_weights = (float*)malloc(pre_bottleneck_size * workspace_size * sizeof(float));
    net->competition_scores = (float*)calloc(pre_bottleneck_size, sizeof(float));
    net->routing_probs = (float*)calloc(pre_bottleneck_size, sizeof(float));
    net->winners = (int*)malloc(workspace_size * sizeof(int));
    
    net->workspace_state = (float*)calloc(workspace_size, sizeof(float));
    net->workspace_importance = (float*)calloc(workspace_size, sizeof(float));
    net->workspace_memory = (float*)calloc(workspace_size * workspace_size, sizeof(float));
    
    net->A_rigidity = (float*)malloc(workspace_size * sizeof(float));
    net->B_fluidity = (float*)malloc(workspace_size * sizeof(float));
    
    net->W_workspace_to_post = (float*)malloc(workspace_size * post_bottleneck_size * sizeof(float));
    net->post_hidden_state = (float*)calloc(post_bottleneck_size, sizeof(float));
    
    net->Memory_manifold = (float*)calloc(manifold_size * post_bottleneck_size, sizeof(float));
    net->Key_weights = (float*)malloc(post_bottleneck_size * manifold_size * sizeof(float));
    net->Value_weights = (float*)malloc(post_bottleneck_size * manifold_size * sizeof(float));
    net->Query_weights = (float*)malloc(post_bottleneck_size * manifold_size * sizeof(float));
    
    net->W_output = (float*)malloc(post_bottleneck_size * output_size * sizeof(float));
    net->output_state = (float*)calloc(output_size, sizeof(float));
    
    // Initialize weights
    srand(time(NULL));
    
    float scale_input = sqrtf(2.0f / (input_size + pre_bottleneck_size));
    float scale_comp = sqrtf(2.0f / (pre_bottleneck_size + workspace_size));
    float scale_expand = sqrtf(2.0f / (workspace_size + post_bottleneck_size));
    float scale_output = sqrtf(2.0f / (post_bottleneck_size + output_size));
    float scale_memory = sqrtf(1.0f / post_bottleneck_size);
    
    for (int i = 0; i < input_size * pre_bottleneck_size; i++) {
        net->W_input_to_pre[i] = randn() * scale_input;
    }
    
    for (int i = 0; i < pre_bottleneck_size * workspace_size; i++) {
        net->competition_weights[i] = randn() * scale_comp;
    }
    
    for (int i = 0; i < workspace_size * post_bottleneck_size; i++) {
        net->W_workspace_to_post[i] = randn() * scale_expand;
    }
    
    for (int i = 0; i < post_bottleneck_size * output_size; i++) {
        net->W_output[i] = randn() * scale_output;
    }
    
    for (int i = 0; i < post_bottleneck_size * manifold_size; i++) {
        net->Key_weights[i] = randn() * scale_memory;
        net->Value_weights[i] = randn() * scale_memory;
        net->Query_weights[i] = randn() * scale_memory;
    }
    
    // Initialize LC for workspace
    for (int i = 0; i < workspace_size; i++) {
        net->A_rigidity[i] = 0.5f + 0.3f * ((float)rand() / RAND_MAX);
        net->B_fluidity[i] = 0.2f + 0.3f * ((float)rand() / RAND_MAX);
    }
    
    return net;
}

EXPORT void destroy_sgw_amn(SGWNetwork *net) {
    if (!net) return;
    
    free(net->W_input_to_pre);
    free(net->pre_hidden_state);
    free(net->competition_weights);
    free(net->competition_scores);
    free(net->routing_probs);
    free(net->winners);
    free(net->workspace_state);
    free(net->workspace_importance);
    free(net->workspace_memory);
    free(net->A_rigidity);
    free(net->B_fluidity);
    free(net->W_workspace_to_post);
    free(net->post_hidden_state);
    free(net->Memory_manifold);
    free(net->Key_weights);
    free(net->Value_weights);
    free(net->Query_weights);
    free(net->W_output);
    free(net->output_state);
    
    free(net);
}

// ============================================================================
// FORWARD PASS
// ============================================================================

EXPORT void sgw_amn_forward(SGWNetwork *net, const float *input, float *output) {
    if (!net || !input || !output) return;
    
    int I = net->input_size;
    int PRE = net->pre_bottleneck_size;
    int WS = net->workspace_size;
    int POST = net->post_bottleneck_size;
    int O = net->output_size;
    int M = net->manifold_size;
    
    // === Step 1: Pre-bottleneck processing ===
    for (int i = 0; i < PRE; i++) {
        net->pre_hidden_state[i] = 0.0f;
        for (int j = 0; j < I; j++) {
            net->pre_hidden_state[i] += input[j] * net->W_input_to_pre[j * PRE + i];
        }
        net->pre_hidden_state[i] = gelu(net->pre_hidden_state[i]);
    }
    
    // === Step 2: Competitive routing to workspace ===
    // Compute competition scores - how badly each neuron wants workspace access
    for (int i = 0; i < PRE; i++) {
        net->competition_scores[i] = 0.0f;
        for (int j = 0; j < WS; j++) {
            net->competition_scores[i] += fabsf(net->pre_hidden_state[i] * 
                                                 net->competition_weights[i * WS + j]);
        }
    }
    
    // Convert to routing probabilities with temperature
    softmax_with_temp(net->routing_probs, net->competition_scores, PRE, net->competition_temp);
    
    // Select top-k winners (k = workspace_size)
    // Simple greedy selection
    float temp_probs[PRE];
    memcpy(temp_probs, net->routing_probs, PRE * sizeof(float));
    
    for (int w = 0; w < WS; w++) {
        int best_idx = 0;
        float best_prob = temp_probs[0];
        for (int i = 1; i < PRE; i++) {
            if (temp_probs[i] > best_prob) {
                best_prob = temp_probs[i];
                best_idx = i;
            }
        }
        net->winners[w] = best_idx;
        temp_probs[best_idx] = -1.0f; // Mark as used
    }
    
    // === Step 3: Route to workspace ===
    memset(net->workspace_state, 0, WS * sizeof(float));
    for (int w = 0; w < WS; w++) {
        int winner_idx = net->winners[w];
        for (int j = 0; j < WS; j++) {
            net->workspace_state[j] += net->pre_hidden_state[winner_idx] * 
                                        net->competition_weights[winner_idx * WS + j];
        }
    }
    
    // === Step 4: Workspace internal processing with LC dynamics ===
    // Apply workspace memory
    float workspace_mem_read[WS];
    for (int i = 0; i < WS; i++) {
        workspace_mem_read[i] = 0.0f;
        for (int j = 0; j < WS; j++) {
            workspace_mem_read[i] += net->workspace_memory[i * WS + j] * net->workspace_state[j];
        }
    }
    
    // Update workspace with LC dynamics
    for (int i = 0; i < WS; i++) {
        float combined = net->workspace_state[i] + workspace_mem_read[i];
        float f_x = tanh_act(combined);
        float adaptive_tau = net->A_rigidity[i] + net->B_fluidity[i] * fabsf(f_x);
        float target = net->B_fluidity[i] * f_x;
        net->workspace_state[i] += net->dt * (-adaptive_tau * net->workspace_state[i] + target);
        
        // Compute importance
        net->workspace_importance[i] = fabsf(net->workspace_state[i]);
    }
    
    // Update workspace memory
    for (int i = 0; i < WS; i++) {
        for (int j = 0; j < WS; j++) {
            net->workspace_memory[i * WS + j] = 
                net->workspace_mem_decay * net->workspace_memory[i * WS + j] +
                (1.0f - net->workspace_mem_decay) * net->workspace_state[i] * net->workspace_state[j];
        }
    }
    
    // === Step 5: Expand from workspace ===
    for (int i = 0; i < POST; i++) {
        net->post_hidden_state[i] = 0.0f;
        for (int j = 0; j < WS; j++) {
            net->post_hidden_state[i] += net->workspace_state[j] * 
                                          net->W_workspace_to_post[j * POST + i];
        }
        net->post_hidden_state[i] = tanh_act(net->post_hidden_state[i]);
    }
    
    // === Step 6: Associative memory manifold ===
    // Write
    float keys[M];
    for (int m = 0; m < M; m++) {
        keys[m] = 0.0f;
        for (int i = 0; i < POST; i++) {
            keys[m] += net->post_hidden_state[i] * net->Key_weights[i * M + m];
        }
        keys[m] = sigmoid(keys[m]);
    }
    
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < POST; i++) {
            float value = net->post_hidden_state[i] * net->Value_weights[i * M + m];
            net->Memory_manifold[m * POST + i] = 
                net->memory_decay * net->Memory_manifold[m * POST + i] + keys[m] * value;
        }
    }
    
    // Read
    float queries[M];
    for (int m = 0; m < M; m++) {
        queries[m] = 0.0f;
        for (int i = 0; i < POST; i++) {
            queries[m] += net->post_hidden_state[i] * net->Query_weights[i * M + m];
        }
        queries[m] = sigmoid(queries[m]);
    }
    
    for (int i = 0; i < POST; i++) {
        float retrieved = 0.0f;
        for (int m = 0; m < M; m++) {
            retrieved += queries[m] * net->Memory_manifold[m * POST + i];
        }
        net->post_hidden_state[i] = 0.7f * net->post_hidden_state[i] + 0.3f * tanh_act(retrieved);
    }
    
    // === Step 7: Output ===
    for (int i = 0; i < O; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < POST; j++) {
            output[i] += net->post_hidden_state[j] * net->W_output[j * O + i];
        }
    }
    
    memcpy(net->output_state, output, O * sizeof(float));
    
    // === Update statistics ===
    // Workspace sparsity
    int active_count = 0;
    for (int i = 0; i < WS; i++) {
        if (fabsf(net->workspace_state[i]) > 0.1f) active_count++;
    }
    net->avg_workspace_sparsity = (float)active_count / WS;
    
    // Competition entropy
    net->avg_competition_entropy = compute_entropy(net->routing_probs, PRE);
    
    // Manifold energy
    float energy = 0.0f;
    for (int i = 0; i < M * POST; i++) {
        energy += net->Memory_manifold[i] * net->Memory_manifold[i];
    }
    net->avg_manifold_energy = energy / (M * POST);
    
    // Information bottleneck rate
    net->information_bottleneck_rate = (float)WS / PRE;
}

// ============================================================================
// TRAINING
// ============================================================================

EXPORT float sgw_amn_train(SGWNetwork *net, const float *input, const float *target) {
    if (!net || !input || !target) return -1.0f;
    
    float output[net->output_size];
    sgw_amn_forward(net, input, output);
    
    // Compute loss
    float loss = 0.0f;
    for (int i = 0; i < net->output_size; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    loss /= net->output_size;
    
    // Add sparsity penalty
    float sparsity_loss = 0.0f;
    for (int i = 0; i < net->workspace_size; i++) {
        sparsity_loss += fabsf(net->workspace_state[i]);
    }
    loss += net->sparsity_penalty * sparsity_loss;
    
    // Simple gradient update (placeholder)
    float grad_output[net->output_size];
    for (int i = 0; i < net->output_size; i++) {
        grad_output[i] = 2.0f * (output[i] - target[i]) / net->output_size;
    }
    
    for (int i = 0; i < net->post_bottleneck_size; i++) {
        for (int j = 0; j < net->output_size; j++) {
            float grad = clip_value(grad_output[j] * net->post_hidden_state[i], -10.0f, 10.0f);
            net->W_output[i * net->output_size + j] -= net->learning_rate * grad;
        }
    }
    
    net->training_steps++;
    net->last_loss = loss;
    
    return loss;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

EXPORT void sgw_amn_reset_memory(SGWNetwork *net) {
    if (!net) return;
    memset(net->workspace_state, 0, net->workspace_size * sizeof(float));
    memset(net->workspace_memory, 0, net->workspace_size * net->workspace_size * sizeof(float));
    memset(net->Memory_manifold, 0, net->manifold_size * net->post_bottleneck_size * sizeof(float));
}

EXPORT float sgw_amn_get_workspace_sparsity(SGWNetwork *net) {
    return net ? net->avg_workspace_sparsity : 0.0f;
}

EXPORT float sgw_amn_get_competition_entropy(SGWNetwork *net) {
    return net ? net->avg_competition_entropy : 0.0f;
}

EXPORT float sgw_amn_get_bottleneck_rate(SGWNetwork *net) {
    return net ? net->information_bottleneck_rate : 0.0f;
}

EXPORT void sgw_amn_print_info(SGWNetwork *net) {
    if (!net) return;
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║    SPARSE GLOBAL WORKSPACE - AMN (SGW-AMN)                    ║\n");
    printf("║    Competitive Attention & Sparse Reasoning                   ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Architecture: %d -> %d -> [%d workspace] -> %d -> %d\n",
           net->input_size, net->pre_bottleneck_size, net->workspace_size,
           net->post_bottleneck_size, net->output_size);
    
    printf("\nGlobal Workspace:\n");
    printf("  Sparsity:              %.2f%% active\n", net->avg_workspace_sparsity * 100);
    printf("  Competition Entropy:   %.4f (high = competitive)\n", net->avg_competition_entropy);
    printf("  Bottleneck Ratio:      %.4f (%d/%d)\n", 
           net->information_bottleneck_rate, net->workspace_size, net->pre_bottleneck_size);
    
    printf("\nMemory:\n");
    printf("  Manifold Energy:       %.6f\n", net->avg_manifold_energy);
    
    printf("\nTraining:\n");
    printf("  Steps:                 %d\n", net->training_steps);
    printf("  Last Loss:             %.6f\n\n", net->last_loss);
}