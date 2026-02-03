/**
 * DUAL-TRACK PERSISTENCE NETWORK (DTPN) - C LIBRARY
 * 
 * Combines three memory persistence mechanisms:
 * 
 * CONCEPT 1: Memory-Preserving Activation
 *   y(t) = activation(W·x(t) + U·h(t-1)) + β·y(t-1)
 *   Creates a mathematical "echo" of past computations
 *   β (Beta): Memory preservation factor (typically 0.3)
 * 
 * CONCEPT 2: Stateful Neurons
 *   memory(t) = (1-α)·memory(t-1) + α·new_info
 *   Persistent internal state with exponential decay
 *   α (Alpha): Memory update rate (typically 0.1)
 * 
 * CONCEPT 3: Global Memory Matrix
 *   M_t = M_{t-1} + σ(x_t·K^T)V
 *   External storage for long-term context
 *   Acts as a shared memory whiteboard
 * 
 * Compile Instructions:
 *   Windows: gcc -shared -o dtpn.dll dtpn.c -lm -O3
 *   Linux:   gcc -shared -fPIC -o dtpn.so dtpn.c -lm -O3
 *   Mac:     gcc -shared -fPIC -o dtpn.dylib dtpn.c -lm -O3
 * 
 * License: GPL V3.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

// ============================================================================
// NETWORK STRUCTURE
// ============================================================================

typedef struct {
    // Architecture dimensions
    int input_size;
    int hidden_size;
    int output_size;
    int memory_matrix_size;  // Size of global memory matrix
    
    // === CONCEPT 1: Memory-Preserving Activation ===
    float *W_input;           // Input weights [input_size × hidden_size]
    float *U_recurrent;       // Recurrent weights [hidden_size × hidden_size]
    float *output_echo;       // Previous output y(t-1) [hidden_size]
    float beta;               // Memory preservation factor (0.0-1.0)
    
    // === CONCEPT 2: Stateful Neurons ===
    float *neuron_memory;     // Internal state per neuron [hidden_size]
    float alpha;              // Memory update rate (0.0-1.0)
    
    // === CONCEPT 3: Global Memory Matrix ===
    float *Memory_matrix;     // M_t [memory_matrix_size × hidden_size]
    float *Key_weights;       // K [hidden_size × memory_matrix_size]
    float *Value_weights;     // V [memory_matrix_size × hidden_size]
    float memory_decay;       // Decay factor for memory matrix
    
    // Output projection
    float *W_output;          // [hidden_size × output_size]
    float *b_output;          // [output_size]
    
    // Current states
    float *hidden_state;      // h(t) [hidden_size]
    float *output_state;      // Current output [output_size]
    
    // Training parameters
    float learning_rate;
    int training_steps;
    float last_loss;
    
    // Statistics
    float avg_echo_magnitude;
    float avg_memory_magnitude;
    float avg_matrix_energy;
    
} DTPNetwork;

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

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static float randn(void) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * M_PI * u2);
}

static float clip(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// ============================================================================
// NETWORK CREATION / DESTRUCTION
// ============================================================================

EXPORT DTPNetwork* create_dtpn(int input_size, int hidden_size, int output_size,
                                int memory_matrix_size, float learning_rate) {
    DTPNetwork *net = (DTPNetwork*)calloc(1, sizeof(DTPNetwork));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    net->memory_matrix_size = memory_matrix_size;
    net->learning_rate = learning_rate;
    
    // Default hyperparameters
    net->beta = 0.3f;           // Memory preservation factor
    net->alpha = 0.1f;          // Memory update rate
    net->memory_decay = 0.99f;  // Global memory decay
    
    // Allocate Concept 1: Memory-Preserving Activation
    net->W_input = (float*)malloc(input_size * hidden_size * sizeof(float));
    net->U_recurrent = (float*)malloc(hidden_size * hidden_size * sizeof(float));
    net->output_echo = (float*)calloc(hidden_size, sizeof(float));
    
    // Allocate Concept 2: Stateful Neurons
    net->neuron_memory = (float*)calloc(hidden_size, sizeof(float));
    
    // Allocate Concept 3: Global Memory Matrix
    net->Memory_matrix = (float*)calloc(memory_matrix_size * hidden_size, sizeof(float));
    net->Key_weights = (float*)malloc(hidden_size * memory_matrix_size * sizeof(float));
    net->Value_weights = (float*)malloc(memory_matrix_size * hidden_size * sizeof(float));
    
    // Output projection
    net->W_output = (float*)malloc(hidden_size * output_size * sizeof(float));
    net->b_output = (float*)calloc(output_size, sizeof(float));
    
    // States
    net->hidden_state = (float*)calloc(hidden_size, sizeof(float));
    net->output_state = (float*)calloc(output_size, sizeof(float));
    
    // Initialize weights using Xavier/Glorot initialization
    srand(time(NULL));
    
    float scale_input = sqrtf(2.0f / (input_size + hidden_size));
    float scale_recurrent = sqrtf(2.0f / (2.0f * hidden_size));
    float scale_output = sqrtf(2.0f / (hidden_size + output_size));
    float scale_memory = sqrtf(1.0f / hidden_size);
    
    // Initialize W_input
    for (int i = 0; i < input_size * hidden_size; i++) {
        net->W_input[i] = randn() * scale_input;
    }
    
    // Initialize U_recurrent
    for (int i = 0; i < hidden_size * hidden_size; i++) {
        net->U_recurrent[i] = randn() * scale_recurrent;
    }
    
    // Initialize output weights
    for (int i = 0; i < hidden_size * output_size; i++) {
        net->W_output[i] = randn() * scale_output;
    }
    
    // Initialize memory matrix weights
    for (int i = 0; i < hidden_size * memory_matrix_size; i++) {
        net->Key_weights[i] = randn() * scale_memory;
    }
    
    for (int i = 0; i < memory_matrix_size * hidden_size; i++) {
        net->Value_weights[i] = randn() * scale_memory;
    }
    
    return net;
}

EXPORT void destroy_dtpn(DTPNetwork *net) {
    if (!net) return;
    
    free(net->W_input);
    free(net->U_recurrent);
    free(net->output_echo);
    free(net->neuron_memory);
    free(net->Memory_matrix);
    free(net->Key_weights);
    free(net->Value_weights);
    free(net->W_output);
    free(net->b_output);
    free(net->hidden_state);
    free(net->output_state);
    free(net);
}

// ============================================================================
// FORWARD PASS
// ============================================================================

EXPORT void dtpn_forward(DTPNetwork *net, const float *input, float *output) {
    if (!net || !input || !output) return;
    
    int H = net->hidden_size;
    int I = net->input_size;
    int O = net->output_size;
    int M = net->memory_matrix_size;
    
    // === CONCEPT 1: Memory-Preserving Activation ===
    // y(t) = activation(W·x(t) + U·h(t-1)) + β·y(t-1)
    
    float *pre_activation = (float*)calloc(H, sizeof(float));
    
    // Compute W·x(t)
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < I; i++) {
            pre_activation[h] += net->W_input[i * H + h] * input[i];
        }
    }
    
    // Add U·h(t-1)
    for (int h = 0; h < H; h++) {
        for (int h2 = 0; h2 < H; h2++) {
            pre_activation[h] += net->U_recurrent[h2 * H + h] * net->hidden_state[h2];
        }
    }
    
    // Apply activation and add echo: y(t) = activation(...) + β·y(t-1)
    float *new_output = (float*)calloc(H, sizeof(float));
    for (int h = 0; h < H; h++) {
        new_output[h] = tanh_act(pre_activation[h]) + net->beta * net->output_echo[h];
    }
    
    // === CONCEPT 2: Stateful Neurons ===
    // memory(t) = (1-α)·memory(t-1) + α·new_info
    
    for (int h = 0; h < H; h++) {
        net->neuron_memory[h] = (1.0f - net->alpha) * net->neuron_memory[h] 
                               + net->alpha * new_output[h];
    }
    
    // === CONCEPT 3: Global Memory Matrix ===
    // M_t = M_{t-1} + σ(x_t·K^T)V
    
    // Compute keys: x_t·K^T
    float *keys = (float*)calloc(M, sizeof(float));
    for (int m = 0; m < M; m++) {
        for (int h = 0; h < H; h++) {
            keys[m] += new_output[h] * net->Key_weights[h * M + m];
        }
        keys[m] = sigmoid(keys[m]);  // Apply sigmoid activation
    }
    
    // Update memory matrix: M_t = decay·M_{t-1} + keys·V
    for (int m = 0; m < M; m++) {
        for (int h = 0; h < H; h++) {
            int idx = m * H + h;
            net->Memory_matrix[idx] = net->memory_decay * net->Memory_matrix[idx]
                                     + keys[m] * net->Value_weights[idx];
        }
    }
    
    // Read from memory matrix and combine with neuron state
    float *memory_read = (float*)calloc(H, sizeof(float));
    for (int h = 0; h < H; h++) {
        for (int m = 0; m < M; m++) {
            memory_read[h] += net->Memory_matrix[m * H + h];
        }
        memory_read[h] /= (float)M;  // Average
    }
    
    // Combine all three concepts for final hidden state
    for (int h = 0; h < H; h++) {
        net->hidden_state[h] = new_output[h] + net->neuron_memory[h] + memory_read[h];
        net->hidden_state[h] = tanh_act(net->hidden_state[h]);  // Final activation
    }
    
    // Update output echo for next timestep
    memcpy(net->output_echo, new_output, H * sizeof(float));
    
    // === OUTPUT PROJECTION ===
    for (int o = 0; o < O; o++) {
        net->output_state[o] = net->b_output[o];
        for (int h = 0; h < H; h++) {
            net->output_state[o] += net->W_output[h * O + o] * net->hidden_state[h];
        }
    }
    
    // Copy to output
    memcpy(output, net->output_state, O * sizeof(float));
    
    // Update statistics
    float echo_sum = 0.0f, memory_sum = 0.0f, matrix_sum = 0.0f;
    for (int h = 0; h < H; h++) {
        echo_sum += fabsf(net->output_echo[h]);
        memory_sum += fabsf(net->neuron_memory[h]);
    }
    for (int i = 0; i < M * H; i++) {
        matrix_sum += fabsf(net->Memory_matrix[i]);
    }
    
    net->avg_echo_magnitude = echo_sum / H;
    net->avg_memory_magnitude = memory_sum / H;
    net->avg_matrix_energy = matrix_sum / (M * H);
    
    // Cleanup
    free(pre_activation);
    free(new_output);
    free(keys);
    free(memory_read);
}

// ============================================================================
// TRAINING
// ============================================================================

EXPORT float dtpn_train(DTPNetwork *net, const float *input, const float *target) {
    if (!net || !input || !target) return -1.0f;
    
    // Forward pass
    float *prediction = (float*)malloc(net->output_size * sizeof(float));
    dtpn_forward(net, input, prediction);
    
    // Compute loss (MSE)
    float loss = 0.0f;
    float *output_error = (float*)malloc(net->output_size * sizeof(float));
    for (int o = 0; o < net->output_size; o++) {
        output_error[o] = prediction[o] - target[o];
        loss += output_error[o] * output_error[o];
    }
    loss /= net->output_size;
    
    // ============================================================================
    // BACKPROPAGATION THROUGH TIME (BPTT)
    // ============================================================================
    
    int H = net->hidden_size;
    int I = net->input_size;
    int O = net->output_size;
    int M = net->memory_matrix_size;
    
    // Allocate gradient buffers
    float *grad_hidden = (float*)calloc(H, sizeof(float));
    float *grad_echo = (float*)calloc(H, sizeof(float));
    float *grad_memory = (float*)calloc(H, sizeof(float));
    float *grad_W_input = (float*)calloc(I * H, sizeof(float));
    float *grad_U_recurrent = (float*)calloc(H * H, sizeof(float));
    float *grad_W_output = (float*)calloc(H * O, sizeof(float));
    float *grad_b_output = (float*)calloc(O, sizeof(float));
    float *grad_Key = (float*)calloc(H * M, sizeof(float));
    float *grad_Value = (float*)calloc(M * H, sizeof(float));
    
    // Step 1: Backprop through output layer
    for (int o = 0; o < O; o++) {
        float grad_out = 2.0f * output_error[o] / O;
        grad_b_output[o] = grad_out;
        
        for (int h = 0; h < H; h++) {
            grad_W_output[h * O + o] = grad_out * net->hidden_state[h];
            grad_hidden[h] += grad_out * net->W_output[h * O + o];
        }
    }
    
    // Step 2: Backprop through hidden state combination
    // hidden = tanh(new_output + neuron_memory + memory_read)
    for (int h = 0; h < H; h++) {
        float tanh_deriv = tanh_derivative(net->hidden_state[h]);
        float grad = grad_hidden[h] * tanh_deriv;
        
        // Gradient flows to all three components
        grad_echo[h] += grad;          // → new_output (Concept 1)
        grad_memory[h] += grad;        // → neuron_memory (Concept 2)
        // grad flows to memory_read (Concept 3) - handled below
    }
    
    // Step 3: Backprop through Concept 1 (Memory-Preserving Activation)
    // y(t) = tanh(pre_activation) + β·y(t-1)
    float *grad_pre_activation = (float*)calloc(H, sizeof(float));
    for (int h = 0; h < H; h++) {
        // Gradient through current activation
        float tanh_val = tanh_act(net->output_echo[h] / (1.0f + net->beta)); // approximate
        grad_pre_activation[h] = grad_echo[h] * tanh_derivative(tanh_val);
        
        // Gradient through echo term (future timestep contribution)
        // This would affect previous timestep's y, but we're doing truncated BPTT
    }
    
    // Step 4: Backprop through recurrent connections
    // pre_activation = W·x + U·h(t-1)
    for (int h = 0; h < H; h++) {
        // Gradient w.r.t. input weights
        for (int i = 0; i < I; i++) {
            grad_W_input[i * H + h] += grad_pre_activation[h] * input[i];
        }
        
        // Gradient w.r.t. recurrent weights (truncated - don't backprop further)
        for (int h2 = 0; h2 < H; h2++) {
            grad_U_recurrent[h2 * H + h] += grad_pre_activation[h] * net->hidden_state[h2];
        }
    }
    
    // Step 5: Backprop through Concept 2 (Stateful Neurons)
    // memory(t) = (1-α)·memory(t-1) + α·new_info
    // We only update based on current gradient (truncated BPTT)
    
    // Step 6: Backprop through Concept 3 (Global Memory Matrix)
    // M_t = decay·M_{t-1} + σ(keys)·V
    // Gradient w.r.t. memory read affects Key and Value weights
    for (int h = 0; h < H; h++) {
        float grad_mem_read = grad_hidden[h] * tanh_derivative(net->hidden_state[h]) / M;
        
        for (int m = 0; m < M; m++) {
            // Gradient w.r.t. Value weights
            float key_activation = sigmoid(0.0f); // Approximate - need to store keys from forward
            grad_Value[m * H + h] += grad_mem_read * key_activation;
            
            // Gradient w.r.t. Key weights (through sigmoid)
            float grad_key = grad_mem_read * net->Value_weights[m * H + h];
            float sig_deriv = sigmoid_derivative(0.0f); // Approximate
            
            for (int h2 = 0; h2 < H; h2++) {
                grad_Key[h2 * M + m] += grad_key * sig_deriv * net->output_echo[h2];
            }
        }
    }
    
    // ============================================================================
    // GRADIENT CLIPPING (prevent exploding gradients)
    // ============================================================================
    
    float grad_norm = 0.0f;
    for (int i = 0; i < I * H; i++) grad_norm += grad_W_input[i] * grad_W_input[i];
    for (int i = 0; i < H * H; i++) grad_norm += grad_U_recurrent[i] * grad_U_recurrent[i];
    for (int i = 0; i < H * O; i++) grad_norm += grad_W_output[i] * grad_W_output[i];
    grad_norm = sqrtf(grad_norm);
    
    float clip_threshold = 10.0f;
    float scale = 1.0f;
    if (grad_norm > clip_threshold) {
        scale = clip_threshold / grad_norm;
    }
    
    // ============================================================================
    // WEIGHT UPDATES
    // ============================================================================
    
    float lr = net->learning_rate;
    
    // Update input weights
    for (int i = 0; i < I * H; i++) {
        net->W_input[i] -= lr * scale * grad_W_input[i];
    }
    
    // Update recurrent weights
    for (int i = 0; i < H * H; i++) {
        net->U_recurrent[i] -= lr * scale * grad_U_recurrent[i];
    }
    
    // Update output weights
    for (int i = 0; i < H * O; i++) {
        net->W_output[i] -= lr * scale * grad_W_output[i];
    }
    
    // Update output bias
    for (int i = 0; i < O; i++) {
        net->b_output[i] -= lr * grad_b_output[i];
    }
    
    // Update memory matrix weights
    for (int i = 0; i < H * M; i++) {
        net->Key_weights[i] -= lr * scale * grad_Key[i] * 0.1f; // Smaller LR for memory
    }
    
    for (int i = 0; i < M * H; i++) {
        net->Value_weights[i] -= lr * scale * grad_Value[i] * 0.1f;
    }
    
    // Cleanup
    free(output_error);
    free(grad_hidden);
    free(grad_echo);
    free(grad_memory);
    free(grad_W_input);
    free(grad_U_recurrent);
    free(grad_W_output);
    free(grad_b_output);
    free(grad_Key);
    free(grad_Value);
    free(grad_pre_activation);
    
    net->last_loss = loss;
    net->training_steps++;
    
    free(prediction);
    return loss;
}

EXPORT float dtpn_train_batch(DTPNetwork *net, const float *inputs, 
                               const float *targets, int batch_size) {
    if (!net || !inputs || !targets || batch_size <= 0) return -1.0f;
    
    float total_loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        const float *input = inputs + b * net->input_size;
        const float *target = targets + b * net->output_size;
        
        float loss = dtpn_train(net, input, target);
        total_loss += loss;
    }
    
    return total_loss / batch_size;
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

EXPORT void dtpn_reset_memory(DTPNetwork *net) {
    if (!net) return;
    
    memset(net->output_echo, 0, net->hidden_size * sizeof(float));
    memset(net->neuron_memory, 0, net->hidden_size * sizeof(float));
    memset(net->Memory_matrix, 0, 
           net->memory_matrix_size * net->hidden_size * sizeof(float));
    memset(net->hidden_state, 0, net->hidden_size * sizeof(float));
    memset(net->output_state, 0, net->output_size * sizeof(float));
}

EXPORT void dtpn_reset_global_memory(DTPNetwork *net) {
    if (!net) return;
    memset(net->Memory_matrix, 0, 
           net->memory_matrix_size * net->hidden_size * sizeof(float));
}

EXPORT void dtpn_get_hidden_state(DTPNetwork *net, float *state_out) {
    if (!net || !state_out) return;
    memcpy(state_out, net->hidden_state, net->hidden_size * sizeof(float));
}

EXPORT void dtpn_set_hidden_state(DTPNetwork *net, const float *state_in) {
    if (!net || !state_in) return;
    memcpy(net->hidden_state, state_in, net->hidden_size * sizeof(float));
}

EXPORT void dtpn_get_memory_matrix(DTPNetwork *net, float *matrix_out) {
    if (!net || !matrix_out) return;
    memcpy(matrix_out, net->Memory_matrix,
           net->memory_matrix_size * net->hidden_size * sizeof(float));
}

// ============================================================================
// PARAMETER GETTERS / SETTERS
// ============================================================================

EXPORT float dtpn_get_beta(DTPNetwork *net) {
    return net ? net->beta : 0.0f;
}

EXPORT void dtpn_set_beta(DTPNetwork *net, float beta) {
    if (net) net->beta = clip(beta, 0.0f, 1.0f);
}

EXPORT float dtpn_get_alpha(DTPNetwork *net) {
    return net ? net->alpha : 0.0f;
}

EXPORT void dtpn_set_alpha(DTPNetwork *net, float alpha) {
    if (net) net->alpha = clip(alpha, 0.0f, 1.0f);
}

EXPORT float dtpn_get_memory_decay(DTPNetwork *net) {
    return net ? net->memory_decay : 0.0f;
}

EXPORT void dtpn_set_memory_decay(DTPNetwork *net, float decay) {
    if (net) net->memory_decay = clip(decay, 0.0f, 1.0f);
}

EXPORT float dtpn_get_learning_rate(DTPNetwork *net) {
    return net ? net->learning_rate : 0.0f;
}

EXPORT void dtpn_set_learning_rate(DTPNetwork *net, float lr) {
    if (net && lr > 0.0f) net->learning_rate = lr;
}

EXPORT int dtpn_get_training_steps(DTPNetwork *net) {
    return net ? net->training_steps : 0;
}

EXPORT float dtpn_get_last_loss(DTPNetwork *net) {
    return net ? net->last_loss : 0.0f;
}

EXPORT float dtpn_get_avg_echo_magnitude(DTPNetwork *net) {
    return net ? net->avg_echo_magnitude : 0.0f;
}

EXPORT float dtpn_get_avg_memory_magnitude(DTPNetwork *net) {
    return net ? net->avg_memory_magnitude : 0.0f;
}

EXPORT float dtpn_get_avg_matrix_energy(DTPNetwork *net) {
    return net ? net->avg_matrix_energy : 0.0f;
}

// ============================================================================
// SAVE / LOAD
// ============================================================================

EXPORT int dtpn_save(DTPNetwork *net, const char *filename) {
    if (!net || !filename) return -1;
    
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;
    
    // Write metadata
    fwrite(&net->input_size, sizeof(int), 1, f);
    fwrite(&net->hidden_size, sizeof(int), 1, f);
    fwrite(&net->output_size, sizeof(int), 1, f);
    fwrite(&net->memory_matrix_size, sizeof(int), 1, f);
    
    // Write hyperparameters
    fwrite(&net->beta, sizeof(float), 1, f);
    fwrite(&net->alpha, sizeof(float), 1, f);
    fwrite(&net->memory_decay, sizeof(float), 1, f);
    fwrite(&net->learning_rate, sizeof(float), 1, f);
    fwrite(&net->training_steps, sizeof(int), 1, f);
    
    // Write weights
    fwrite(net->W_input, sizeof(float), net->input_size * net->hidden_size, f);
    fwrite(net->U_recurrent, sizeof(float), net->hidden_size * net->hidden_size, f);
    fwrite(net->W_output, sizeof(float), net->hidden_size * net->output_size, f);
    fwrite(net->b_output, sizeof(float), net->output_size, f);
    fwrite(net->Key_weights, sizeof(float), net->hidden_size * net->memory_matrix_size, f);
    fwrite(net->Value_weights, sizeof(float), net->memory_matrix_size * net->hidden_size, f);
    
    // Write states
    fwrite(net->output_echo, sizeof(float), net->hidden_size, f);
    fwrite(net->neuron_memory, sizeof(float), net->hidden_size, f);
    fwrite(net->Memory_matrix, sizeof(float), 
           net->memory_matrix_size * net->hidden_size, f);
    fwrite(net->hidden_state, sizeof(float), net->hidden_size, f);
    
    fclose(f);
    return 0;
}

EXPORT int dtpn_load(DTPNetwork *net, const char *filename) {
    if (!net || !filename) return -1;
    
    FILE *f = fopen(filename, "rb");
    if (!f) return -1;
    
    // Read and verify metadata
    int input_size, hidden_size, output_size, memory_matrix_size;
    fread(&input_size, sizeof(int), 1, f);
    fread(&hidden_size, sizeof(int), 1, f);
    fread(&output_size, sizeof(int), 1, f);
    fread(&memory_matrix_size, sizeof(int), 1, f);
    
    if (input_size != net->input_size || hidden_size != net->hidden_size ||
        output_size != net->output_size || 
        memory_matrix_size != net->memory_matrix_size) {
        fclose(f);
        return -2;  // Size mismatch
    }
    
    // Read hyperparameters
    fread(&net->beta, sizeof(float), 1, f);
    fread(&net->alpha, sizeof(float), 1, f);
    fread(&net->memory_decay, sizeof(float), 1, f);
    fread(&net->learning_rate, sizeof(float), 1, f);
    fread(&net->training_steps, sizeof(int), 1, f);
    
    // Read weights
    fread(net->W_input, sizeof(float), net->input_size * net->hidden_size, f);
    fread(net->U_recurrent, sizeof(float), net->hidden_size * net->hidden_size, f);
    fread(net->W_output, sizeof(float), net->hidden_size * net->output_size, f);
    fread(net->b_output, sizeof(float), net->output_size, f);
    fread(net->Key_weights, sizeof(float), net->hidden_size * net->memory_matrix_size, f);
    fread(net->Value_weights, sizeof(float), net->memory_matrix_size * net->hidden_size, f);
    
    // Read states
    fread(net->output_echo, sizeof(float), net->hidden_size, f);
    fread(net->neuron_memory, sizeof(float), net->hidden_size, f);
    fread(net->Memory_matrix, sizeof(float), 
          net->memory_matrix_size * net->hidden_size, f);
    fread(net->hidden_state, sizeof(float), net->hidden_size, f);
    
    fclose(f);
    return 0;
}

// ============================================================================
// INFO / DEBUG
// ============================================================================

EXPORT void dtpn_print_info(DTPNetwork *net) {
    if (!net) return;
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║        DUAL-TRACK PERSISTENCE NETWORK (DTPN) INFO             ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Architecture: %d → %d → %d\n", 
           net->input_size, net->hidden_size, net->output_size);
    printf("Memory Matrix Size: %d\n\n", net->memory_matrix_size);
    
    printf("Memory Mechanisms:\n");
    printf("  [1] Echo Preservation (β):    %.3f\n", net->beta);
    printf("  [2] Stateful Update (α):      %.3f\n", net->alpha);
    printf("  [3] Matrix Decay:              %.4f\n", net->memory_decay);
    
    printf("\nTraining:\n");
    printf("  Learning Rate:                 %.4f\n", net->learning_rate);
    printf("  Training Steps:                %d\n", net->training_steps);
    printf("  Last Loss:                     %.6f\n", net->last_loss);
    
    printf("\nMemory Statistics:\n");
    printf("  Avg Echo Magnitude:            %.6f\n", net->avg_echo_magnitude);
    printf("  Avg Neuron Memory:             %.6f\n", net->avg_memory_magnitude);
    printf("  Avg Matrix Energy:             %.6f\n", net->avg_matrix_energy);
    
    printf("\n");
}