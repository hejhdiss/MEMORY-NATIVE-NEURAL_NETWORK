/**
 * MEMORY-NATIVE NEURAL NETWORK - EXTENDED C LIBRARY
 * 
 * Implements ALL THREE Memory Concepts:
 * 1. Memory-Preserving Activation: y(t) = activation(W×x(t) + U×h(t-1)) + β×y(t-1)
 * 2. Stateful Neurons: memory(t) = (1-α)×memory(t-1) + α×new_info
 * 3. Learnable Memory Dynamics: Network learns what/when to remember
 * 
 * Compile to DLL/SO:
 * Windows: gcc -shared -o memory_net_extended.dll memory_net_extended.c -lm -O3
 * Linux:   gcc -shared -fPIC -o memory_net_extended.so memory_net_extended.c -lm -O3
 * Mac:     gcc -shared -fPIC -o memory_net_extended.dylib memory_net_extended.c -lm -O3
 * Licensed under GPL V3.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

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
    
    // Standard weights
    float *W_hidden;           // input -> hidden weights
    float *W_output;           // hidden -> output weights
    float *U_recurrent;        // hidden(t-1) -> hidden(t) (LSTM-like)
    float *bias_hidden;
    float *bias_output;
    
    // CONCEPT 3: Learnable Memory Gate Weights
    float *W_memory_gate;      // Learns what to remember
    float *U_memory_gate;      // Gate based on current memory
    float *bias_memory_gate;
    
    float *W_candidate;        // Candidate memory weights
    float *bias_candidate;
    
    // Persistent memory states
    float *hidden_memory;         // Internal memory (Concept 2)
    float *hidden_prev_output;    // Previous output (Concept 1)
    float *hidden_state;          // Current hidden state
    float *hidden_prev_state;     // Previous hidden state (for LSTM-like)
    float *output_state;          // Current output state
    float *output_prev_output;    // Output layer memory preservation
    
    // Memory gate states (Concept 3)
    float *memory_gate_state;     // Current memory gate values [0,1]
    float *candidate_memory;      // Candidate memory values
    
    // Memory parameters
    float beta;              // Memory preservation factor (Concept 1)
    float alpha;             // Memory update rate (Concept 2)
    float output_beta;       // Output layer memory preservation
    float learning_rate;
    
    // Feature flags
    bool use_recurrent;      // Use LSTM-like recurrent connections
    bool use_learnable_gates; // Use learnable memory gates (Concept 3)
    bool use_output_memory;  // Apply memory to output layer
    
    // Partial training masks
    bool *freeze_mask_hidden;
    bool *freeze_mask_output;
    bool *freeze_mask_memory;  // For memory gate weights
    
    // Gradient accumulators (for batch training)
    float *grad_W_hidden;
    float *grad_W_output;
    float *grad_U_recurrent;
    float *grad_W_memory_gate;
    float *grad_W_candidate;
    int grad_accumulation_count;
    
    // Statistics
    int training_steps;
    float last_loss;
    float avg_memory_magnitude;
    float avg_gate_value;
    
} Network;

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

// ============================================================================
// NETWORK CREATION AND DESTRUCTION
// ============================================================================

EXPORT Network* create_network(int input_size, int hidden_size, int output_size,
                               float beta, float alpha, float learning_rate) {
    Network *net = (Network*)calloc(1, sizeof(Network));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    net->beta = beta;
    net->alpha = alpha;
    net->output_beta = 0.2f;  // Default output memory
    net->learning_rate = learning_rate;
    net->training_steps = 0;
    net->last_loss = 0.0f;
    net->grad_accumulation_count = 0;
    
    // Feature flags (all enabled by default)
    net->use_recurrent = true;
    net->use_learnable_gates = true;
    net->use_output_memory = true;
    
    // Allocate standard weights
    int hidden_weights = input_size * hidden_size;
    int output_weights = hidden_size * output_size;
    int recurrent_weights = hidden_size * hidden_size;
    
    net->W_hidden = (float*)malloc(hidden_weights * sizeof(float));
    net->W_output = (float*)malloc(output_weights * sizeof(float));
    net->U_recurrent = (float*)malloc(recurrent_weights * sizeof(float));
    net->bias_hidden = (float*)calloc(hidden_size, sizeof(float));
    net->bias_output = (float*)calloc(output_size, sizeof(float));
    
    // Allocate learnable memory gate weights (Concept 3)
    net->W_memory_gate = (float*)malloc(hidden_weights * sizeof(float));
    net->U_memory_gate = (float*)malloc(recurrent_weights * sizeof(float));
    net->bias_memory_gate = (float*)calloc(hidden_size, sizeof(float));
    
    net->W_candidate = (float*)malloc(hidden_weights * sizeof(float));
    net->bias_candidate = (float*)calloc(hidden_size, sizeof(float));
    
    // Allocate memory states
    net->hidden_memory = (float*)calloc(hidden_size, sizeof(float));
    net->hidden_prev_output = (float*)calloc(hidden_size, sizeof(float));
    net->hidden_state = (float*)calloc(hidden_size, sizeof(float));
    net->hidden_prev_state = (float*)calloc(hidden_size, sizeof(float));
    net->output_state = (float*)calloc(output_size, sizeof(float));
    net->output_prev_output = (float*)calloc(output_size, sizeof(float));
    
    net->memory_gate_state = (float*)calloc(hidden_size, sizeof(float));
    net->candidate_memory = (float*)calloc(hidden_size, sizeof(float));
    
    // Allocate freeze masks (all trainable initially)
    net->freeze_mask_hidden = (bool*)malloc(hidden_weights * sizeof(bool));
    net->freeze_mask_output = (bool*)malloc(output_weights * sizeof(bool));
    net->freeze_mask_memory = (bool*)malloc(hidden_weights * sizeof(bool));
    memset(net->freeze_mask_hidden, 1, hidden_weights * sizeof(bool));
    memset(net->freeze_mask_output, 1, output_weights * sizeof(bool));
    memset(net->freeze_mask_memory, 1, hidden_weights * sizeof(bool));
    
    // Allocate gradient accumulators
    net->grad_W_hidden = (float*)calloc(hidden_weights, sizeof(float));
    net->grad_W_output = (float*)calloc(output_weights, sizeof(float));
    net->grad_U_recurrent = (float*)calloc(recurrent_weights, sizeof(float));
    net->grad_W_memory_gate = (float*)calloc(hidden_weights, sizeof(float));
    net->grad_W_candidate = (float*)calloc(hidden_weights, sizeof(float));
    
    // Xavier/He initialization
    srand(time(NULL));
    float scale_hidden = sqrtf(2.0f / (input_size + hidden_size));
    float scale_output = sqrtf(2.0f / (hidden_size + output_size));
    float scale_recurrent = sqrtf(1.0f / hidden_size);
    
    for (int i = 0; i < hidden_weights; i++) {
        net->W_hidden[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
        net->W_memory_gate[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
        net->W_candidate[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
    }
    
    for (int i = 0; i < output_weights; i++) {
        net->W_output[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_output;
    }
    
    for (int i = 0; i < recurrent_weights; i++) {
        net->U_recurrent[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_recurrent;
        net->U_memory_gate[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_recurrent;
    }
    
    return net;
}

EXPORT void destroy_network(Network *net) {
    if (!net) return;
    
    free(net->W_hidden);
    free(net->W_output);
    free(net->U_recurrent);
    free(net->bias_hidden);
    free(net->bias_output);
    
    free(net->W_memory_gate);
    free(net->U_memory_gate);
    free(net->bias_memory_gate);
    free(net->W_candidate);
    free(net->bias_candidate);
    
    free(net->hidden_memory);
    free(net->hidden_prev_output);
    free(net->hidden_state);
    free(net->hidden_prev_state);
    free(net->output_state);
    free(net->output_prev_output);
    
    free(net->memory_gate_state);
    free(net->candidate_memory);
    
    free(net->freeze_mask_hidden);
    free(net->freeze_mask_output);
    free(net->freeze_mask_memory);
    
    free(net->grad_W_hidden);
    free(net->grad_W_output);
    free(net->grad_U_recurrent);
    free(net->grad_W_memory_gate);
    free(net->grad_W_candidate);
    
    free(net);
}

// ============================================================================
// FORWARD PASS WITH ALL THREE CONCEPTS
// ============================================================================

EXPORT void forward(Network *net, const float *input, float *output) {
    if (!net || !input || !output) return;
    
    // HIDDEN LAYER with ALL THREE memory concepts
    float avg_gate = 0.0f;
    
    for (int h = 0; h < net->hidden_size; h++) {
        // Standard input processing: W × x(t)
        float activation = net->bias_hidden[h];
        for (int i = 0; i < net->input_size; i++) {
            activation += net->W_hidden[h * net->input_size + i] * input[i];
        }
        
        // Add recurrent connection: U × h(t-1) (LSTM-like)
        if (net->use_recurrent) {
            for (int h2 = 0; h2 < net->hidden_size; h2++) {
                activation += net->U_recurrent[h * net->hidden_size + h2] * 
                             net->hidden_prev_state[h2];
            }
        }
        
        // CONCEPT 2: Add internal memory contribution
        activation += net->hidden_memory[h];
        
        // Apply activation function
        float new_value = tanh_act(activation);
        
        // CONCEPT 3: LEARNABLE MEMORY DYNAMICS
        float memory_contribution = 0.0f;
        if (net->use_learnable_gates) {
            // Compute memory gate: decides what to remember
            float gate_input = net->bias_memory_gate[h];
            for (int i = 0; i < net->input_size; i++) {
                gate_input += net->W_memory_gate[h * net->input_size + i] * input[i];
            }
            for (int h2 = 0; h2 < net->hidden_size; h2++) {
                gate_input += net->U_memory_gate[h * net->hidden_size + h2] * 
                             net->hidden_memory[h2];
            }
            float memory_gate = sigmoid(gate_input);
            net->memory_gate_state[h] = memory_gate;
            avg_gate += memory_gate;
            
            // Compute candidate memory
            float candidate = net->bias_candidate[h];
            for (int i = 0; i < net->input_size; i++) {
                candidate += net->W_candidate[h * net->input_size + i] * input[i];
            }
            candidate = tanh_act(candidate);
            net->candidate_memory[h] = candidate;
            
            // Update memory with learned gating
            // new_memory = gate × candidate + (1 - gate) × old_memory
            memory_contribution = memory_gate * candidate + 
                                (1.0f - memory_gate) * net->hidden_memory[h];
        }
        
        // CONCEPT 1: Memory-Preserving Activation - add β × y(t-1)
        float memory_echo = net->beta * net->hidden_prev_output[h];
        float hidden_output = new_value + memory_echo;
        
        // Store states for next iteration
        net->hidden_prev_output[h] = net->hidden_state[h];
        net->hidden_prev_state[h] = net->hidden_state[h];
        net->hidden_state[h] = hidden_output;
        
        // CONCEPT 2: Update internal memory (traditional way)
        if (!net->use_learnable_gates) {
            // memory(t) = (1-α) × memory(t-1) + α × new_info
            net->hidden_memory[h] = (1.0f - net->alpha) * net->hidden_memory[h] + 
                                   net->alpha * new_value;
        } else {
            // Use learned memory update
            net->hidden_memory[h] = memory_contribution;
        }
    }
    
    net->avg_gate_value = avg_gate / net->hidden_size;
    
    // OUTPUT LAYER with optional memory preservation
    for (int o = 0; o < net->output_size; o++) {
        float activation = net->bias_output[o];
        for (int h = 0; h < net->hidden_size; h++) {
            activation += net->W_output[o * net->hidden_size + h] * net->hidden_state[h];
        }
        
        float output_value = tanh_act(activation);
        
        // Optional output memory preservation
        if (net->use_output_memory) {
            output_value += net->output_beta * net->output_prev_output[o];
            net->output_prev_output[o] = net->output_state[o];
        }
        
        net->output_state[o] = output_value;
        output[o] = output_value;
    }
    
    // Update memory magnitude statistic
    float sum_mag = 0.0f;
    for (int h = 0; h < net->hidden_size; h++) {
        sum_mag += fabsf(net->hidden_memory[h]);
    }
    net->avg_memory_magnitude = sum_mag / net->hidden_size;
}

// ============================================================================
// PREDICTION (wrapper for forward)
// ============================================================================

EXPORT void predict(Network *net, const float *input, float *output) {
    forward(net, input, output);
}

// ============================================================================
// TRAINING WITH GRADIENT ACCUMULATION AND PARTIAL TRAINING
// ============================================================================

EXPORT float train(Network *net, const float *input, const float *target) {
    if (!net || !input || !target) return -1.0f;
    
    // Forward pass
    float *output = (float*)malloc(net->output_size * sizeof(float));
    forward(net, input, output);
    
    // Compute loss (MSE)
    float loss = 0.0f;
    float *output_error = (float*)malloc(net->output_size * sizeof(float));
    for (int i = 0; i < net->output_size; i++) {
        output_error[i] = output[i] - target[i];
        loss += output_error[i] * output_error[i];
    }
    loss /= net->output_size;
    net->last_loss = loss;
    
    // Backward pass - output layer
    float *hidden_error = (float*)calloc(net->hidden_size, sizeof(float));
    
    for (int o = 0; o < net->output_size; o++) {
        float delta = output_error[o] * tanh_derivative(net->output_state[o]);
        
        // Update output weights (respecting freeze mask)
        for (int h = 0; h < net->hidden_size; h++) {
            int idx = o * net->hidden_size + h;
            if (net->freeze_mask_output[idx]) {
                float grad = delta * net->hidden_state[h];
                net->W_output[idx] -= net->learning_rate * grad;
            }
            hidden_error[h] += delta * net->W_output[idx];
        }
        
        // Update output bias
        net->bias_output[o] -= net->learning_rate * delta;
    }
    
    // Backward pass - hidden layer with memory concepts
    for (int h = 0; h < net->hidden_size; h++) {
        float delta = hidden_error[h] * tanh_derivative(net->hidden_state[h]);
        
        // Update input->hidden weights (respecting freeze mask)
        for (int i = 0; i < net->input_size; i++) {
            int idx = h * net->input_size + i;
            if (net->freeze_mask_hidden[idx]) {
                float grad = delta * input[i];
                net->W_hidden[idx] -= net->learning_rate * grad;
            }
        }
        
        // Update recurrent weights
        if (net->use_recurrent) {
            for (int h2 = 0; h2 < net->hidden_size; h2++) {
                int idx = h * net->hidden_size + h2;
                float grad = delta * net->hidden_prev_state[h2];
                net->U_recurrent[idx] -= net->learning_rate * grad;
            }
        }
        
        // Update learnable memory gate weights (Concept 3)
        if (net->use_learnable_gates && net->freeze_mask_memory[h * net->input_size]) {
            // Gate gradient
            float gate_delta = delta * sigmoid_derivative(net->memory_gate_state[h]);
            gate_delta *= (net->candidate_memory[h] - net->hidden_memory[h]);
            
            for (int i = 0; i < net->input_size; i++) {
                int idx = h * net->input_size + i;
                float grad = gate_delta * input[i];
                net->W_memory_gate[idx] -= net->learning_rate * grad;
            }
            
            // Candidate memory gradient
            float cand_delta = delta * net->memory_gate_state[h] * 
                              tanh_derivative(net->candidate_memory[h]);
            for (int i = 0; i < net->input_size; i++) {
                int idx = h * net->input_size + i;
                float grad = cand_delta * input[i];
                net->W_candidate[idx] -= net->learning_rate * grad;
            }
        }
        
        // Update hidden bias
        net->bias_hidden[h] -= net->learning_rate * delta;
    }
    
    free(output);
    free(output_error);
    free(hidden_error);
    
    net->training_steps++;
    return loss;
}

// Accumulate gradients without updating weights (for mini-batch training)
EXPORT float accumulate_gradients(Network *net, const float *input, const float *target) {
    if (!net || !input || !target) return -1.0f;
    
    // Forward pass
    float *output = (float*)malloc(net->output_size * sizeof(float));
    forward(net, input, output);
    
    // Compute loss
    float loss = 0.0f;
    float *output_error = (float*)malloc(net->output_size * sizeof(float));
    for (int i = 0; i < net->output_size; i++) {
        output_error[i] = output[i] - target[i];
        loss += output_error[i] * output_error[i];
    }
    loss /= net->output_size;
    
    // Backward pass - accumulate gradients
    float *hidden_error = (float*)calloc(net->hidden_size, sizeof(float));
    
    // Output layer gradients
    for (int o = 0; o < net->output_size; o++) {
        float delta = output_error[o] * tanh_derivative(net->output_state[o]);
        
        for (int h = 0; h < net->hidden_size; h++) {
            int idx = o * net->hidden_size + h;
            if (net->freeze_mask_output[idx]) {
                net->grad_W_output[idx] += delta * net->hidden_state[h];
            }
            hidden_error[h] += delta * net->W_output[idx];
        }
    }
    
    // Hidden layer gradients
    for (int h = 0; h < net->hidden_size; h++) {
        float delta = hidden_error[h] * tanh_derivative(net->hidden_state[h]);
        
        // Input->hidden gradients
        for (int i = 0; i < net->input_size; i++) {
            int idx = h * net->input_size + i;
            if (net->freeze_mask_hidden[idx]) {
                net->grad_W_hidden[idx] += delta * input[i];
            }
        }
        
        // Recurrent gradients
        if (net->use_recurrent) {
            for (int h2 = 0; h2 < net->hidden_size; h2++) {
                int idx = h * net->hidden_size + h2;
                net->grad_U_recurrent[idx] += delta * net->hidden_prev_state[h2];
            }
        }
        
        // Memory gate gradients
        if (net->use_learnable_gates && net->freeze_mask_memory[h * net->input_size]) {
            float gate_delta = delta * sigmoid_derivative(net->memory_gate_state[h]);
            gate_delta *= (net->candidate_memory[h] - net->hidden_memory[h]);
            
            for (int i = 0; i < net->input_size; i++) {
                int idx = h * net->input_size + i;
                net->grad_W_memory_gate[idx] += gate_delta * input[i];
            }
            
            float cand_delta = delta * net->memory_gate_state[h] * 
                              tanh_derivative(net->candidate_memory[h]);
            for (int i = 0; i < net->input_size; i++) {
                int idx = h * net->input_size + i;
                net->grad_W_candidate[idx] += cand_delta * input[i];
            }
        }
    }
    
    free(output);
    free(output_error);
    free(hidden_error);
    
    net->grad_accumulation_count++;
    return loss;
}

// Apply accumulated gradients
EXPORT void apply_accumulated_gradients(Network *net) {
    if (!net || net->grad_accumulation_count == 0) return;
    
    float scale = 1.0f / net->grad_accumulation_count;
    int hidden_weights = net->input_size * net->hidden_size;
    int output_weights = net->hidden_size * net->output_size;
    int recurrent_weights = net->hidden_size * net->hidden_size;
    
    // Apply hidden layer gradients
    for (int i = 0; i < hidden_weights; i++) {
        if (net->freeze_mask_hidden[i]) {
            net->W_hidden[i] -= net->learning_rate * net->grad_W_hidden[i] * scale;
        }
    }
    
    // Apply output layer gradients
    for (int i = 0; i < output_weights; i++) {
        if (net->freeze_mask_output[i]) {
            net->W_output[i] -= net->learning_rate * net->grad_W_output[i] * scale;
        }
    }
    
    // Apply recurrent gradients
    if (net->use_recurrent) {
        for (int i = 0; i < recurrent_weights; i++) {
            net->U_recurrent[i] -= net->learning_rate * net->grad_U_recurrent[i] * scale;
        }
    }
    
    // Apply memory gate gradients
    if (net->use_learnable_gates) {
        for (int i = 0; i < hidden_weights; i++) {
            if (net->freeze_mask_memory[i]) {
                net->W_memory_gate[i] -= net->learning_rate * net->grad_W_memory_gate[i] * scale;
                net->W_candidate[i] -= net->learning_rate * net->grad_W_candidate[i] * scale;
            }
        }
    }
    
    // Reset gradient accumulators
    memset(net->grad_W_hidden, 0, hidden_weights * sizeof(float));
    memset(net->grad_W_output, 0, output_weights * sizeof(float));
    memset(net->grad_U_recurrent, 0, recurrent_weights * sizeof(float));
    memset(net->grad_W_memory_gate, 0, hidden_weights * sizeof(float));
    memset(net->grad_W_candidate, 0, hidden_weights * sizeof(float));
    
    net->grad_accumulation_count = 0;
    net->training_steps++;
}

// Reset accumulated gradients
EXPORT void reset_gradients(Network *net) {
    if (!net) return;
    
    int hidden_weights = net->input_size * net->hidden_size;
    int output_weights = net->hidden_size * net->output_size;
    int recurrent_weights = net->hidden_size * net->hidden_size;
    
    memset(net->grad_W_hidden, 0, hidden_weights * sizeof(float));
    memset(net->grad_W_output, 0, output_weights * sizeof(float));
    memset(net->grad_U_recurrent, 0, recurrent_weights * sizeof(float));
    memset(net->grad_W_memory_gate, 0, hidden_weights * sizeof(float));
    memset(net->grad_W_candidate, 0, hidden_weights * sizeof(float));
    
    net->grad_accumulation_count = 0;
}

EXPORT float train_batch(Network *net, const float *inputs, const float *targets, int batch_size) {
    if (!net || !inputs || !targets || batch_size <= 0) return -1.0f;
    
    float total_loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        const float *input = inputs + b * net->input_size;
        const float *target = targets + b * net->output_size;
        
        float loss = train(net, input, target);
        if (loss >= 0.0f) {
            total_loss += loss;
        }
    }
    
    return total_loss / batch_size;
}

// ============================================================================
// FEATURE CONTROL
// ============================================================================

EXPORT void set_use_recurrent(Network *net, bool enable) {
    if (net) net->use_recurrent = enable;
}

EXPORT void set_use_learnable_gates(Network *net, bool enable) {
    if (net) net->use_learnable_gates = enable;
}

EXPORT void set_use_output_memory(Network *net, bool enable) {
    if (net) net->use_output_memory = enable;
}

EXPORT bool get_use_recurrent(Network *net) {
    return net ? net->use_recurrent : false;
}

EXPORT bool get_use_learnable_gates(Network *net) {
    return net ? net->use_learnable_gates : false;
}

EXPORT bool get_use_output_memory(Network *net) {
    return net ? net->use_output_memory : false;
}

// ============================================================================
// PARTIAL TRAINING - ADVANCED FREEZE/UNFREEZE CONTROLS
// ============================================================================

EXPORT void freeze_hidden_layer(Network *net) {
    if (!net) return;
    int size = net->input_size * net->hidden_size;
    memset(net->freeze_mask_hidden, 0, size * sizeof(bool));
}

EXPORT void unfreeze_hidden_layer(Network *net) {
    if (!net) return;
    int size = net->input_size * net->hidden_size;
    memset(net->freeze_mask_hidden, 1, size * sizeof(bool));
}

EXPORT void freeze_output_layer(Network *net) {
    if (!net) return;
    int size = net->hidden_size * net->output_size;
    memset(net->freeze_mask_output, 0, size * sizeof(bool));
}

EXPORT void unfreeze_output_layer(Network *net) {
    if (!net) return;
    int size = net->hidden_size * net->output_size;
    memset(net->freeze_mask_output, 1, size * sizeof(bool));
}

EXPORT void freeze_memory_gates(Network *net) {
    if (!net) return;
    int size = net->input_size * net->hidden_size;
    memset(net->freeze_mask_memory, 0, size * sizeof(bool));
}

EXPORT void unfreeze_memory_gates(Network *net) {
    if (!net) return;
    int size = net->input_size * net->hidden_size;
    memset(net->freeze_mask_memory, 1, size * sizeof(bool));
}

// Freeze specific percentage of weights (randomly selected)
EXPORT void freeze_hidden_percentage(Network *net, float percentage) {
    if (!net || percentage < 0.0f || percentage > 1.0f) return;
    
    int size = net->input_size * net->hidden_size;
    int num_freeze = (int)(size * percentage);
    
    // Unfreeze all first
    memset(net->freeze_mask_hidden, 1, size * sizeof(bool));
    
    // Randomly freeze the specified percentage
    for (int i = 0; i < num_freeze; i++) {
        int idx = rand() % size;
        net->freeze_mask_hidden[idx] = false;
    }
}

EXPORT void freeze_output_percentage(Network *net, float percentage) {
    if (!net || percentage < 0.0f || percentage > 1.0f) return;
    
    int size = net->hidden_size * net->output_size;
    int num_freeze = (int)(size * percentage);
    
    memset(net->freeze_mask_output, 1, size * sizeof(bool));
    
    for (int i = 0; i < num_freeze; i++) {
        int idx = rand() % size;
        net->freeze_mask_output[idx] = false;
    }
}

// Freeze by weight magnitude (freeze smallest/largest weights)
EXPORT void freeze_by_magnitude(Network *net, float percentage, bool freeze_smallest) {
    if (!net || percentage < 0.0f || percentage > 1.0f) return;
    
    int hidden_size = net->input_size * net->hidden_size;
    int num_freeze = (int)(hidden_size * percentage);
    
    // Create array of (index, magnitude) pairs
    typedef struct {
        int idx;
        float mag;
    } WeightMag;
    
    WeightMag *weights = (WeightMag*)malloc(hidden_size * sizeof(WeightMag));
    for (int i = 0; i < hidden_size; i++) {
        weights[i].idx = i;
        weights[i].mag = fabsf(net->W_hidden[i]);
    }
    
    // Simple bubble sort (good enough for this use case)
    for (int i = 0; i < hidden_size - 1; i++) {
        for (int j = 0; j < hidden_size - i - 1; j++) {
            if (weights[j].mag > weights[j+1].mag) {
                WeightMag temp = weights[j];
                weights[j] = weights[j+1];
                weights[j+1] = temp;
            }
        }
    }
    
    // Unfreeze all
    memset(net->freeze_mask_hidden, 1, hidden_size * sizeof(bool));
    
    // Freeze based on magnitude
    if (freeze_smallest) {
        // Freeze smallest weights
        for (int i = 0; i < num_freeze; i++) {
            net->freeze_mask_hidden[weights[i].idx] = false;
        }
    } else {
        // Freeze largest weights
        for (int i = hidden_size - num_freeze; i < hidden_size; i++) {
            net->freeze_mask_hidden[weights[i].idx] = false;
        }
    }
    
    free(weights);
}

// Freeze specific neurons (all incoming/outgoing weights)
EXPORT void freeze_specific_neurons(Network *net, const int *neuron_indices, int num_neurons) {
    if (!net || !neuron_indices) return;
    
    for (int n = 0; n < num_neurons; n++) {
        int neuron_idx = neuron_indices[n];
        if (neuron_idx < 0 || neuron_idx >= net->hidden_size) continue;
        
        // Freeze all weights connected to this neuron
        for (int i = 0; i < net->input_size; i++) {
            int idx = neuron_idx * net->input_size + i;
            net->freeze_mask_hidden[idx] = false;
        }
        
        // Also freeze outgoing weights
        for (int o = 0; o < net->output_size; o++) {
            int idx = o * net->hidden_size + neuron_idx;
            net->freeze_mask_output[idx] = false;
        }
    }
}

// Set custom freeze masks
EXPORT void set_freeze_mask_hidden(Network *net, const bool *mask) {
    if (!net || !mask) return;
    int size = net->input_size * net->hidden_size;
    memcpy(net->freeze_mask_hidden, mask, size * sizeof(bool));
}

EXPORT void set_freeze_mask_output(Network *net, const bool *mask) {
    if (!net || !mask) return;
    int size = net->hidden_size * net->output_size;
    memcpy(net->freeze_mask_output, mask, size * sizeof(bool));
}

EXPORT void set_freeze_mask_memory(Network *net, const bool *mask) {
    if (!net || !mask) return;
    int size = net->input_size * net->hidden_size;
    memcpy(net->freeze_mask_memory, mask, size * sizeof(bool));
}

// Get freeze masks
EXPORT void get_freeze_mask_hidden(Network *net, bool *mask_out) {
    if (!net || !mask_out) return;
    int size = net->input_size * net->hidden_size;
    memcpy(mask_out, net->freeze_mask_hidden, size * sizeof(bool));
}

EXPORT void get_freeze_mask_output(Network *net, bool *mask_out) {
    if (!net || !mask_out) return;
    int size = net->hidden_size * net->output_size;
    memcpy(mask_out, net->freeze_mask_output, size * sizeof(bool));
}

// Get number of frozen weights
EXPORT int count_frozen_hidden(Network *net) {
    if (!net) return 0;
    int size = net->input_size * net->hidden_size;
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (!net->freeze_mask_hidden[i]) count++;
    }
    return count;
}

EXPORT int count_frozen_output(Network *net) {
    if (!net) return 0;
    int size = net->hidden_size * net->output_size;
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (!net->freeze_mask_output[i]) count++;
    }
    return count;
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

EXPORT void reset_memory(Network *net) {
    if (!net) return;
    memset(net->hidden_memory, 0, net->hidden_size * sizeof(float));
    memset(net->hidden_prev_output, 0, net->hidden_size * sizeof(float));
    memset(net->hidden_prev_state, 0, net->hidden_size * sizeof(float));
    memset(net->output_prev_output, 0, net->output_size * sizeof(float));
    memset(net->memory_gate_state, 0, net->hidden_size * sizeof(float));
}

EXPORT void get_memory_state(Network *net, float *memory_out) {
    if (!net || !memory_out) return;
    memcpy(memory_out, net->hidden_memory, net->hidden_size * sizeof(float));
}

EXPORT void set_memory_state(Network *net, const float *memory_in) {
    if (!net || !memory_in) return;
    memcpy(net->hidden_memory, memory_in, net->hidden_size * sizeof(float));
}

EXPORT void get_gate_state(Network *net, float *gates_out) {
    if (!net || !gates_out) return;
    memcpy(gates_out, net->memory_gate_state, net->hidden_size * sizeof(float));
}

// ============================================================================
// PARAMETER GETTERS/SETTERS
// ============================================================================

EXPORT float get_beta(Network *net) {
    return net ? net->beta : 0.0f;
}

EXPORT void set_beta(Network *net, float beta) {
    if (net) net->beta = beta;
}

EXPORT float get_alpha(Network *net) {
    return net ? net->alpha : 0.0f;
}

EXPORT void set_alpha(Network *net, float alpha) {
    if (net) net->alpha = alpha;
}

EXPORT float get_output_beta(Network *net) {
    return net ? net->output_beta : 0.0f;
}

EXPORT void set_output_beta(Network *net, float beta) {
    if (net) net->output_beta = beta;
}

EXPORT float get_learning_rate(Network *net) {
    return net ? net->learning_rate : 0.0f;
}

EXPORT void set_learning_rate(Network *net, float lr) {
    if (net) net->learning_rate = lr;
}

EXPORT int get_training_steps(Network *net) {
    return net ? net->training_steps : 0;
}

EXPORT float get_last_loss(Network *net) {
    return net ? net->last_loss : 0.0f;
}

EXPORT float get_avg_memory_magnitude(Network *net) {
    return net ? net->avg_memory_magnitude : 0.0f;
}

EXPORT float get_avg_gate_value(Network *net) {
    return net ? net->avg_gate_value : 0.0f;
}

// ============================================================================
// SAVE/LOAD - COMPLETE STATE PRESERVATION
// ============================================================================

EXPORT int save_network(Network *net, const char *filename) {
    if (!net || !filename) return -1;
    
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;
    
    // Write metadata
    fwrite(&net->input_size, sizeof(int), 1, f);
    fwrite(&net->hidden_size, sizeof(int), 1, f);
    fwrite(&net->output_size, sizeof(int), 1, f);
    fwrite(&net->beta, sizeof(float), 1, f);
    fwrite(&net->alpha, sizeof(float), 1, f);
    fwrite(&net->output_beta, sizeof(float), 1, f);
    fwrite(&net->learning_rate, sizeof(float), 1, f);
    fwrite(&net->training_steps, sizeof(int), 1, f);
    
    // Write feature flags
    fwrite(&net->use_recurrent, sizeof(bool), 1, f);
    fwrite(&net->use_learnable_gates, sizeof(bool), 1, f);
    fwrite(&net->use_output_memory, sizeof(bool), 1, f);
    
    // Write weights
    int hidden_weights = net->input_size * net->hidden_size;
    int output_weights = net->hidden_size * net->output_size;
    int recurrent_weights = net->hidden_size * net->hidden_size;
    
    fwrite(net->W_hidden, sizeof(float), hidden_weights, f);
    fwrite(net->W_output, sizeof(float), output_weights, f);
    fwrite(net->U_recurrent, sizeof(float), recurrent_weights, f);
    fwrite(net->bias_hidden, sizeof(float), net->hidden_size, f);
    fwrite(net->bias_output, sizeof(float), net->output_size, f);
    
    // Write learnable memory weights
    fwrite(net->W_memory_gate, sizeof(float), hidden_weights, f);
    fwrite(net->U_memory_gate, sizeof(float), recurrent_weights, f);
    fwrite(net->bias_memory_gate, sizeof(float), net->hidden_size, f);
    fwrite(net->W_candidate, sizeof(float), hidden_weights, f);
    fwrite(net->bias_candidate, sizeof(float), net->hidden_size, f);
    
    // Write ALL memory states (CRITICAL)
    fwrite(net->hidden_memory, sizeof(float), net->hidden_size, f);
    fwrite(net->hidden_prev_output, sizeof(float), net->hidden_size, f);
    fwrite(net->hidden_prev_state, sizeof(float), net->hidden_size, f);
    fwrite(net->output_prev_output, sizeof(float), net->output_size, f);
    fwrite(net->memory_gate_state, sizeof(float), net->hidden_size, f);
    
    fclose(f);
    return 0;
}

EXPORT int load_network(Network *net, const char *filename) {
    if (!net || !filename) return -1;
    
    FILE *f = fopen(filename, "rb");
    if (!f) return -1;
    
    // Read and verify metadata
    int input_size, hidden_size, output_size;
    fread(&input_size, sizeof(int), 1, f);
    fread(&hidden_size, sizeof(int), 1, f);
    fread(&output_size, sizeof(int), 1, f);
    
    if (input_size != net->input_size || 
        hidden_size != net->hidden_size || 
        output_size != net->output_size) {
        fclose(f);
        return -2; // Size mismatch
    }
    
    fread(&net->beta, sizeof(float), 1, f);
    fread(&net->alpha, sizeof(float), 1, f);
    fread(&net->output_beta, sizeof(float), 1, f);
    fread(&net->learning_rate, sizeof(float), 1, f);
    fread(&net->training_steps, sizeof(int), 1, f);
    
    // Read feature flags
    fread(&net->use_recurrent, sizeof(bool), 1, f);
    fread(&net->use_learnable_gates, sizeof(bool), 1, f);
    fread(&net->use_output_memory, sizeof(bool), 1, f);
    
    // Read weights
    int hidden_weights = net->input_size * net->hidden_size;
    int output_weights = net->hidden_size * net->output_size;
    int recurrent_weights = net->hidden_size * net->hidden_size;
    
    fread(net->W_hidden, sizeof(float), hidden_weights, f);
    fread(net->W_output, sizeof(float), output_weights, f);
    fread(net->U_recurrent, sizeof(float), recurrent_weights, f);
    fread(net->bias_hidden, sizeof(float), net->hidden_size, f);
    fread(net->bias_output, sizeof(float), net->output_size, f);
    
    // Read learnable memory weights
    fread(net->W_memory_gate, sizeof(float), hidden_weights, f);
    fread(net->U_memory_gate, sizeof(float), recurrent_weights, f);
    fread(net->bias_memory_gate, sizeof(float), net->hidden_size, f);
    fread(net->W_candidate, sizeof(float), hidden_weights, f);
    fread(net->bias_candidate, sizeof(float), net->hidden_size, f);
    
    // Read ALL memory states (restores complete network state!)
    fread(net->hidden_memory, sizeof(float), net->hidden_size, f);
    fread(net->hidden_prev_output, sizeof(float), net->hidden_size, f);
    fread(net->hidden_prev_state, sizeof(float), net->hidden_size, f);
    fread(net->output_prev_output, sizeof(float), net->output_size, f);
    fread(net->memory_gate_state, sizeof(float), net->hidden_size, f);
    
    fclose(f);
    return 0;
}

// ============================================================================
// INFO/DEBUG
// ============================================================================

EXPORT void print_network_info(Network *net) {
    if (!net) return;
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║              MEMORY-NATIVE NEURAL NETWORK INFO                ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Architecture: %d -> %d -> %d\n", 
           net->input_size, net->hidden_size, net->output_size);
    
    printf("\nMemory Parameters:\n");
    printf("  Beta (preservation):     %.3f\n", net->beta);
    printf("  Alpha (update rate):     %.3f\n", net->alpha);
    printf("  Output Beta:             %.3f\n", net->output_beta);
    printf("  Learning Rate:           %.4f\n", net->learning_rate);
    
    printf("\nFeatures:\n");
    printf("  Recurrent Connections:   %s\n", net->use_recurrent ? "✓" : "✗");
    printf("  Learnable Memory Gates:  %s\n", net->use_learnable_gates ? "✓" : "✗");
    printf("  Output Memory:           %s\n", net->use_output_memory ? "✓" : "✗");
    
    printf("\nTraining Stats:\n");
    printf("  Training steps:          %d\n", net->training_steps);
    printf("  Last loss:               %.6f\n", net->last_loss);
    
    printf("\nMemory State:\n");
    printf("  Avg memory magnitude:    %.6f\n", net->avg_memory_magnitude);
    printf("  Avg gate value:          %.6f\n", net->avg_gate_value);
    
    printf("\n");
}