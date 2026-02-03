/**
 * MEMORY-NATIVE NEURAL NETWORK - C LIBRARY FOR DLL
 * 
 * Compile to DLL/SO:
 * Windows: gcc -shared -o memory_net.dll memory_net_dll.c -lm -O3
 * Linux:   gcc -shared -fPIC -o memory_net.so memory_net_dll.c -lm -O3
 * Mac:     gcc -shared -fPIC -o memory_net.dylib memory_net_dll.c -lm -O3
 * Licensed under GPL V3.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

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
    
    // Weights
    float *W_hidden;      // input -> hidden weights
    float *W_output;      // hidden -> output weights
    float *bias_hidden;
    float *bias_output;
    
    // Persistent memory states
    float *hidden_memory;      // Internal memory (Concept 2)
    float *hidden_prev_output; // Previous output (Concept 1)
    float *hidden_state;       // Current hidden state
    float *output_state;       // Current output state
    
    // Memory parameters
    float beta;   // Memory preservation factor (0-1)
    float alpha;  // Memory update rate (0-1)
    float learning_rate;
    
    // Partial training masks
    bool *freeze_mask_hidden;
    bool *freeze_mask_output;
    
    // Statistics
    int training_steps;
    float last_loss;
    
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

// ============================================================================
// NETWORK CREATION AND DESTRUCTION
// ============================================================================

EXPORT Network* create_network(int input_size, int hidden_size, int output_size,
                               float beta, float alpha, float learning_rate) {
    Network *net = (Network*)malloc(sizeof(Network));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    net->beta = beta;
    net->alpha = alpha;
    net->learning_rate = learning_rate;
    net->training_steps = 0;
    net->last_loss = 0.0f;
    
    // Allocate weights
    int hidden_weights = input_size * hidden_size;
    int output_weights = hidden_size * output_size;
    
    net->W_hidden = (float*)malloc(hidden_weights * sizeof(float));
    net->W_output = (float*)malloc(output_weights * sizeof(float));
    net->bias_hidden = (float*)calloc(hidden_size, sizeof(float));
    net->bias_output = (float*)calloc(output_size, sizeof(float));
    
    // Allocate memory states
    net->hidden_memory = (float*)calloc(hidden_size, sizeof(float));
    net->hidden_prev_output = (float*)calloc(hidden_size, sizeof(float));
    net->hidden_state = (float*)calloc(hidden_size, sizeof(float));
    net->output_state = (float*)calloc(output_size, sizeof(float));
    
    // Allocate freeze masks (all trainable initially)
    net->freeze_mask_hidden = (bool*)malloc(hidden_weights * sizeof(bool));
    net->freeze_mask_output = (bool*)malloc(output_weights * sizeof(bool));
    memset(net->freeze_mask_hidden, 1, hidden_weights * sizeof(bool));
    memset(net->freeze_mask_output, 1, output_weights * sizeof(bool));
    
    // Xavier initialization
    float scale_hidden = sqrtf(2.0f / (input_size + hidden_size));
    float scale_output = sqrtf(2.0f / (hidden_size + output_size));
    
    for (int i = 0; i < hidden_weights; i++) {
        net->W_hidden[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
    }
    
    for (int i = 0; i < output_weights; i++) {
        net->W_output[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_output;
    }
    
    return net;
}

EXPORT void destroy_network(Network *net) {
    if (!net) return;
    
    free(net->W_hidden);
    free(net->W_output);
    free(net->bias_hidden);
    free(net->bias_output);
    free(net->hidden_memory);
    free(net->hidden_prev_output);
    free(net->hidden_state);
    free(net->output_state);
    free(net->freeze_mask_hidden);
    free(net->freeze_mask_output);
    free(net);
}

// ============================================================================
// FORWARD PASS WITH ALL THREE CONCEPTS
// ============================================================================

EXPORT void forward(Network *net, const float *input, float *output) {
    if (!net || !input || !output) return;
    
    // HIDDEN LAYER with memory concepts
    for (int h = 0; h < net->hidden_size; h++) {
        // Compute weighted sum
        float activation = net->bias_hidden[h];
        for (int i = 0; i < net->input_size; i++) {
            activation += net->W_hidden[h * net->input_size + i] * input[i];
        }
        
        // CONCEPT 2: Add internal memory contribution
        activation += net->hidden_memory[h];
        
        // Apply activation function
        float new_value = tanh_act(activation);
        
        // CONCEPT 1: Memory preservation - add β × y(t-1)
        float memory_echo = net->beta * net->hidden_prev_output[h];
        float hidden_output = new_value + memory_echo;
        
        // Store current output as previous for next iteration
        net->hidden_prev_output[h] = net->hidden_state[h];
        net->hidden_state[h] = hidden_output;
        
        // CONCEPT 2: Update internal memory
        // memory(t) = (1-α) × memory(t-1) + α × new_info
        net->hidden_memory[h] = (1.0f - net->alpha) * net->hidden_memory[h] + 
                                 net->alpha * new_value;
    }
    
    // OUTPUT LAYER (standard)
    for (int o = 0; o < net->output_size; o++) {
        float activation = net->bias_output[o];
        for (int h = 0; h < net->hidden_size; h++) {
            activation += net->W_output[o * net->hidden_size + h] * net->hidden_state[h];
        }
        net->output_state[o] = tanh_act(activation);
        output[o] = net->output_state[o];
    }
}

// ============================================================================
// PREDICTION (wrapper for forward)
// ============================================================================

EXPORT void predict(Network *net, const float *input, float *output) {
    forward(net, input, output);
}

// ============================================================================
// TRAINING
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
                
                // Clip for stability
                if (net->W_output[idx] > 5.0f) net->W_output[idx] = 5.0f;
                if (net->W_output[idx] < -5.0f) net->W_output[idx] = -5.0f;
            }
            
            // Accumulate error for hidden layer
            hidden_error[h] += delta * net->W_output[idx];
        }
        
        // Update output bias
        net->bias_output[o] -= net->learning_rate * delta;
    }
    
    // Backward pass - hidden layer
    for (int h = 0; h < net->hidden_size; h++) {
        float delta = hidden_error[h] * tanh_derivative(net->hidden_state[h]);
        
        // Update hidden weights (respecting freeze mask)
        for (int i = 0; i < net->input_size; i++) {
            int idx = h * net->input_size + i;
            if (net->freeze_mask_hidden[idx]) {
                float grad = delta * input[i];
                net->W_hidden[idx] -= net->learning_rate * grad;
                
                // Clip for stability
                if (net->W_hidden[idx] > 5.0f) net->W_hidden[idx] = 5.0f;
                if (net->W_hidden[idx] < -5.0f) net->W_hidden[idx] = -5.0f;
            }
        }
        
        // Update hidden bias
        net->bias_hidden[h] -= net->learning_rate * delta;
    }
    
    // Cleanup
    free(output);
    free(output_error);
    free(hidden_error);
    
    net->training_steps++;
    net->last_loss = loss;
    
    return loss;
}

// ============================================================================
// BATCH TRAINING
// ============================================================================

EXPORT float train_batch(Network *net, const float *inputs, const float *targets, 
                         int batch_size) {
    if (!net || !inputs || !targets || batch_size <= 0) return -1.0f;
    
    float total_loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        const float *input = inputs + (b * net->input_size);
        const float *target = targets + (b * net->output_size);
        
        float loss = train(net, input, target);
        total_loss += loss;
    }
    
    return total_loss / batch_size;
}

// ============================================================================
// PARTIAL TRAINING - FREEZE/UNFREEZE
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

EXPORT void freeze_percentage(Network *net, float percentage) {
    if (!net || percentage < 0.0f || percentage > 1.0f) return;
    
    // Freeze percentage of hidden layer weights
    int hidden_size = net->input_size * net->hidden_size;
    int num_freeze = (int)(hidden_size * percentage);
    
    // Unfreeze all first
    memset(net->freeze_mask_hidden, 1, hidden_size * sizeof(bool));
    
    // Randomly freeze the specified percentage
    for (int i = 0; i < num_freeze; i++) {
        int idx = rand() % hidden_size;
        net->freeze_mask_hidden[idx] = false;
    }
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

EXPORT void reset_memory(Network *net) {
    if (!net) return;
    memset(net->hidden_memory, 0, net->hidden_size * sizeof(float));
    memset(net->hidden_prev_output, 0, net->hidden_size * sizeof(float));
}

EXPORT void get_memory_state(Network *net, float *memory_out) {
    if (!net || !memory_out) return;
    memcpy(memory_out, net->hidden_memory, net->hidden_size * sizeof(float));
}

EXPORT void set_memory_state(Network *net, const float *memory_in) {
    if (!net || !memory_in) return;
    memcpy(net->hidden_memory, memory_in, net->hidden_size * sizeof(float));
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

// ============================================================================
// SAVE/LOAD
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
    fwrite(&net->learning_rate, sizeof(float), 1, f);
    fwrite(&net->training_steps, sizeof(int), 1, f);
    
    // Write weights
    int hidden_weights = net->input_size * net->hidden_size;
    int output_weights = net->hidden_size * net->output_size;
    
    fwrite(net->W_hidden, sizeof(float), hidden_weights, f);
    fwrite(net->W_output, sizeof(float), output_weights, f);
    fwrite(net->bias_hidden, sizeof(float), net->hidden_size, f);
    fwrite(net->bias_output, sizeof(float), net->output_size, f);
    
    // Write memory states (CRITICAL - this preserves learning!)
    fwrite(net->hidden_memory, sizeof(float), net->hidden_size, f);
    fwrite(net->hidden_prev_output, sizeof(float), net->hidden_size, f);
    
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
    fread(&net->learning_rate, sizeof(float), 1, f);
    fread(&net->training_steps, sizeof(int), 1, f);
    
    // Read weights
    int hidden_weights = net->input_size * net->hidden_size;
    int output_weights = net->hidden_size * net->output_size;
    
    fread(net->W_hidden, sizeof(float), hidden_weights, f);
    fread(net->W_output, sizeof(float), output_weights, f);
    fread(net->bias_hidden, sizeof(float), net->hidden_size, f);
    fread(net->bias_output, sizeof(float), net->output_size, f);
    
    // Read memory states (restores full network state!)
    fread(net->hidden_memory, sizeof(float), net->hidden_size, f);
    fread(net->hidden_prev_output, sizeof(float), net->hidden_size, f);
    
    fclose(f);
    return 0;
}

// ============================================================================
// INFO/DEBUG
// ============================================================================

EXPORT void print_network_info(Network *net) {
    if (!net) return;
    
    printf("Network: %d -> %d -> %d\n", 
           net->input_size, net->hidden_size, net->output_size);
    printf("Beta: %.3f, Alpha: %.3f, LR: %.4f\n", 
           net->beta, net->alpha, net->learning_rate);
    printf("Training steps: %d, Last loss: %.6f\n", 
           net->training_steps, net->last_loss);
    
    // Memory statistics
    float avg_memory = 0.0f;
    for (int i = 0; i < net->hidden_size; i++) {
        avg_memory += fabsf(net->hidden_memory[i]);
    }
    avg_memory /= net->hidden_size;
    printf("Average internal memory magnitude: %.6f\n", avg_memory);
}

EXPORT void get_weights_hidden(Network *net, float *weights_out) {
    if (!net || !weights_out) return;
    int size = net->input_size * net->hidden_size;
    memcpy(weights_out, net->W_hidden, size * sizeof(float));
}

EXPORT void get_weights_output(Network *net, float *weights_out) {
    if (!net || !weights_out) return;
    int size = net->hidden_size * net->output_size;
    memcpy(weights_out, net->W_output, size * sizeof(float));
}