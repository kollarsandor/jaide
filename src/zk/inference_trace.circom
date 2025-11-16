pragma circom 2.0.0;

template IsZero() {
    signal input in;
    signal output out;
    
    signal inv;
    
    inv <-- in != 0 ? 1 / in : 0;
    
    out <== 1 - in * inv;
    
    in * out === 0;
    
    (1 - out) * (in - 0) === (in - 0);
}

template RankerStep(dim, learning_factor) {
    signal input segment_hash;
    signal input base_score;
    signal input position;
    signal output final_score;
    
    signal position_safe;
    position_safe <== position + 1;
    
    signal bias;
    signal inv_denominator;
    signal denominator;
    
    signal pos_squared;
    pos_squared <== position_safe * position_safe;
    denominator <== pos_squared + 2;
    
    inv_denominator <-- 1 / denominator;
    
    inv_denominator * denominator === 1;
    
    signal numerator;
    numerator <== learning_factor * position_safe;
    bias <== numerator * inv_denominator;
    
    final_score <== base_score + bias;
}

template RSFLayerForward(dim) {
    signal input x[dim];
    signal input weights_s[dim][dim];
    signal input weights_t[dim][dim];
    signal output y[dim];
    
    var actual_dim = dim;
    var half = (dim + 1) >> 1;
    var padded_dim = half * 2;
    
    signal x1[half];
    signal x2[half];
    
    for (var i = 0; i < half; i++) {
        if (i < dim) {
            x1[i] <== i < actual_dim ? x[i] : 0;
        } else {
            x1[i] <== 0;
        }
        
        if (half + i < dim) {
            x2[i] <== x[half + i];
        } else {
            x2[i] <== 0;
        }
    }
    
    signal s_x2[half];
    for (var i = 0; i < half; i++) {
        signal partial_sums[half + 1];
        partial_sums[0] <== 0;
        for (var j = 0; j < half; j++) {
            signal product;
            product <== weights_s[i][j] * x2[j];
            partial_sums[j + 1] <== partial_sums[j] + product;
        }
        s_x2[i] <== partial_sums[half];
    }
    
    signal y1[half];
    for (var i = 0; i < half; i++) {
        signal s_squared;
        signal s_cubed;
        signal s_fourth;
        signal exp_approx;
        
        s_squared <== s_x2[i] * s_x2[i];
        s_cubed <== s_squared * s_x2[i];
        s_fourth <== s_squared * s_squared;
        
        signal term1;
        signal term2;
        signal term3;
        signal term4;
        
        term1 <== s_x2[i];
        
        signal s_squared_scaled;
        signal s_cubed_scaled;
        signal s_fourth_scaled;
        
        s_squared_scaled <== s_squared * 500;
        term2 <-- s_squared_scaled \ 1000;
        term2 * 1000 === s_squared_scaled - (s_squared_scaled % 1000);
        
        s_cubed_scaled <== s_cubed * 167;
        term3 <-- s_cubed_scaled \ 1000;
        term3 * 1000 === s_cubed_scaled - (s_cubed_scaled % 1000);
        
        s_fourth_scaled <== s_fourth * 42;
        term4 <-- s_fourth_scaled \ 1000;
        term4 * 1000 === s_fourth_scaled - (s_fourth_scaled % 1000);
        
        signal exp_partial1;
        signal exp_partial2;
        signal exp_partial3;
        
        exp_partial1 <== 1000 + term1;
        exp_partial2 <== exp_partial1 + term2;
        exp_partial3 <== exp_partial2 + term3;
        exp_approx <== exp_partial3 + term4;
        
        signal y1_numerator;
        y1_numerator <== x1[i] * exp_approx;
        y1[i] <-- y1_numerator \ 1000;
        y1[i] * 1000 === y1_numerator - (y1_numerator % 1000);
    }
    
    signal t_y1[half];
    for (var i = 0; i < half; i++) {
        signal partial_sums[half + 1];
        partial_sums[0] <== 0;
        for (var j = 0; j < half; j++) {
            signal product;
            product <== weights_t[i][j] * y1[j];
            partial_sums[j + 1] <== partial_sums[j] + product;
        }
        t_y1[i] <== partial_sums[half];
    }
    
    signal y2[half];
    for (var i = 0; i < half; i++) {
        y2[i] <== x2[i] + t_y1[i];
    }
    
    for (var i = 0; i < half && i < dim; i++) {
        y[i] <== y1[i];
    }
    for (var i = 0; i < half && (half + i) < dim; i++) {
        y[half + i] <== y2[i];
    }
}

template Num2Bits(n) {
    signal input in;
    signal output out[n];
    
    assert(n <= 252);
    
    var lc = 0;
    var e = 1;
    
    for (var i = 0; i < n; i++) {
        out[i] <-- (in >> i) & 1;
        
        out[i] * (out[i] - 1) === 0;
        
        lc = lc + out[i] * e;
        
        e = e * 2;
    }
    
    lc === in;
}

template LessThan(n) {
    assert(n <= 252);
    assert(n <= 64);
    signal input in[2];
    signal output out;
    
    component num2bits = Num2Bits(n + 1);
    
    signal offset;
    offset <== (1 << n);
    
    signal adjusted;
    adjusted <== in[0] - in[1] + offset;
    
    num2bits.in <== adjusted;
    
    out <== 1 - num2bits.out[n];
}

template InferenceTrace(num_layers, dim, error_scale) {
    signal input tokens[dim];
    signal input layer_weights_s[num_layers][dim][dim];
    signal input layer_weights_t[num_layers][dim][dim];
    signal input expected_output[dim];
    signal output is_valid;
    
    signal layer_outputs[num_layers + 1][dim];
    
    for (var i = 0; i < dim; i++) {
        layer_outputs[0][i] <== tokens[i];
    }
    
    component rsf_layers[num_layers];
    for (var layer = 0; layer < num_layers; layer++) {
        rsf_layers[layer] = RSFLayerForward(dim);
        
        for (var i = 0; i < dim; i++) {
            rsf_layers[layer].x[i] <== layer_outputs[layer][i];
            for (var j = 0; j < dim; j++) {
                rsf_layers[layer].weights_s[i][j] <== layer_weights_s[layer][i][j];
                rsf_layers[layer].weights_t[i][j] <== layer_weights_t[layer][i][j];
            }
        }
        
        for (var i = 0; i < dim; i++) {
            layer_outputs[layer + 1][i] <== rsf_layers[layer].y[i];
        }
    }
    
    signal differences[dim];
    signal squared_diff[dim];
    
    for (var i = 0; i < dim; i++) {
        differences[i] <== layer_outputs[num_layers][i] - expected_output[i];
        squared_diff[i] <== differences[i] * differences[i];
    }
    
    signal partial_sums[dim + 1];
    partial_sums[0] <== 0;
    
    for (var i = 0; i < dim; i++) {
        signal scaled_error;
        signal scaled_numerator;
        scaled_numerator <== squared_diff[i];
        scaled_error <-- scaled_numerator \ error_scale;
        scaled_error * error_scale === scaled_numerator - (scaled_numerator % error_scale);
        partial_sums[i + 1] <== partial_sums[i] + scaled_error;
    }
    
    signal total_error;
    total_error <== partial_sums[dim];
    
    signal error_threshold;
    error_threshold <== error_scale;
    
    component threshold_check = LessThan(64);
    threshold_check.in[0] <== total_error;
    threshold_check.in[1] <== error_threshold;
    
    is_valid <== threshold_check.out;
    
    is_valid * (1 - is_valid) === 0;
}

component main {public [tokens, expected_output]} = InferenceTrace(4, 16, 1000);
