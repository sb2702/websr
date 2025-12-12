import numpy as np
import tensorflow as tf
import json

def export_tensorflow_to_webgpu(model, output_file, network_name="custom/cnn-2x-l"):
    """
    Direct export from TensorFlow model to WebGPU JSON format
    Bypasses the GLSL intermediate step for cleaner, more reliable conversion
    """
    
    network_dict = {
        'name': network_name,
        'layers': {}
    }
    
    # Process each layer in the model
    for i, layer in enumerate(model.layers):
        layer_name = format_layer_name(layer.name)
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            export_conv2d_layer(layer, layer_name, network_dict)
        elif isinstance(layer, tf.keras.layers.Add):
            export_add_layer(layer, layer_name, network_dict)
        elif 'depth_to_space' in layer.name.lower():
            # Skip - handled by display layer in WebGPU
            continue
        elif isinstance(layer, tf.keras.layers.InputLayer):
            # Skip input layer
            continue
        elif 'upsampling' in layer.name.lower():
            # Skip upsampling - handled in pixel_shuffle
            continue
        elif 'concatenate' in layer.name.lower():
            # Skip concatenate - handled by the final conv layer
            continue
        elif 'crelu' in layer.name.lower():
            # Skip CReLU layers - they're just activations, handled in WebGPU shaders
            continue
        else:
            print(f"Skipping unsupported layer: {layer.name} ({type(layer)})")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(network_dict, f, indent=2)
    
    print(f"Exported {len(network_dict['layers'])} layers to {output_file}")
    return network_dict

def format_layer_name(name):
    """Format layer name for WebGPU compatibility"""
    if "." in name:
        name = name.split(".")[-1]
    
    # Handle special naming for the final concatenation layer
    if "conv2d_7" in name:
        name = name.replace("conv2d_7", "conv2d_last")
    
    return name + "_tf"

def export_conv2d_layer(layer, layer_name, network_dict):
    """Export Conv2D layer, splitting into 4-channel chunks"""
    weights = layer.get_weights()[0]  # [height, width, in_channels, out_channels]
    bias = layer.get_weights()[1] if len(layer.get_weights()) > 1 else np.zeros(weights.shape[-1])
    
    # Get dimensions
    kernel_h, kernel_w, in_channels, out_channels = weights.shape
    
    # Enforce 4-channel alignment
    if out_channels % 4 != 0:
        raise ValueError(f"Layer {layer_name} has {out_channels} output channels, which is not a multiple of 4. "
                        f"WebGPU implementation expects layers to have channels in multiples of 4.")
    
    num_chunks = out_channels // 4
    print(f"Exporting {layer_name}: {weights.shape} -> {num_chunks} 4-channel chunks")
    
    for chunk_idx in range(num_chunks):
        # Determine chunk name
        chunk_name = layer_name if chunk_idx == 0 else layer_name + str(chunk_idx)
        
        # Calculate channel range for this chunk (always exactly 4 channels)
        start_channel = chunk_idx * 4
        end_channel = start_channel + 4
        
        # Extract weights and bias for this chunk
        chunk_weights = weights[:, :, :, start_channel:end_channel]
        chunk_bias = bias[start_channel:end_channel]
        
        # Export this chunk
        export_conv_chunk(layer, chunk_name, chunk_weights, chunk_bias, network_dict)

def export_conv_chunk(layer, chunk_name, weights, bias, network_dict):
    """Export a single 4-channel chunk of a Conv2D layer"""
    kernel_h, kernel_w, in_channels, out_channels = weights.shape
    
    # Convert weights to WebGPU format
    flattened_weights = []
    
    # Process input channels first, then spatial positions (to match GLSL order)
    # GLSL does: all 9 positions × input1, then all 9 positions × input2, etc.
    for in_start in range(0, in_channels, 4):
        in_end = min(in_start + 4, in_channels)
        
        # For each spatial kernel position in column-major order
        for j in range(kernel_w):  # x direction first
            for i in range(kernel_h):  # then y direction
                # Create 4x4 matrix (WebGPU uses mat4x4)
                weight_matrix = np.zeros((4, 4))
                
                # Fill with actual weights (padded to 4x4)
                actual_in = in_end - in_start
                actual_out = min(4, out_channels)
                
                weight_matrix[:actual_in, :actual_out] = weights[i, j, in_start:in_end, :actual_out]
                
                flattened_weights.extend(weight_matrix.flatten())
    
    # Pad bias to 4 elements  
    padded_bias = np.zeros(4)
    padded_bias[:min(4, len(bias))] = bias[:min(4, len(bias))]
    
    network_dict['layers'][chunk_name] = {
        'name': chunk_name,
        'type': 'conv', 
        'inputs': get_layer_inputs(layer),
        'output': chunk_name,
        'weights': flattened_weights,
        'bias': padded_bias.tolist()
    }

def export_crelu_conv_layer(layer, base_name, weights, bias, network_dict):
    """Export CReLU Conv2D as dual branches"""
    kernel_h, kernel_w, in_channels, out_channels = weights.shape
    
    # For CReLU, we create two branches: base_name and base_name1
    # Both branches use the same weights since CReLU splits happen at activation, not weights
    for branch_idx in range(2):
        branch_name = base_name if branch_idx == 0 else base_name + "1"
        
        # Convert weights using the same logic as regular layers
        flattened_weights = convert_weights_to_webgpu_format(weights, kernel_h, kernel_w, in_channels, out_channels)
        
        # Pad bias to 4 elements
        padded_bias = np.zeros(4)
        padded_bias[:min(4, len(bias))] = bias[:min(4, len(bias))]
        
        # Add to network dict
        network_dict['layers'][branch_name] = {
            'name': branch_name,
            'type': 'conv',
            'inputs': get_layer_inputs(layer, branch_idx),
            'output': branch_name,
            'weights': flattened_weights,
            'bias': padded_bias.tolist()
        }

def export_regular_conv_layer(layer, layer_name, weights, bias, network_dict):
    """Export regular Conv2D layer"""
    kernel_h, kernel_w, in_channels, out_channels = weights.shape
    
    # Convert weights to WebGPU format
    flattened_weights = []
    
    # For 3x3 kernels, iterate through all 9 positions
    # GLSL processes in column-major order: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)
    for kx in range(kernel_w):  # x direction first
        for ky in range(kernel_h):  # then y direction
            # Create 4x4 matrix (WebGPU uses mat4x4)
            weight_matrix = np.zeros((4, 4))
            
            # Fill with actual weights (padded to 4x4)
            actual_in = min(4, in_channels) 
            actual_out = min(4, out_channels)
            weight_matrix[:actual_in, :actual_out] = weights[ky, kx, :actual_in, :actual_out]
            
            flattened_weights.extend(weight_matrix.flatten())
    
    # Pad bias to 4 elements  
    padded_bias = np.zeros(4)
    padded_bias[:min(4, len(bias))] = bias[:min(4, len(bias))]
    
    network_dict['layers'][layer_name] = {
        'name': layer_name,
        'type': 'conv', 
        'inputs': get_layer_inputs(layer),
        'output': layer_name,
        'weights': flattened_weights,
        'bias': padded_bias.tolist()
    }

def export_add_layer(layer, layer_name, network_dict):
    """Export Add layer (becomes pixel_shuffle in WebGPU)"""
    # The Add layer in the SR model becomes pixel_shuffle in WebGPU
    pixel_shuffle_name = "pixel_shuffle"
    
    network_dict['layers'][pixel_shuffle_name] = {
        'name': pixel_shuffle_name,
        'type': 'pixel_shuffle',
        'inputs': get_layer_inputs(layer),
        'output': 'canvas',  # Final output goes to canvas
        'bias': [0.0, 0.0, 0.0, 0.0]
    }

def get_layer_inputs(layer, branch_idx=None):
    """Get input layer names for a given layer"""
    if not hasattr(layer, '_inbound_nodes') or len(layer._inbound_nodes) == 0:
        return ["input_image"]
    
    inputs = []
    inbound_layers = layer._inbound_nodes[0].inbound_layers
    
    if not isinstance(inbound_layers, list):
        inbound_layers = [inbound_layers]
    
    for inp_layer in inbound_layers:
        # Handle Concatenate layers by expanding to all their inputs
        if 'concatenate' in inp_layer.name.lower():
            # Recursively get inputs from the concatenate layer
            concat_inputs = get_layer_inputs(inp_layer)
            inputs.extend(concat_inputs)
        else:
            # Skip CReLU layers and trace back to the actual Conv2D layer
            actual_layer = inp_layer
            while hasattr(actual_layer, '_inbound_nodes') and len(actual_layer._inbound_nodes) > 0:
                # If this is a CReLU layer, trace back to the Conv2D that feeds it
                if 'crelu' in actual_layer.name.lower():
                    prev_layers = actual_layer._inbound_nodes[0].inbound_layers
                    if not isinstance(prev_layers, list):
                        prev_layers = [prev_layers]
                    if len(prev_layers) > 0 and isinstance(prev_layers[0], tf.keras.layers.Conv2D):
                        actual_layer = prev_layers[0]
                        break
                else:
                    break
            
            # Get the WebGPU-formatted name of the actual Conv2D layer
            inp_name = format_layer_name(actual_layer.name)
            
            # If the input comes through CReLU, it means it was split into dual outputs
            if 'crelu' in inp_layer.name.lower():
                # Previous conv layer was split due to CReLU, so reference both parts
                inputs.append(inp_name)
                inputs.append(inp_name + "1")
            else:
                # Regular single input  
                inputs.append(inp_name)
    
    return inputs if inputs else ["input_image"]

def export_debug_activations(model, test_input, output_file):
    """
    Export intermediate activations for debugging
    Exactly matches the original notebook approach
    """
    # Get all Conv2D layers - exactly like the original
    activation_layers = []
    activation_names = []
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            activation_layers.append(layer.output)
            activation_names.append(layer.name + '_tf')
    
    # Create model to get intermediate outputs
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=activation_layers)
    model_activations = activation_model.predict(test_input)
    
    # Export exactly like the original notebook
    to_export = {}
    
    for i in range(len(activation_layers) - 1):
        layer_name = activation_names[i]
        layer_output = model_activations[i][0, :, :, :]  # Get all channels
        
        # Export in 4-channel chunks
        num_channels = layer_output.shape[-1]
        num_chunks = (num_channels + 3) // 4  # Round up to handle non-multiple of 4
        
        for chunk_idx in range(num_chunks):
            start_ch = chunk_idx * 4
            end_ch = min(start_ch + 4, num_channels)
            
            chunk_name = layer_name if chunk_idx == 0 else layer_name + str(chunk_idx)
            to_export[chunk_name] = layer_output[:, :, start_ch:end_ch].tolist()

    # Handle the last layer dynamically with proper naming
    last_layer_idx = len(activation_layers) - 1
    last_layer_output = model_activations[last_layer_idx][0, :, :, :]
    
    # Export last layer in 4-channel chunks with "conv2d_last" naming
    last_num_channels = last_layer_output.shape[-1]
    last_num_chunks = (last_num_channels + 3) // 4
    
    for chunk_idx in range(last_num_chunks):
        start_ch = chunk_idx * 4
        end_ch = min(start_ch + 4, last_num_channels)
        
        # Use conv2d_last naming convention
        if chunk_idx == 0:
            chunk_name = "conv2d_last_tf" 
        elif chunk_idx == 1:
            chunk_name = "conv2d_last_tf1"
        else:
            chunk_name = f"conv2d_last_tf{chunk_idx}"
            
        to_export[chunk_name] = last_layer_output[:, :, start_ch:end_ch].tolist()
    
    # Save debug data
    with open(output_file, 'w') as f:
        json.dump(to_export, f)
    
    print(f"Exported debug activations to {output_file}")
    return to_export

# Example usage function
def export_model_example():
    """
    Example of how to use the export functions
    """
    # Assume you have a trained model
    # model = your_trained_model
    # test_input = np.expand_dims(test_image, axis=0)
    
    # Export weights
    # network_dict = export_tensorflow_to_webgpu(
    #     model, 
    #     'custom-network-weights.json',
    #     'custom/cnn-2x-l'
    # )
    
    # Export debug activations
    # debug_data = export_debug_activations(
    #     model,
    #     test_input, 
    #     'custom-debug-weights.json'
    # )
    
    print("Use the functions above with your trained model")

if __name__ == "__main__":
    export_model_example()