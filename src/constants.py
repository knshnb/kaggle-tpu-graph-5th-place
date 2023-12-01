OPCODES = {
    "abs": 1,
    "add": 2,
    "add-dependency": 3,
    "after-all": 4,
    "all-reduce": 5,
    "all-to-all": 6,
    "atan2": 7,
    "batch-norm-grad": 8,
    "batch-norm-inference": 9,
    "batch-norm-training": 10,
    "bitcast": 11,
    "bitcast-convert": 12,
    "broadcast": 13,
    "call": 14,
    "ceil": 15,
    "cholesky": 16,
    "clamp": 17,
    "collective-permute": 18,
    "count-leading-zeros": 19,
    "compare": 20,
    "complex": 21,
    "concatenate": 22,
    "conditional": 23,
    "constant": 24,
    "convert": 25,
    "convolution": 26,
    "copy": 27,
    "copy-done": 28,
    "copy-start": 29,
    "cosine": 30,
    "custom-call": 31,
    "divide": 32,
    "domain": 33,
    "dot": 34,
    "dynamic-slice": 35,
    "dynamic-update-slice": 36,
    "exponential": 37,
    "exponential-minus-one": 38,
    "fft": 39,
    "floor": 40,
    "fusion": 41,
    "gather": 42,
    "get-dimension-size": 43,
    "set-dimension-size": 44,
    "get-tuple-element": 45,
    "imag": 46,
    "infeed": 47,
    "iota": 48,
    "is-finite": 49,
    "log": 50,
    "log-plus-one": 51,
    "and": 52,
    "not": 53,
    "or": 54,
    "xor": 55,
    "map": 56,
    "maximum": 57,
    "minimum": 58,
    "multiply": 59,
    "negate": 60,
    "outfeed": 61,
    "pad": 62,
    "parameter": 63,
    "partition-id": 64,
    "popcnt": 65,
    "power": 66,
    "real": 67,
    "recv": 68,
    "recv-done": 69,
    "reduce": 70,
    "reduce-precision": 71,
    "reduce-window": 72,
    "remainder": 73,
    "replica-id": 74,
    "reshape": 75,
    "reverse": 76,
    "rng": 77,
    "rng-get-and-update-state": 78,
    "rng-bit-generator": 79,
    "round-nearest-afz": 80,
    "rsqrt": 81,
    "scatter": 82,
    "select": 83,
    "select-and-scatter": 84,
    "send": 85,
    "send-done": 86,
    "shift-left": 87,
    "shift-right-arithmetic": 88,
    "shift-right-logical": 89,
    "sign": 90,
    "sine": 91,
    "slice": 92,
    "sort": 93,
    "sqrt": 94,
    "subtract": 95,
    "tanh": 96,
    "transpose": 98,
    "triangular-solve": 99,
    "tuple": 100,
    "while": 102,
    "cbrt": 103,
    "all-gather": 104,
    "collective-permute-start": 105,
    "collective-permute-done": 106,
    "logistic": 107,
    "dynamic-reshape": 108,
    "all-reduce-start": 109,
    "all-reduce-done": 110,
    "reduce-scatter": 111,
    "all-gather-start": 112,
    "all-gather-done": 113,
    "opt-barrier": 114,
    "async-start": 115,
    "async-update": 116,
    "async-done": 117,
    "round-nearest-even": 118,
    "stochastic-convert": 119,
    "tan": 120,
}
_opcode_name_groups = {
    "element-wise-unary": [
        "abs",
        "ceil",
        "cosine",
        "sine",
        "exponential",
        "exponential-minus-one",
        "floor",
        "is-finite",
        "log",
        "log-plus-one",
        "logistic",
        "not",
        "negate",
        "round-nearest-even",
        "round-nearest-afz",
        "sqrt",
        "rsqrt",
        "cbrt",
        "sign",
        "tan",
        "tanh",
    ],
    "element-wise-binary": [
        "add",
        "subtract",
        "multiply",
        "divide",
        "remainder",
        "maximum",
        "minimum",
        "atan2",
        "and",
        "or",
        "xor",
    ],
}
opcode_groups = {key: [OPCODES[name] for name in names] for key, names in _opcode_name_groups.items()}

# (min, max, is_log)
NORMAL_FEATS = {
    # is_root - whether this node is the output
    0: (0.0, 1.0, False),
    # shape_element_type_is_pred
    3: (0.0, 1.0, False),
    # shape_element_type_is_s32
    6: (0.0, 1.0, False),
    # shape_element_type_is_s64
    7: (0.0, 1.0, False),
    # shape_element_type_is_u32
    10: (0.0, 1.0, False),
    # shape_element_type_is_f32
    13: (0.0, 1.0, False),
    # shape_element_type_is_bf16
    15: (0.0, 1.0, False),
    # shape_element_type_is_tuple
    18: (0.0, 1.0, False),
    # shape_element_type_is_token
    20: (0.0, 1.0, False),
    # shape_dimensions_sum
    27: (0.0, 184320000.0, True),
    # shape_dimensions_product
    28: (1.0, 2949120000.0, True),
    # shape_tuple_shapes_size - for tuples only, the shapes of constituent shapes in the tuple sequence
    29: (0.0, 3503.0, True),
    # parameter_number = K - indicating that is is the Kth parameter to the computation, only for Parameter operation
    30: (0.0, 35.0, False),
    # window_size_sum
    43: (0.0, 1024.0, True),
    # window_size_product
    44: (1.0, 2097152.0, True),
    # window_stride_sum
    51: (0.0, 8.0, True),
    # window_stride_product
    52: (1.0, 16.0, True),
    # window_padding_low_sum
    59: (0.0, 127.0, True),
    # window_padding_low_product
    60: (0.0, 16.0, True),
    # window_padding_high_sum
    67: (-2.0, 8.0, True),
    # window_padding_high_product
    68: (0.0, 18.0, True),
    # window_window_dilation_sum
    75: (0.0, 8.0, True),
    # window_window_dilation_product
    76: (1.0, 16.0, True),
    # window_base_dilation_sum
    83: (0.0, 8.0, True),
    # window_base_dilation_product
    84: (1.0, 16.0, True),
    # window_window_reversal_true_count
    91: (0.0, 3.0, False),
    # window_window_reversal_false_count
    92: (0.0, 5.0, False),
    # feature_group_count - the number of feature groups, used for a convolution. Must be a divisor of the input feature dimension and output feature dimension. If not specified, it will use a default value of 1.
    107: (0.0, 3840.0, True),
    # batch_group_count - the number of batch groups, used for a convolution.
    108: (0.0, 3840.0, True),
    # slice_dims_start_0
    109: (0.0, 1024.0, True),
    # slice_dims_start_1
    110: (0.0, 119646.0, True),
    # slice_dims_start_sum
    111: (0.0, 119646.0, True),
    # slice_dims_start_product
    112: (0.0, 1024.0, True),
    # slice_dims_stride_0
    113: (0.0, 1.0, False),
    # slice_dims_stride_1
    114: (0.0, 1.0, False),
    # slice_dims_stride_sum
    115: (0.0, 5.0, False),
    # slice_dims_limit_0
    117: (0.0, 12375.0, True),
    # slice_dims_limit_1
    118: (0.0, 196608.0, True),
    # slice_dims_limit_sum
    119: (0.0, 196614.0, True),
    # slice_dims_limit_product
    120: (1.0, 805306400.0, True),
    # dynamic_slice_sizes_0
    121: (0.0, 12.0, True),
    # dynamic_slice_sizes_1
    122: (0.0, 2048.0, True),
    # dynamic_slice_sizes_sum
    123: (0.0, 33784.0, True),
    # dynamic_slice_sizes_product
    124: (1.0, 25887744.0, True),
    # padding_config_edge_padding_low_0
    125: (0.0, 1.0, False),
    # padding_config_edge_padding_low_1
    126: (0.0, 64.0, True),
    # padding_config_edge_padding_low_sum
    127: (-39.0, 448.0, True),
    # padding_config_edge_padding_low_product
    128: (0.0, 1.0, False),
    # padding_config_edge_padding_high_0
    129: (0.0, 3072.0, True),
    # padding_config_edge_padding_high_1
    130: (0.0, 1023.0, True),
    # padding_config_edge_padding_high_sum
    131: (-24.0, 33784.0, True),
    # padding_config_edge_padding_high_product
    132: (0.0, 25887744.0, True),
    # is_stable - whether this Sort operation should be stable
    133: (0.0, 1.0, False),
}
DIM_FEATS = {
    # shape_dimensions_0
    21: (0.0, 184320000.0, True),
    # shape_dimensions_1
    22: (0.0, 184320000.0, True),
    # shape_dimensions_2
    23: (0.0, 184320000.0, True),
    # shape_dimensions_3
    24: (0.0, 184320000.0, True),
    # shape_dimensions_4
    25: (0.0, 184320000.0, True),
    # shape_dimensions_5
    26: (0.0, 184320000.0, True),
    # dimensions_0
    31: (0.0, 5.0, False),
    # dimensions_1
    32: (0.0, 5.0, False),
    # dimensions_2
    33: (0.0, 5.0, False),
    # dimensions_3
    34: (0.0, 5.0, False),
    # dimensions_4
    35: (0.0, 5.0, False),
    # dimensions_5
    36: (0.0, 5.0, False),
    # window_size_0
    37: (0.0, 512.0, True),
    # window_size_1
    38: (0.0, 512.0, True),
    # window_size_2
    39: (0.0, 512.0, True),
    # window_size_3
    40: (0.0, 512.0, True),
    # window_size_4
    41: (0.0, 512.0, True),
    # window_size_5
    42: (0.0, 512.0, True),
    # window_stride_0
    45: (0.0, 4.0, False),
    # window_stride_1
    46: (0.0, 4.0, False),
    # window_stride_2
    47: (0.0, 4.0, False),
    # window_stride_3
    48: (0.0, 4.0, False),
    # window_stride_4
    49: (0.0, 4.0, False),
    # window_stride_5
    50: (0.0, 4.0, False),
    # window_padding_low_0
    53: (0.0, 127.0, True),
    # window_padding_low_1
    54: (0.0, 127.0, True),
    # window_padding_low_2
    55: (0.0, 127.0, True),
    # window_padding_low_3
    56: (0.0, 127.0, True),
    # window_padding_low_4
    57: (0.0, 127.0, True),
    # window_padding_low_5
    58: (0.0, 127.0, True),
    # window_padding_high_0
    61: (-1.0, 4.0, False),
    # window_padding_high_1
    62: (-1.0, 4.0, False),
    # window_padding_high_2
    63: (-1.0, 4.0, False),
    # window_padding_high_3
    64: (-1.0, 4.0, False),
    # window_padding_high_4
    65: (-1.0, 4.0, False),
    # window_padding_high_5
    66: (-1.0, 4.0, False),
    # window_window_dilation_0
    69: (0.0, 4.0, False),
    # window_window_dilation_1
    70: (0.0, 4.0, False),
    # window_window_dilation_2
    71: (0.0, 4.0, False),
    # window_window_dilation_3
    72: (0.0, 4.0, False),
    # window_window_dilation_4
    73: (0.0, 4.0, False),
    # window_window_dilation_5
    74: (0.0, 4.0, False),
    # window_base_dilation_0
    77: (0.0, 4.0, False),
    # window_base_dilation_1
    78: (0.0, 4.0, False),
    # window_base_dilation_2
    79: (0.0, 4.0, False),
    # window_base_dilation_3
    80: (0.0, 4.0, False),
    # window_base_dilation_4
    81: (0.0, 4.0, False),
    # window_base_dilation_5
    82: (0.0, 4.0, False),
    # window_window_reversal_0
    85: (0.0, 1.0, False),
    # window_window_reversal_1
    86: (0.0, 1.0, False),
    # window_window_reversal_2
    87: (0.0, 1.0, False),
    # window_window_reversal_3
    88: (0.0, 1.0, False),
    # window_window_reversal_4
    89: (0.0, 1.0, False),
    # window_window_reversal_5
    90: (0.0, 1.0, False),
}
DIM_NUMBER_FEATS = {
    # convolution_dim_numbers_input_batch_dim - the dimension number that represents batch in the input
    93: (0.0, 4.0, False),
    # convolution_dim_numbers_input_feature_dim - the dimension number that represents features in the input
    94: (0.0, 4.0, False),
    # convolution_dim_numbers_input_spatial_dims_0
    95: (0.0, 3.0, False),
    # convolution_dim_numbers_input_spatial_dims_1
    96: (0.0, 2.0, False),
    # convolution_dim_numbers_input_spatial_dims_2
    97: (0.0, 3.0, False),
    # convolution_dim_numbers_input_spatial_dims_3
    98: (0.0, 0.0, False),
    # convolution_dim_numbers_kernel_input_feature_dim - the dimension number that represents input features in the convolutional kernel (rhs)
    99: (0.0, 4.0, False),
    # convolution_dim_numbers_kernel_output_feature_dim - the dimension number that represents output features in the convolutional kernel (rhs)
    100: (0.0, 4.0, False),
    # convolution_dim_numbers_kernel_spatial_dims_0
    101: (0.0, 1.0, False),
    # convolution_dim_numbers_kernel_spatial_dims_1
    102: (0.0, 2.0, False),
    # convolution_dim_numbers_kernel_spatial_dims_2
    103: (0.0, 3.0, False),
    # convolution_dim_numbers_kernel_spatial_dims_3
    104: (0.0, 0.0, False),
    # convolution_dim_numbers_output_batch_dim - the dimension number that represents batch in the output
    105: (0.0, 3.0, False),
    # convolution_dim_numbers_output_feature_dim - the dimension number that represents features in the output
    106: (0.0, 4.0, False),
}
LAYOUT_FEATS = {
    # layout_minor_to_major_0
    134: (0.0, 5.0, False),
    # layout_minor_to_major_1
    135: (0.0, 4.0, False),
    # layout_minor_to_major_2
    136: (0.0, 5.0, False),
    # layout_minor_to_major_3
    137: (0.0, 3.0, False),
    # layout_minor_to_major_4
    138: (0.0, 2.0, False),
    # layout_minor_to_major_5
    139: (0.0, 0.0, False),
}
