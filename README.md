# The official repository of the paper 

# The normalizing flow
The normalizing flow is implemented in [**jammy_flows**](https://github.com/thoglu/jammy_flows) version 1.1.0.
It needs to be set it up [in conditional mode](https://thoglu.github.io/jammy_flows/usage/training.html#conditional-pdf) (`conditional_input_dim=N`, where `N` is a previous hidden dimension that maps to the normalizing flow parameters) and with the right [settings](https://thoglu.github.io/jammy_flows/usage/suggested_settings.html#spherical-pdf-2-sphere) from the paper. In the following, the options for the model are given.

<details>
<summary>Baseline jammy-flows config - used in most trainings</summary>

```python
opt_dict=dict()
opt_dict["f"]=dict()
opt_dict["f"]["add_vertical_rq_spline_flow"]=1
opt_dict["f"]["spline_num_basis_functions"]=-1
opt_dict["f"]["vertical_smooth"]=1
opt_dict["f"]["vertical_flow_defs"]="rr"
opt_dict["f"]["circular_flow_defs"] = "oo"
opt_dict["f"]["vertical_fix_boundary_derivative"]=1
opt_dict["f"]["min_kappa"]=1e-10
opt_dict["f"]["kappa_prediction"]="direct_log_real_bounded"
opt_dict["f"]["kappa_clamping"]=0
opt_dict["f"]["vertical_restrict_max_min_width_height_ratio"]=-1.0
opt_dict["f"]["vertical_fix_first_width_n_height_to_zero"]=1
opt_dict["f"]["vertical_independent_width_height_parametrization"]=1 
opt_dict["f"]["add_circular_rq_spline_flow"]=1 # add circle flow
opt_dict["f"]["circular_add_rotation"]=0 # no extra rotation on circle flow
opt_dict["f"]["vertical_also_fix_second_width_to_zero"]=1
opt_dict["f"]["rotation_mode"]="householder"

pdf=jammy_flows.pdf("s2", "fffffffffffffff", options_overwrite=opt_dict, conditional_input_dim=*N*)
pdf.double() # double precision usually necessary to avoid numerical issues
```

</details>
```

<details>
<summary>Pure von-Mises Fisher config - used in a small subset of trainings</summary>

```python
opt_dict=dict()
opt_dict["f"]=dict()
opt_dict["f"]["add_vertical_rq_spline_flow"]=0
opt_dict["f"]["min_kappa"]=1e-10
opt_dict["f"]["kappa_prediction"]="direct_log_real_bounded"
opt_dict["f"]["kappa_clamping"]=0
opt_dict["f"]["rotation_mode"]="householder"

pdf=jammy_flows.pdf("s2", "f", options_overwrite=opt_dict, conditional_input_dim=*N*)
pdf.double() # double precision usually necessary to avoid numerical issues
```

</details>
```



## Transformer encoding

The paper uses the transformer implementation [here](https://github.com/icecube/learning_ground_base/encoders/mh_attention_encoder_new.py) from the repo [learning_ground_base](https://github.com/icecube/learning_ground_base).

The top configs from the paper are shown below, and can be directly used with the [model implementation](https://github.com/icecube/learning_ground_base/encoders/mh_attention_encoder_new.py).

<details>
<summary>Top shower configuration (shower model ID 1)</summary>

```yaml
settings.input_dim: 27
settings.output_dim: 192
settings.io_mlp_hidden_dims: '256'
settings.io_add_skip_connection: 1
settings.attn_do_perlayer_out_projection: 1
settings.skip_input_projection: 0
settings.attn_computational_dim: 96
settings.attn_num_layers: 20
settings.attn_num_heads_per_layer: 1
settings.attn_use_layer_norm_1: 1
settings.attn_use_layer_norm_2: 1
settings.attn_use_extra_layer_norm: 0
settings.attn_layer_norm_first: 1
settings.attn_use_residual_addition: 3
settings.dtype: 'float32'
settings.attn_projection_type: 'joint_qkv'
settings.attn_inprojection_mlp_dims: '512'
settings.attn_inprojection_add_skip: 0
settings.attn_inprojection_add_mean_diff: 0
settings.attn_perform_final_mapping: 0
settings.attn_package: "xformer" 
settings.attn_dropout: 0.0
settings.attn_internal_mlp_dim: 512
settings.attn_use_weighted_mean: 0
settings.attn_aggregation_mode: 'mean_n_diagonal'
settings.attn_use_computational_class_token: 0
settings.attn_original_input_position_feature_number: -1
settings.attn_add_original_input_to_feature_input: 0
settings.attn_abs_position_mode: 'none'
settings.attn_abs_position_layer_indices: '0'
settings.attn_rel_position_layer_indices: '0'
settings.attn_rel_position_input_feeding_type: 0
settings.attn_rel_position_mode_value: 'none'
settings.attn_rel_position_hidden_dim: 64
settings.attn_rel_position_use_skip_connection: 1
settings.attn_rel_position_as_parallel_to_normal_track: 0
settings.attn_rel_position_max_computational_dim: 100
settings.force_sdpa_precision: ''
settings.attn_num_token_types: 1
```

</details>

<details>
<summary>Top through-going track configuration (track model ID 1)</summary>

```yaml
settings.input_dim: 27
settings.output_dim: 192
settings.io_mlp_hidden_dims: '256'
settings.io_add_skip_connection: 1
settings.attn_do_perlayer_out_projection: 1
settings.skip_input_projection: 0
settings.attn_computational_dim: 96
settings.attn_num_layers: 20
settings.attn_num_heads_per_layer: 1
settings.attn_use_layer_norm_1: 1
settings.attn_use_layer_norm_2: 1
settings.attn_use_extra_layer_norm: 0
settings.attn_layer_norm_first: 1
settings.attn_use_residual_addition: 3
settings.dtype: 'float32'
settings.attn_projection_type: 'joint_qkv'
settings.attn_inprojection_mlp_dims: '512'
settings.attn_inprojection_add_skip: 0
settings.attn_inprojection_add_mean_diff: 0
settings.attn_perform_final_mapping: 0
settings.attn_package: "xformer"
settings.attn_dropout: 0.0
settings.attn_internal_mlp_dim: 512
settings.attn_use_weighted_mean: 0
settings.attn_aggregation_mode: 'mean_n_diagonal'
settings.attn_use_computational_class_token: 0
settings.attn_original_input_position_feature_number: -1
settings.attn_add_original_input_to_feature_input: 0
settings.attn_abs_position_mode: 'sinusoidal'
settings.attn_abs_position_layer_indices: '0'
settings.attn_rel_position_layer_indices: '0'
settings.attn_rel_position_input_feeding_type: 0
settings.attn_rel_position_mode_value: 'none'
settings.attn_rel_position_hidden_dim: 64
settings.attn_rel_position_use_skip_connection: 1
settings.attn_rel_position_as_parallel_to_normal_track: 0
settings.attn_rel_position_max_computational_dim: 100
settings.force_sdpa_precision: ''
settings.attn_num_token_types: 1

```

</details>

<details>
<summary>Top starting track configuration (track model ID 2)</summary>

```yaml
settings.input_dim: 27
settings.output_dim: 192
settings.io_mlp_hidden_dims: '256'
settings.io_add_skip_connection: 1
settings.attn_do_perlayer_out_projection: 1
settings.skip_input_projection: 0
settings.attn_computational_dim: 96
settings.attn_num_layers: 20
settings.attn_num_heads_per_layer: 1
settings.attn_use_layer_norm_1: 1
settings.attn_use_layer_norm_2: 1
settings.attn_use_extra_layer_norm: 0
settings.attn_layer_norm_first: 1
settings.attn_use_residual_addition: 3
settings.dtype: 'float32'
settings.attn_projection_type: 'joint_qkv'
settings.attn_inprojection_mlp_dims: '512'
settings.attn_inprojection_add_skip: 0
settings.attn_inprojection_add_mean_diff: 1
settings.attn_perform_final_mapping: 0
settings.attn_package: "xformer" 
settings.attn_dropout: 0.0
settings.attn_internal_mlp_dim: 512
settings.attn_use_weighted_mean: 0
settings.attn_aggregation_mode: 'mean_n_diagonal'
settings.attn_use_computational_class_token: 0
settings.attn_original_input_position_feature_number: -1
settings.attn_add_original_input_to_feature_input: 0
settings.attn_abs_position_mode: 'sinusoidal'
settings.attn_abs_position_layer_indices: '0'
settings.attn_rel_position_layer_indices: '0'
settings.attn_rel_position_input_feeding_type: 0
settings.attn_rel_position_mode_value: 'none'
settings.attn_rel_position_hidden_dim: 64
settings.attn_rel_position_use_skip_connection: 1
settings.attn_rel_position_as_parallel_to_normal_track: 0
settings.attn_rel_position_max_computational_dim: 100
settings.force_sdpa_precision: ''
settings.attn_num_token_types: 1
```

</details>