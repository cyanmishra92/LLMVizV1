# -*- coding: utf-8 -*-
"""Streamlit web application for LLM Inference Simulation."""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
# Import the core calculation logic
from llm_calc import LLMCalculator, format_num, format_bytes
import io
# Import the visualization module
from model_viz import ModelVisualizer, create_model_visualizations

# --- Page Config ---
st.set_page_config(layout="wide", page_title="LLM Inference Simulator")

# --- Title & Description ---
st.title("LLM Inference Simulator: Prefill & Decode Analysis")
st.markdown("""
An interactive tool to estimate compute (FLOPs) and memory usage (Parameters, KV Cache, Activations)
for Large Language Model inference stages. Adjust parameters in the sidebar or upload a config file
to see how they impact resource requirements. Calculation breakdowns are available in the respective tabs.
""")

# --- Helper Function for Displaying Breakdowns ---
def display_breakdown(title, data, value_key, formula_key):
    with st.expander(f"Show Calculation: {title}"):
        st.markdown(f"**Formula:** `{data[formula_key]}`")
        st.markdown(f"**Result:** `{format_num(data[value_key]) if isinstance(data[value_key], (int, float)) and value_key != 'gb' else format_bytes(data[value_key]) if value_key == 'bytes' else data[value_key]}`")

def display_flops_breakdown(title, data):
    with st.expander(f"Show Calculation: {title}"):
        st.markdown("**FLOPs per Component (per layer):**")
        for k, v in data["breakdown_per_layer"].items():
            if k != "total_per_layer":
                st.markdown(f"  - `{k}`: {format_num(v)} FLOPs (`{data['formulas'][k]}`)")
        if "output_lm_head_flops" in data:
             st.markdown(f"  - `Output LM Head`: {format_num(data['output_lm_head_flops'])} FLOPs (`{data['formulas']['output_lm_head']}`)")
        st.markdown(f"**Total FLOPs:** {format_num(data['total_flops'])} (`{data['formulas']['total']}`)")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

input_method = st.sidebar.radio("Input Method", ["Manual Parameter Input", "Upload Configuration File (JSON)"])

# Initialize variables
config = None
model_name = "Custom Model"
num_layers, hidden_size, num_attention_heads, mlp_hidden_dim, vocab_size, precision = None, None, None, None, None, None

if input_method == "Manual Parameter Input":
    st.sidebar.subheader("Model Architecture")
    num_layers = st.sidebar.number_input("Number of Layers (L)", min_value=1, value=32, step=1)
    hidden_size = st.sidebar.number_input("Hidden Size (H)", min_value=64, value=4096, step=64)
    num_attention_heads = st.sidebar.number_input("Number of Attention Heads", min_value=1, value=32, step=1)
    suggested_mlp_dim = int(hidden_size * 8 / 3) # SwiGLU approximation
    if hidden_size % num_attention_heads != 0:
        st.sidebar.warning(f"Hidden size ({hidden_size}) is not perfectly divisible by the number of attention heads ({num_attention_heads}). Head dimension will be floor({hidden_size / num_attention_heads}).")

    mlp_hidden_dim = st.sidebar.number_input("MLP Hidden Dimension (M)", min_value=64, value=suggested_mlp_dim, step=64, help="Often 4*H for standard MLP, or ~2.7*H for SwiGLU.")
    vocab_size = st.sidebar.number_input("Vocabulary Size (V)", min_value=1000, value=32000, step=1000)
    precision = st.sidebar.selectbox("Precision", ["FP16", "BF16", "FP32", "INT8"], index=0)
    config = { # Store manual inputs in a config-like dict
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "mlp_hidden_dim": mlp_hidden_dim,
        "vocab_size": vocab_size,
        "precision": precision
    }

elif input_method == "Upload Configuration File (JSON)":
    st.sidebar.subheader("Upload Config")
    uploaded_file = st.sidebar.file_uploader("Choose a JSON config file", type="json")

    try:
        with open("config_example.json", "r") as f:
            example_json_content = f.read()
        st.sidebar.download_button(
            label="Download Example Config",
            data=example_json_content,
            file_name="config_example.json",
            mime="application/json"
        )
    except FileNotFoundError:
        st.sidebar.warning("Example config file not found.")

    if uploaded_file is not None:
        try:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            uploaded_config = json.loads(string_data)

            if "architecture" in uploaded_config and isinstance(uploaded_config["architecture"], dict):
                arch = uploaded_config["architecture"]
                config = {
                    "num_layers": int(arch.get("num_layers", 32)),
                    "hidden_size": int(arch.get("hidden_size", 4096)),
                    "num_attention_heads": int(arch.get("num_attention_heads", 32)),
                    "mlp_hidden_dim": int(arch.get("mlp_hidden_dim", int(arch.get("hidden_size", 4096) * 8 / 3))),
                    "vocab_size": int(arch.get("vocab_size", 32000)),
                    "precision": str(uploaded_config.get("precision", "FP16")).upper()
                }
                model_name = uploaded_config.get("model_name", "Uploaded Model")
                st.sidebar.success(f"Loaded config for: {model_name}")
                st.sidebar.json(config)
            else:
                st.sidebar.error("Invalid JSON structure. Expected keys like 'architecture' with model params.")
                config = None
        except json.JSONDecodeError:
            st.sidebar.error("Invalid JSON file. Could not decode.")
            config = None
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")
            config = None
    else:
        st.sidebar.info("Upload a JSON file with model parameters.")

st.sidebar.subheader("Runtime Parameters")
sequence_length = st.sidebar.slider("Prompt Sequence Length (S_prompt)", min_value=1, max_value=8192, value=512, step=1)
batch_size = st.sidebar.slider("Batch Size (B)", min_value=1, max_value=256, value=1, step=1)
num_tokens_to_generate = st.sidebar.slider("Tokens to Generate (N_gen)", min_value=1, max_value=4096, value=128, step=1)

# --- Main Area Output ---
st.header("Simulation Results")

# Create tabs for the application
tab_summary, tab_params, tab_prefill, tab_decode, tab_viz, tab_arch, tab_model_viz, tab_parallel = st.tabs([
    "📊 Summary", "⚙️ Model Parameters", "➡️ Prefill Stage", "🔄 Decode Stage",
    "📈 Visualizations", "🏛️ Model Structure", "🖼️ Model Visualization", "⚡ Parallelization"
])

run_simulation = st.sidebar.button("Run Simulation")

if run_simulation and config:
    try:
        # Instantiate calculator
        calc = LLMCalculator(
            num_layers=config["num_layers"],
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_attention_heads"],
            mlp_hidden_dim=config["mlp_hidden_dim"],
            vocab_size=config["vocab_size"],
            precision=config["precision"]
        )

        # Get full details
        details = calc.get_full_details(batch_size, sequence_length, num_tokens_to_generate)
        summary = details["summary"]

        # Model Visualization Tab
        with tab_model_viz:
            st.subheader("Model Architecture Visualization")

            # Create visualizations
            visualizations = create_model_visualizations(config)

            # Display the visualizations
            st.subheader("Full Model Structure")
            st.plotly_chart(visualizations["full_model"], use_container_width=True)

            # Create a layer selector
            selected_layer = st.slider("Select Layer for Detailed View",
                                    min_value=1,
                                    max_value=config["num_layers"],
                                    value=1)

            # Show layer details for the selected layer
            visualizer = ModelVisualizer(config)
            st.plotly_chart(visualizer.visualize_layer_details(selected_layer-1),
                            use_container_width=True)

            # Show the attention mechanism
            st.subheader("Multi-Head Attention Visualization")
            st.plotly_chart(visualizations["attention"], use_container_width=True)

            # Add expander with explanation
            with st.expander("About the Visualizations"):
                st.markdown("""
                These visualizations represent:
                1. **Full Model Structure**: Shows all layers in the model from input to output.
                2. **Layer Detail**: Displays the components within a transformer layer, including the attention mechanism and feed-forward network.
                3. **Multi-Head Attention**: Illustrates how the attention mechanism works with multiple heads in parallel.

                The visualizations are simplified representations and do not show all computational details.
                """)

        # Summary Tab
        with tab_summary:
            st.subheader(f"Input Configuration ({model_name})")
            st.json({
                "Model Architecture": config,
                "Runtime Parameters": {
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "num_tokens_to_generate": num_tokens_to_generate
                }
            })

            st.subheader("Overall Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Total Parameters", f"{format_num(summary['total_params'])}")
            display_breakdown("Total Parameters", details["params"]["total_params"], "params", "formula")

            col2.metric("Parameter Memory", f"{format_bytes(summary['param_memory_gb'] * 1024**3)}")
            display_breakdown("Parameter Memory", details["params"]["param_memory"], "bytes", "formula")

            col1.metric("Total Prefill FLOPs", f"{format_num(summary['prefill_flops'])} FLOPs")
            display_flops_breakdown("Total Prefill FLOPs", details["prefill"]["flops"]) # Use new function

            col2.metric(f"Total Decode FLOPs ({num_tokens_to_generate} tokens)", f"{format_num(summary['total_decode_flops'])} FLOPs")
            # Breakdown for total decode is complex, maybe show avg per token breakdown
            avg_token_flops_details = calc.get_decode_flops_details_per_token(batch_size, sequence_length + num_tokens_to_generate // 2) # Use avg seq len
            display_flops_breakdown(f"Avg Decode FLOPs/Token ({format_num(summary['avg_decode_flops_per_token'])} FLOPs)", avg_token_flops_details)

            col1.metric("Peak KV Cache Memory", f"{format_bytes(summary['peak_kv_cache_gb'] * 1024**3)}")
            kv_details_peak = calc.get_kv_cache_details(batch_size, sequence_length + num_tokens_to_generate)
            display_breakdown("Peak KV Cache Memory", kv_details_peak, "bytes", "formula")

            col2.metric("Estimated Peak Total Memory", f"{format_bytes(summary['estimated_peak_total_memory_gb'] * 1024**3)}")
            with st.expander("Show Calculation: Estimated Peak Total Memory"):
                st.markdown(f"Peak Memory = max(Prefill Peak, Decode Peak)")
                prefill_peak = summary['param_memory_gb'] + summary['peak_prefill_activation_gb'] + details['prefill']['kv_cache_memory']['gb']
                decode_peak = summary['param_memory_gb'] + summary['peak_decode_activation_gb'] + summary['peak_kv_cache_gb']
                st.markdown(f"  - Prefill Peak ≈ Params + Prefill Acts + Prefill KV ≈ {summary['param_memory_gb']:.2f} + {summary['peak_prefill_activation_gb']:.2f} + {details['prefill']['kv_cache_memory']['gb']:.2f} ≈ {prefill_peak:.2f} GB")
                st.markdown(f"  - Decode Peak ≈ Params + Decode Acts (Peak) + Decode KV (Peak) ≈ {summary['param_memory_gb']:.2f} + {summary['peak_decode_activation_gb']:.2f} + {summary['peak_kv_cache_gb']:.2f} ≈ {decode_peak:.2f} GB")
                st.markdown(f"Result: {max(prefill_peak, decode_peak):.2f} GB")

            col1.metric("Peak Prefill Activation Memory", f"{format_bytes(summary['peak_prefill_activation_gb'] * 1024**3)}")
            display_breakdown("Peak Prefill Activation Memory", details["prefill"]["activation_memory"], "bytes", "formula")

            col2.metric("Peak Decode Activation Memory (per step)", f"{format_bytes(summary['peak_decode_activation_gb'] * 1024**3)}")
            # Find the step with peak activation memory for breakdown
            peak_act_step_details = max(details["decode"]["per_token_calcs"], key=lambda x: x["activation_memory"]["gb"])["activation_memory"]
            display_breakdown("Peak Decode Activation Memory (per step)", peak_act_step_details, "bytes", "formula")

        # Parameters Tab
        with tab_params:
            st.subheader("Parameter Breakdown")
            param_detail_list = []
            for k, v in details["params"].items():
                if k not in ["total_params", "param_memory"]:
                    param_detail_list.append({"Component": k, "Count": v["params"], "Formula": v["formula"]})

            param_df = pd.DataFrame(param_detail_list)
            param_df["Percentage"] = (param_df["Count"] / summary["total_params"] * 100).round(2)
            param_df = param_df.set_index("Component")
            # Add total row
            total_row = pd.DataFrame([{"Count": summary["total_params"], "Percentage": 100.00, "Formula": details["params"]["total_params"]["formula"] }], index=["TOTAL"])
            param_df = pd.concat([param_df, total_row])

            st.dataframe(param_df.style.format({"Count": "{:,.0f}", "Percentage": "{:.2f}%"}))

            st.metric("Total Parameter Memory", f"{format_bytes(summary['param_memory_gb'] * 1024**3)}")
            display_breakdown("Total Parameter Memory", details["params"]["param_memory"], "bytes", "formula")

        # Prefill Tab
        with tab_prefill:
            st.subheader("Prefill Stage Analysis")
            st.metric("Total FLOPs", f"{format_num(summary['prefill_flops'])} FLOPs")
            display_flops_breakdown("Total Prefill FLOPs", details["prefill"]["flops"]) # Use new function

            st.metric("Peak Activation Memory", f"{format_bytes(summary['peak_prefill_activation_gb'] * 1024**3)}")
            display_breakdown("Peak Prefill Activation Memory", details["prefill"]["activation_memory"], "bytes", "formula")

            st.metric("KV Cache Generated", f"{format_bytes(details['prefill']['kv_cache_memory']['gb'] * 1024**3)}")
            display_breakdown("KV Cache Generated", details["prefill"]["kv_cache_memory"], "bytes", "formula")

            st.markdown("*(Prefill processes the input prompt in parallel. It's typically compute-bound due to large matrix multiplications.)*")

        # Decode Tab
        with tab_decode:
            st.subheader("Decode Stage Analysis (Autoregressive Generation)")
            st.metric(f"Total FLOPs ({num_tokens_to_generate} tokens)", f"{format_num(summary['total_decode_flops'])} FLOPs")
            st.metric("Average FLOPs per Token", f"{format_num(summary['avg_decode_flops_per_token'])} FLOPs")
            # Show breakdown for average token
            display_flops_breakdown(f"Avg Decode FLOPs/Token", avg_token_flops_details)

            st.metric("Peak Activation Memory (per step)", f"{format_bytes(summary['peak_decode_activation_gb'] * 1024**3)}")
            display_breakdown("Peak Decode Activation Memory (per step)", peak_act_step_details, "bytes", "formula")

            st.metric("Peak KV Cache Memory (at end)", f"{format_bytes(summary['peak_kv_cache_gb'] * 1024**3)}")
            display_breakdown("Peak KV Cache Memory", kv_details_peak, "bytes", "formula")

            st.markdown("*(Decode generates tokens one by one. It's often memory-bandwidth bound due to loading weights and the growing KV cache.)*")

        # Visualization Tab
        with tab_viz:
            st.subheader("Visualizations")

            # 1. FLOPs Comparison
            flops_data = pd.DataFrame({
                'Stage': ['Prefill', f'Decode ({num_tokens_to_generate} tokens)'],
                'TFLOPs': [summary['prefill_flops'] / 1e12, summary['total_decode_flops'] / 1e12]
            })
            fig_flops = px.bar(flops_data, x='Stage', y='TFLOPs', title='Prefill vs. Total Decode FLOPs', text_auto='.2f')
            st.plotly_chart(fig_flops, use_container_width=True)

            # 2. Memory Breakdown (Peak)
            mem_breakdown_data = pd.DataFrame([
                 {'Category': 'Parameters', 'Memory (GB)': summary['param_memory_gb']},
                 {'Category': 'Max KV Cache', 'Memory (GB)': summary['peak_kv_cache_gb']},
                 {'Category': 'Max Activations', 'Memory (GB)': max(summary['peak_prefill_activation_gb'], summary['peak_decode_activation_gb'])},
            ])
            fig_mem = px.bar(mem_breakdown_data, x=['Estimated Peak Total'] * len(mem_breakdown_data), y='Memory (GB)', color='Category',
                             title=f"Estimated Peak Memory Breakdown ({summary['estimated_peak_total_memory_gb']:.2f} GB Total)",
                             text_auto='.2f')
            fig_mem.update_layout(xaxis_title=None, xaxis_showticklabels=False)
            st.plotly_chart(fig_mem, use_container_width=True)

            # 3. KV Cache Growth
            kv_cache_growth_gb = [details['prefill']['kv_cache_memory']['gb']] + details['decode']['cumulative']['kv_cache_list_gb']
            kv_df = pd.DataFrame({
                'Generated Tokens': list(range(num_tokens_to_generate + 1)),
                'KV Cache (GB)': kv_cache_growth_gb
            })
            fig_kv = px.line(kv_df, x='Generated Tokens', y='KV Cache (GB)', title='KV Cache Size Growth During Generation', markers=True)
            fig_kv.update_layout(yaxis_title="KV Cache Size (GB)")
            st.plotly_chart(fig_kv, use_container_width=True)

            # 4. Cumulative Decode FLOPs
            decode_flops_growth_tflops = [0] + [f / 1e12 for f in details['decode']['cumulative']['decode_flops_list_tflops']]
            cumulative_flops = [sum(decode_flops_growth_tflops[1:i+1]) for i in range(num_tokens_to_generate + 1)]
            decode_flops_df = pd.DataFrame({
                'Generated Tokens': list(range(num_tokens_to_generate + 1)),
                'Cumulative Decode TFLOPs': cumulative_flops
            })
            fig_dec_flops = px.line(decode_flops_df, x='Generated Tokens', y='Cumulative Decode TFLOPs', title='Cumulative Decode FLOPs During Generation', markers=True)
            st.plotly_chart(fig_dec_flops, use_container_width=True)

        with tab_arch:
            st.subheader("Assumed Model Structure (Decoder-Only Transformer)")
            st.markdown("""
            This simulator assumes a standard decoder-only transformer architecture, common in models like GPT, Llama, etc.
            The key components and data flow are:

            1.  **Input Embedding:** Input tokens (prompt) are converted into vectors of `hidden_size` (H).
                *   Params: `vocab_size` (V) * `hidden_size` (H)

            2.  **Transformer Blocks (repeated L times):**
                *   **Layer Normalization:** Applied before attention and MLP layers (Pre-LN assumed).
                *   **Multi-Head Self-Attention (MHA):**
                    *   Input projected into Query (Q), Key (K), Value (V) matrices (`num_heads` times).
                        *   Params (QKV Proj): 3 * H * H
                    *   Attention scores calculated: `softmax(Q @ K.T / sqrt(head_dim)) @ V`.
                        *   FLOPs (Prefill): ~ 4 * B * S^2 * H
                        *   FLOPs (Decode): ~ 4 * B * S_total * H (per token)
                    *   Output projection combines head outputs back to `hidden_size`.
                        *   Params (Output Proj): H * H
                *   **Residual Connection:** Output of attention added to its input.
                *   **Layer Normalization:** Applied before the MLP layer.
                *   **Feed-Forward Network (MLP):** Typically a 2 or 3-layer network (SwiGLU assumed here).
                    *   Expands `hidden_size` to `mlp_hidden_dim` (M) and contracts back.
                    *   Params (SwiGLU): 3 * H * M (Gate, Up, Down projections)
                    *   FLOPs (Prefill): ~ 6 * B * S * H * M
                    *   FLOPs (Decode): ~ 6 * B * H * M (per token)
                *   **Residual Connection:** Output of MLP added to its input.

            3.  **Final Layer Normalization:** Applied after the last transformer block.

            4.  **Output Layer (Unembedding / LM Head):** Projects the final hidden state to `vocab_size` to get logits for the next token.
                *   Params: H * V

            **Key Inference Stages:**
            *   **Prefill:** Processes the entire input prompt (`sequence_length`) in parallel to generate the initial KV cache. Compute-intensive.
            *   **Decode:** Generates output tokens one by one, using the KV cache from previous steps. Memory-bandwidth intensive.
            """)

        # Model Visualization Tab content is already provided above

        with tab_parallel:
            st.subheader("LLM Parallelization Strategies")

            st.markdown("""
            This tab visualizes different parallelization strategies used for LLM inference and training.
            Each strategy has different tradeoffs in terms of memory usage, computation, and communication overhead.
            """)

            # Add a selector for parallelism type
            parallelism_type = st.selectbox(
            "Select Parallelization Strategy",
            ["batch", "data", "tensor", "pipeline", "zero", "hybrid"],
            format_func=lambda x: {
                "batch": "Batch Parallelism",
                "data": "Data Parallelism",
                "tensor": "Tensor Parallelism",
                "pipeline": "Pipeline Parallelism",
                "zero": "ZeRO (Zero Redundancy Optimizer)",
                "hybrid": "Hybrid Parallelism"
            }[x]
            )

            # Add parameter controls based on selected parallelism
            col1, col2 = st.columns(2)

            params = {}
            if parallelism_type == "batch":
                params["num_devices"] = col1.slider("Number of Devices", min_value=2, max_value=8, value=4)
                params["batch_size"] = col2.slider("Total Batch Size", min_value=4, max_value=64, value=16, step=4)

            elif parallelism_type == "data":
                params["num_devices"] = col1.slider("Number of Devices", min_value=2, max_value=8, value=4)
                params["batch_size"] = col2.slider("Total Batch Size", min_value=4, max_value=64, value=16, step=4)

            elif parallelism_type == "tensor":
                params["num_devices"] = col1.slider("Number of Devices", min_value=2, max_value=8, value=4)

            elif parallelism_type == "pipeline":
                params["num_devices"] = col1.slider("Number of Pipeline Stages", min_value=2, max_value=8, value=4)
                params["pipeline_chunks"] = col2.slider("Number of Microbatches", min_value=1, max_value=8, value=4)

            elif parallelism_type == "zero":
                params["num_devices"] = col1.slider("Number of Devices", min_value=2, max_value=8, value=4)

            elif parallelism_type == "hybrid":
                params["num_pipeline_stages"] = col1.slider("Pipeline Stages", min_value=1, max_value=4, value=2)
                params["num_tensor_parallel"] = col1.slider("Tensor Parallel Size", min_value=1, max_value=4, value=2)
                params["num_data_parallel"] = col2.slider("Data Parallel Size", min_value=1, max_value=4, value=2)

            # Display visualization based on the selected type
            fig = create_parallelism_visualization(config, parallelism_type, **params)
            st.plotly_chart(fig, use_container_width=True)

            # Add explanations for each parallelism type
            with st.expander("About This Parallelization Strategy"):
                if parallelism_type == "batch":
                    st.markdown("""
                    **Batch Parallelism** is the simplest form of parallelism where a batch of inputs is divided among multiple devices.

                    **How it works:**
                    - The input batch is split into smaller batches
                    - Each device processes its portion independently with a full copy of the model
                    - No communication is needed during processing
                    - Results are combined after processing

                    **Benefits:**
                    - Simple to implement
                    - Linear scaling of throughput
                    - No communication overhead during computation

                    **Limitations:**
                    - Each device needs a full copy of the model
                    - Memory requirements don't decrease
                    - Not suitable for models that don't fit on a single device
                    """)

                elif parallelism_type == "data":
                    st.markdown("""
                    **Data Parallelism** is a common approach for training where the model is replicated across devices, but each processes different data.

                    **How it works:**
                    - Full model copy exists on each device
                    - Each device processes different samples (forward pass)
                    - Gradients are synchronized across devices (backward pass)
                    - Parameter updates are synchronized

                    **Benefits:**
                    - Simple to implement
                    - Good scaling for computation
                    - Works well when model fits on a single device

                    **Limitations:**
                    - Does not reduce per-device memory requirements
                    - Communication overhead increases with model size
                    - Not suitable for models larger than single device memory
                    """)

                elif parallelism_type == "tensor":
                    st.markdown("""
                    **Tensor Parallelism** (also known as horizontal parallelism) splits individual model operators across devices.

                    **How it works:**
                    - Mathematical operations in the model are partitioned
                    - Common approach: Split attention heads across devices
                    - Each device computes a portion of each layer's operations
                    - Results are synchronized during computation

                    **Benefits:**
                    - Reduces memory requirements per device
                    - Allows fitting larger models than single-device memory
                    - Balanced computation across devices

                    **Limitations:**
                    - Requires significant communication during computation
                    - May introduce performance bottlenecks from communication
                    - More complex to implement

                    *Common implementations: Megatron-LM, Mesh TensorFlow*
                    """)

                elif parallelism_type == "pipeline":
                    st.markdown("""
                    **Pipeline Parallelism** splits the model across devices layer-wise, creating a processing pipeline.

                    **How it works:**
                    - Model layers are divided among devices
                    - Input is processed sequentially through the pipeline
                    - Multiple inputs can be in different pipeline stages simultaneously
                    - Microbatches are used to increase pipeline efficiency

                    **Benefits:**
                    - Reduces memory requirements per device
                    - Can scale to very large models
                    - Reduces activation memory requirements

                    **Limitations:**
                    - Pipeline bubbles cause inefficiency
                    - Balancing computation across stages is challenging
                    - May introduce latency for individual requests

                    *Common implementations: GPipe, PipeDream*
                    """)

                elif parallelism_type == "zero":
                    st.markdown("""
                    **ZeRO (Zero Redundancy Optimizer)** shards optimizer states, gradients, and optionally parameters across devices.

                    **How it works:**
                    - ZeRO Stage 1: Optimizer states are partitioned
                    - ZeRO Stage 2: Gradients are also partitioned
                    - ZeRO Stage 3: Parameters are also partitioned
                    - All-gather collectives bring necessary parameters to each device during computation

                    **Benefits:**
                    - Linear reduction in memory requirements with number of devices
                    - Allows training models many times larger than single device memory
                    - Maintains computational efficiency of data parallelism

                    **Limitations:**
                    - Increased communication volume
                    - Communication may become bottleneck
                    - Complex implementation

                    *Implemented in: DeepSpeed, Megatron-DeepSpeed, PyTorch FSDP*
                    """)

                elif parallelism_type == "hybrid":
                    st.markdown("""
                    **Hybrid Parallelism** combines multiple parallelism strategies for maximum scalability.

                    **How it works:**
                    - Combines 2 or more parallelism strategies simultaneously
                    - Common combination: 3D Parallelism (Data + Pipeline + Tensor)
                    - Each strategy addresses different scaling challenges

                    **Benefits:**
                    - Can scale to extremely large models (trillions of parameters)
                    - Better balance of computation vs. communication
                    - Adaptable to different hardware configurations

                    **Limitations:**
                    - Very complex to implement and debug
                    - Requires careful tuning for optimal performance
                    - Requires sophisticated orchestration

                    *Implemented in: Megatron-DeepSpeed, Microsoft's Turing-NLG/GPT, Google's PaLM architecture*
                    """)

            # Add additional resource links
            with st.expander("Additional Resources"):
                st.markdown("""
                **Research Papers:**
                - [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
                - [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
                - [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
                - [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

                **Frameworks:**
                - [DeepSpeed](https://github.com/microsoft/DeepSpeed)
                - [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
                - [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
                - [Alpa](https://github.com/alpa-projects/alpa)
                """)

    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
        st.exception(e) # Show traceback for debugging

elif run_simulation and not config:
    st.warning("Please configure model parameters manually or upload a config file before running the simulation.")

else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation'.")

# Add footer
st.markdown("---")
st.markdown("""
### About
This LLM Inference Simulator helps estimate computational resources required for running large language models.
Built with Streamlit, it provides interactive visualization of model architecture and resource requirements.

For more information, check out the [GitHub repository](https://github.com/yourusername/llm-inference-simulator).
""")
