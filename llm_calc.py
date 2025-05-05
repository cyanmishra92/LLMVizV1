# -*- coding: utf-8 -*-
"""Core calculation logic for LLM inference simulation."""

import math

# Helper function for formatting numbers
def format_num(n):
    if n >= 1e12:
        return f"{n / 1e12:.2f} T"
    if n >= 1e9:
        return f"{n / 1e9:.2f} B"
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.2f} k"
    return f"{n:,.0f}"

def format_bytes(b):
    if b >= 1024**4:
        return f"{b / 1024**4:.2f} TB"
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.2f} MB"
    if b >= 1024:
        return f"{b / 1024:.2f} KB"
    return f"{b:,.0f} Bytes"

class LLMCalculator:
    """Calculates FLOPs and memory for LLM inference."""

    def __init__(self, num_layers, hidden_size, num_attention_heads, mlp_hidden_dim, vocab_size, precision="FP16"):
        """Initialize with model hyperparameters."""
        self.L = num_layers
        self.H = hidden_size
        self.n_heads = num_attention_heads
        self.M = mlp_hidden_dim
        self.V = vocab_size
        self.precision = precision
        self.bytes_per_param = self._get_bytes_per_param(precision)

        if self.H % self.n_heads != 0:
            print(f"Warning: hidden_size ({self.H}) may not be divisible by num_attention_heads ({self.n_heads})")
        self.head_dim = self.H // self.n_heads

    def _get_bytes_per_param(self, precision):
        """Get bytes per parameter based on precision string."""
        precision_map = {
            "FP32": 4,
            "FP16": 2,
            "BF16": 2,
            "INT8": 1,
        }
        return precision_map.get(precision.upper(), 2)

    def get_param_details(self):
        """Calculate parameter counts and return detailed breakdown with formulas."""
        details = {}

        # Embedding
        embed_params = self.V * self.H
        details["embedding"] = {
            "params": embed_params,
            "formula": f"V * H = {format_num(self.V)} * {format_num(self.H)} = {format_num(embed_params)}"
        }

        # Attention (per layer)
        attn_qkv_params_layer = 3 * self.H * self.H
        attn_out_params_layer = self.H * self.H
        details["attn_qkv_proj (per layer)"] = {
            "params": attn_qkv_params_layer,
            "formula": f"3 * H * H = 3 * {format_num(self.H)} * {format_num(self.H)} = {format_num(attn_qkv_params_layer)}"
        }
        details["attn_output_proj (per layer)"] = {
            "params": attn_out_params_layer,
            "formula": f"H * H = {format_num(self.H)} * {format_num(self.H)} = {format_num(attn_out_params_layer)}"
        }

        # MLP (per layer, SwiGLU)
        mlp_gate_up_params_layer = 2 * self.H * self.M
        mlp_down_params_layer = self.M * self.H
        details["mlp_gate_up_proj (per layer)"] = {
            "params": mlp_gate_up_params_layer,
            "formula": f"2 * H * M = 2 * {format_num(self.H)} * {format_num(self.M)} = {format_num(mlp_gate_up_params_layer)}"
        }
        details["mlp_down_proj (per layer)"] = {
            "params": mlp_down_params_layer,
            "formula": f"M * H = {format_num(self.M)} * {format_num(self.H)} = {format_num(mlp_down_params_layer)}"
        }

        # Output Layer
        output_params = self.H * self.V
        details["output_lm_head"] = {
            "params": output_params,
            "formula": f"H * V = {format_num(self.H)} * {format_num(self.V)} = {format_num(output_params)}"
        }

        # Totals
        total_params = (
            embed_params
            + self.L * (attn_qkv_params_layer + attn_out_params_layer + mlp_gate_up_params_layer + mlp_down_params_layer)
            + output_params
        )
        details["total_params"] = {
            "params": total_params,
            "formula": f"Embedding + L * (Attn + MLP) + Output = {format_num(embed_params)} + {self.L} * ({format_num(attn_qkv_params_layer + attn_out_params_layer)} + {format_num(mlp_gate_up_params_layer + mlp_down_params_layer)}) + {format_num(output_params)} = {format_num(total_params)}"
        }

        # Memory
        memory_bytes = total_params * self.bytes_per_param
        details["param_memory"] = {
            "bytes": memory_bytes,
            "gb": memory_bytes / (1024**3),
            "formula": f"Total Params * Bytes/Param = {format_num(total_params)} * {self.bytes_per_param} = {format_bytes(memory_bytes)}"
        }

        return details

    def get_kv_cache_details(self, batch_size, sequence_length):
        """Calculate KV cache memory and return details with formula."""
        B = batch_size
        S = sequence_length
        memory_bytes = B * S * self.L * 2 * self.H * self.bytes_per_param
        details = {
            "bytes": memory_bytes,
            "gb": memory_bytes / (1024**3),
            "formula": f"B * S * L * 2 * H * Bytes/Param = {B} * {S} * {self.L} * 2 * {format_num(self.H)} * {self.bytes_per_param} = {format_bytes(memory_bytes)}"
        }
        return details

    def get_prefill_flops_details(self, batch_size, sequence_length):
        """Calculate prefill FLOPs and return details with formula breakdown."""
        B = batch_size
        S = sequence_length
        L = self.L
        H = self.H
        M = self.M

        # Attention FLOPs (per layer)
        qkv_flops = 6 * B * S * H**2
        scores_flops = 2 * B * S**2 * H
        score_v_flops = 2 * B * S**2 * H
        output_proj_flops = 2 * B * S * H**2
        attn_flops_per_layer = qkv_flops + scores_flops + score_v_flops + output_proj_flops

        # MLP FLOPs (per layer, SwiGLU)
        gate_up_flops = 4 * B * S * H * M
        down_flops = 2 * B * S * M * H
        mlp_flops_per_layer = gate_up_flops + down_flops

        # Total
        total_flops = L * (attn_flops_per_layer + mlp_flops_per_layer)

        details = {
            "total_flops": total_flops,
            "breakdown_per_layer": {
                "attn_qkv_proj": qkv_flops,
                "attn_scores (QK^T)": scores_flops,
                "attn_score_v (AV)": score_v_flops,
                "attn_output_proj": output_proj_flops,
                "mlp_gate_up_proj": gate_up_flops,
                "mlp_down_proj": down_flops,
                "total_per_layer": attn_flops_per_layer + mlp_flops_per_layer
            },
            "formulas": {
                "attn_qkv_proj": f"6 * B * S * H^2 = 6 * {B} * {S} * {format_num(H)}^2 = {format_num(qkv_flops)}",
                "attn_scores (QK^T)": f"2 * B * S^2 * H = 2 * {B} * {S}^2 * {format_num(H)} = {format_num(scores_flops)}",
                "attn_score_v (AV)": f"2 * B * S^2 * H = 2 * {B} * {S}^2 * {format_num(H)} = {format_num(score_v_flops)}",
                "attn_output_proj": f"2 * B * S * H^2 = 2 * {B} * {S} * {format_num(H)}^2 = {format_num(output_proj_flops)}",
                "mlp_gate_up_proj": f"4 * B * S * H * M = 4 * {B} * {S} * {format_num(H)} * {format_num(M)} = {format_num(gate_up_flops)}",
                "mlp_down_proj": f"2 * B * S * M * H = 2 * {B} * {S} * {format_num(M)} * {format_num(H)} = {format_num(down_flops)}",
                "total": f"L * (Attn + MLP) = {L} * ({format_num(attn_flops_per_layer)} + {format_num(mlp_flops_per_layer)}) = {format_num(total_flops)}"
            }
        }
        return details

    def get_decode_flops_details_per_token(self, batch_size, total_sequence_length):
        """Calculate decode FLOPs per token and return details with formula breakdown."""
        B = batch_size
        S_total = total_sequence_length
        L = self.L
        H = self.H
        M = self.M
        V = self.V

        # Attention FLOPs (per layer)
        qkv_flops = 6 * B * H**2
        scores_flops = 2 * B * H * S_total
        score_v_flops = 2 * B * H * S_total
        output_proj_flops = 2 * B * H**2
        attn_flops_per_layer = qkv_flops + scores_flops + score_v_flops + output_proj_flops

        # MLP FLOPs (per layer, SwiGLU)
        gate_up_flops = 4 * B * H * M
        down_flops = 2 * B * M * H
        mlp_flops_per_layer = gate_up_flops + down_flops

        # Output LM Head
        output_flops = B * 1 * (2 * H * V)

        # Total
        total_flops_layers = L * (attn_flops_per_layer + mlp_flops_per_layer)
        total_flops = total_flops_layers + output_flops

        details = {
            "total_flops": total_flops,
            "breakdown_per_layer": {
                "attn_qkv_proj": qkv_flops,
                "attn_scores (QK^T)": scores_flops,
                "attn_score_v (AV)": score_v_flops,
                "attn_output_proj": output_proj_flops,
                "mlp_gate_up_proj": gate_up_flops,
                "mlp_down_proj": down_flops,
                "total_per_layer": attn_flops_per_layer + mlp_flops_per_layer
            },
            "output_lm_head_flops": output_flops,
            "formulas": {
                "attn_qkv_proj": f"6 * B * H^2 = 6 * {B} * {format_num(H)}^2 = {format_num(qkv_flops)}",
                "attn_scores (QK^T)": f"2 * B * H * S_total = 2 * {B} * {format_num(H)} * {S_total} = {format_num(scores_flops)}",
                "attn_score_v (AV)": f"2 * B * H * S_total = 2 * {B} * {format_num(H)} * {S_total} = {format_num(score_v_flops)}",
                "attn_output_proj": f"2 * B * H^2 = 2 * {B} * {format_num(H)}^2 = {format_num(output_proj_flops)}",
                "mlp_gate_up_proj": f"4 * B * H * M = 4 * {B} * {format_num(H)} * {format_num(M)} = {format_num(gate_up_flops)}",
                "mlp_down_proj": f"2 * B * M * H = 2 * {B} * {format_num(M)} * {format_num(H)} = {format_num(down_flops)}",
                "output_lm_head": f"2 * B * H * V = 2 * {B} * {format_num(H)} * {format_num(V)} = {format_num(output_flops)}",
                "total": f"L * (Attn + MLP) + Output = {L} * ({format_num(attn_flops_per_layer)} + {format_num(mlp_flops_per_layer)}) + {format_num(output_flops)} = {format_num(total_flops)}"
            }
        }
        return details

    def get_activation_details(self, batch_size, sequence_length, is_prefill):
        """Estimate activation memory and return details with formula."""
        B = batch_size
        S = sequence_length
        L = self.L
        H = self.H
        M = self.M
        n_heads = self.n_heads

        # Adjust S for decode stage approximation
        S_eff = S if is_prefill else 1
        S_attn = S # Attention calculation always depends on the full sequence length

        if H == 0: return {"bytes": 0, "gb": 0.0, "formula": "H is zero"}

        # Rough estimate formula: B * S_eff * H * L * (1 + n_heads * S_attn / H + M / H) * bytes
        # Terms represent roughly: Layer outputs, Attention scores, MLP intermediate
        try:
            term1 = 1 # Layer outputs
            term2 = n_heads * S_attn / H # Attention scores
            term3 = M / H # MLP intermediate
            factor = term1 + term2 + term3
            memory_bytes = B * S_eff * H * L * factor * self.bytes_per_param
            formula = f"B * S_eff * H * L * (1 + n_heads * S_attn / H + M / H) * Bytes/Param = {B} * {S_eff} * {format_num(H)} * {L} * (1 + {n_heads} * {S_attn} / {H} + {M} / {H}) * {self.bytes_per_param} = {format_bytes(memory_bytes)}"
        except (OverflowError, ZeroDivisionError):
            memory_bytes = float('inf')
            formula = "Calculation resulted in overflow or division by zero."

        details = {
            "bytes": memory_bytes,
            "gb": memory_bytes / (1024**3) if memory_bytes != float('inf') else float('inf'),
            "formula": formula
        }
        return details

    def get_full_details(self, batch_size, sequence_length, num_tokens_to_generate):
        """Calculate and return a comprehensive dictionary with all metrics and breakdowns."""
        B = batch_size
        S_prompt = sequence_length
        N_gen = num_tokens_to_generate

        details = {"inputs": {"B": B, "S_prompt": S_prompt, "N_gen": N_gen, "L": self.L, "H": self.H, "n_heads": self.n_heads, "M": self.M, "V": self.V, "precision": self.precision}}

        # Parameters
        details["params"] = self.get_param_details()
        param_memory_gb = details["params"]["param_memory"]["gb"]

        # Prefill Stage
        details["prefill"] = {}
        details["prefill"]["flops"] = self.get_prefill_flops_details(B, S_prompt)
        details["prefill"]["activation_memory"] = self.get_activation_details(B, S_prompt, is_prefill=True)
        details["prefill"]["kv_cache_memory"] = self.get_kv_cache_details(B, S_prompt)
        prefill_flops = details["prefill"]["flops"]["total_flops"]
        prefill_activation_gb = details["prefill"]["activation_memory"]["gb"]
        prefill_kv_cache_gb = details["prefill"]["kv_cache_memory"]["gb"]

        # Decode Stage
        details["decode"] = {"per_token_calcs": [], "cumulative": {}}
        total_decode_flops = 0
        peak_decode_kv_cache_gb = 0
        peak_decode_activation_gb = 0
        decode_flops_list = []
        kv_cache_list = []

        for i in range(N_gen):
            current_total_seq_len = S_prompt + i + 1
            token_details = {}
            token_details["token_index"] = i + 1
            token_details["total_seq_len"] = current_total_seq_len

            # FLOPs for this token
            flops_details = self.get_decode_flops_details_per_token(B, current_total_seq_len)
            token_details["flops"] = flops_details
            total_decode_flops += flops_details["total_flops"]
            decode_flops_list.append(flops_details["total_flops"])

            # KV cache after this token
            kv_details = self.get_kv_cache_details(B, current_total_seq_len)
            token_details["kv_cache_memory"] = kv_details
            peak_decode_kv_cache_gb = max(peak_decode_kv_cache_gb, kv_details["gb"])
            kv_cache_list.append(kv_details["gb"])

            # Activation memory for this step
            act_details = self.get_activation_details(B, current_total_seq_len, is_prefill=False)
            token_details["activation_memory"] = act_details
            peak_decode_activation_gb = max(peak_decode_activation_gb, act_details["gb"])

            details["decode"]["per_token_calcs"].append(token_details)

        details["decode"]["cumulative"] = {
            "total_flops": total_decode_flops,
            "avg_flops_per_token": total_decode_flops / N_gen if N_gen > 0 else 0,
            "peak_activation_gb": peak_decode_activation_gb,
            "peak_kv_cache_gb": peak_decode_kv_cache_gb,
            "decode_flops_list_tflops": [f / 1e12 for f in decode_flops_list],
            "kv_cache_list_gb": kv_cache_list
        }

        # Total Memory Estimation
        peak_mem_prefill_gb = param_memory_gb + prefill_activation_gb + prefill_kv_cache_gb
        peak_mem_decode_gb = param_memory_gb + peak_decode_activation_gb + peak_decode_kv_cache_gb
        estimated_peak_total_memory_gb = max(peak_mem_prefill_gb, peak_mem_decode_gb)

        details["summary"] = {
            "total_params": details["params"]["total_params"]["params"],
            "param_memory_gb": param_memory_gb,
            "prefill_flops": prefill_flops,
            "total_decode_flops": total_decode_flops,
            "avg_decode_flops_per_token": details["decode"]["cumulative"]["avg_flops_per_token"],
            "peak_prefill_activation_gb": prefill_activation_gb,
            "peak_decode_activation_gb": peak_decode_activation_gb,
            "peak_kv_cache_gb": peak_decode_kv_cache_gb,
            "estimated_peak_total_memory_gb": estimated_peak_total_memory_gb
        }

        return details

# Example Usage (for testing)
if __name__ == '__main__':
    calc = LLMCalculator(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        mlp_hidden_dim=11008,
        vocab_size=32000,
        precision='FP16'
    )
    details = calc.get_full_details(batch_size=1, sequence_length=512, num_tokens_to_generate=128)
    summary = details["summary"]

    print(f"Model: Llama 2 7B (Example)")
    print(f"Precision: {calc.precision} ({calc.bytes_per_param} bytes)")
    print("---")
    print(f"Total Parameters: {format_num(summary['total_params'])} ({details['params']['total_params']['formula']})")
    print(f"Parameter Memory: {format_bytes(details['params']['param_memory']['bytes'])} ({details['params']['param_memory']['formula']})")
    print("---")
    print(f"Prefill Stage (Batch={details['inputs']['B']}, SeqLen={details['inputs']['S_prompt']}):")
    print(f"  FLOPs: {format_num(summary['prefill_flops'])} FLOPs ({details['prefill']['flops']['formulas']['total']})")
    print(f"  Peak Activation Memory: {format_bytes(details['prefill']['activation_memory']['bytes'])} ({details['prefill']['activation_memory']['formula']})")
    print("---")
    print(f"Decode Stage (Batch={details['inputs']['B']}, GenTokens={details['inputs']['N_gen']}):")
    print(f"  Total FLOPs: {format_num(summary['total_decode_flops'])} FLOPs")
    print(f"  Avg FLOPs/Token: {format_num(summary['avg_decode_flops_per_token'])} FLOPs")
    print(f"  Peak Activation Memory (per step): {format_bytes(details['decode']['cumulative']['peak_activation_gb'] * 1024**3)}")
    print(f"  Peak KV Cache Memory: {format_bytes(details['decode']['cumulative']['peak_kv_cache_gb'] * 1024**3)}")
    print("---")
    print(f"Estimated Peak Total Memory: {format_bytes(summary['estimated_peak_total_memory_gb'] * 1024**3)}")
    print("---")

