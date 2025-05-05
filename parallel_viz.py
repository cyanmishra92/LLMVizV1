import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st
import json
from typing import Dict, Any, List, Tuple

class ParallelizationVisualizer:
    """Create visualizations of parallelization strategies for LLM inference."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with model configuration."""
        self.config = config
        self.num_layers = config.get("num_layers", 12)
        self.hidden_size = config.get("hidden_size", 768)
        self.num_attention_heads = config.get("num_attention_heads", 12)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.mlp_hidden_dim = config.get("mlp_hidden_dim", 3072)
        self.vocab_size = config.get("vocab_size", 30000)
        
        # Default colors
        self.colors = {
            'device': 'rgba(99, 110, 250, 0.2)',
            'device_border': 'rgba(99, 110, 250, 1.0)',
            'tensor': 'rgba(239, 85, 59, 0.7)',
            'data': 'rgba(0, 204, 150, 0.7)',
            'communication': 'rgba(171, 99, 250, 0.7)',
            'compute': 'rgba(255, 161, 90, 0.7)',
            'text': 'black'
        }
    
    def _create_device_box(self, x0, y0, x1, y1, name="Device", device_id=None):
        """Create a box representing a computing device (GPU/TPU)."""
        text = f"{name} {device_id}" if device_id is not None else name
        shape = dict(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=self.colors['device_border'], width=2),
            fillcolor=self.colors['device'],
            layer="below"
        )
        
        annotation = dict(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=text,
            showarrow=False,
            font=dict(size=14, color=self.colors['text'])
        )
        
        return shape, annotation
    
    def _create_tensor_box(self, x0, y0, x1, y1, name="Tensor", color=None):
        """Create a box representing a tensor."""
        if color is None:
            color = self.colors['tensor']
        
        shape = dict(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="rgba(0,0,0,0.5)", width=1),
            fillcolor=color,
            layer="above"
        )
        
        annotation = dict(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=name,
            showarrow=False,
            font=dict(size=12, color=self.colors['text'])
        )
        
        return shape, annotation
    
    def _create_arrow(self, x0, y0, x1, y1, color=None, width=2, dash=None):
        """Create an arrow representing data flow."""
        if color is None:
            color = self.colors['communication']
        return dict(
            type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=color, width=width, dash=dash),
            layer="above"
        )
    
    def _create_animation_frames(self, base_fig, steps):
        """Create animation frames from a sequence of steps."""
        frames = []
        for i, step in enumerate(steps):
            frame = go.Frame(
                data=step.get('data', []),
                layout=step.get('layout', {}),
                name=f"frame{i}"
            )
            frames.append(frame)
        return frames

    def visualize_batch_parallelism(self, num_devices=4, batch_size=16, seq_length=512):
        """Visualize batch parallelism with independent processing of batches across devices."""
        # Calculate batch split across devices
        batch_per_device = batch_size // num_devices
        remainder = batch_size % num_devices
        device_batches = [batch_per_device + (1 if i < remainder else 0) for i in range(num_devices)]
        
        # Create figure
        fig = go.Figure()
        
        # Define device dimensions
        device_width = 200
        device_height = 150
        device_spacing = 50
        total_width = num_devices * (device_width + device_spacing)
        
        # Add devices
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            device_shape, device_annotation = self._create_device_box(
                x0=x_center - device_width/2,
                y0=100,
                x1=x_center + device_width/2,
                y1=100 + device_height,
                name="GPU",
                device_id=i
            )
            fig.add_shape(device_shape)
            fig.add_annotation(device_annotation)
            
            # Add batch indicator
            fig.add_annotation(
                x=x_center,
                y=100 + device_height/2,
                text=f"Batch: {device_batches[i]} sequences",
                showarrow=False,
                font=dict(size=12)
            )
        
        # Add input data distribution visualization
        input_shape, input_annotation = self._create_tensor_box(
            x0=total_width/2 - 150,
            y0=20,
            x1=total_width/2 + 150,
            y1=70,
            name=f"Input Batch (Size={batch_size})",
            color=self.colors['data']
        )
        fig.add_shape(input_shape)
        fig.add_annotation(input_annotation)
        
        # Add arrows showing data distribution
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            fig.add_shape(self._create_arrow(
                x0=total_width/2,
                y0=70,
                x1=x_center,
                y1=100,
                color=self.colors['data']
            ))
        
        # Add annotations explaining batch parallelism
        fig.add_annotation(
            x=total_width/2,
            y=300,
            text="Batch Parallelism: Each device processes an independent subset of the batch",
            showarrow=False,
            font=dict(size=16)
        )
        
        # Configure layout
        fig.update_layout(
            title="Batch Parallelism Visualization",
            width=max(800, total_width + 100),
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(range=[0, total_width], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(range=[0, 350], showticklabels=False, showgrid=False, zeroline=False)
        )
        
        # Add animation frames (different distribution schemes)
        frames = []
        # Create 5 different batch distribution scenarios
        for frame_idx in range(5):
            new_batches = [batch_per_device + (frame_idx + i) % num_devices for i in range(num_devices)]
            new_frame = go.Frame(
                name=f"distribution_{frame_idx}",
                layout={}
            )
            
            # Update batch indicators
            for i in range(num_devices):
                x_center = (i + 0.5) * (device_width + device_spacing)
                new_frame.layout.updatemenus = [{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 1000, "redraw": True}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "type": "buttons"
                }]
                
                new_frame.layout.annotations = [
                    {
                        "x": x_center,
                        "y": 100 + device_height/2,
                        "text": f"Batch: {new_batches[i]} sequences",
                        "showarrow": False,
                        "font": {"size": 12}
                    }
                ]
            frames.append(new_frame)
        
        # Add frames to figure
        fig.frames = frames
        
        return fig
    
    def visualize_data_parallelism(self, num_devices=4, batch_size=16):
        """Visualize data parallelism with model replication and gradient aggregation."""
        # Create figure
        fig = go.Figure()
        
        # Define device dimensions
        device_width = 150
        device_height = 300
        device_spacing = 50
        total_width = num_devices * (device_width + device_spacing)
        
        # Add devices with model copies
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            
            # Device box
            device_shape, device_annotation = self._create_device_box(
                x0=x_center - device_width/2,
                y0=100,
                x1=x_center + device_width/2,
                y1=100 + device_height,
                name="GPU",
                device_id=i
            )
            fig.add_shape(device_shape)
            fig.add_annotation(device_annotation)
            
            # Model representation inside device
            model_height = device_height * 0.8
            model_width = device_width * 0.8
            model_shape, model_annotation = self._create_tensor_box(
                x0=x_center - model_width/2,
                y0=100 + (device_height - model_height)/2,
                x1=x_center + model_width/2,
                y1=100 + (device_height - model_height)/2 + model_height,
                name=f"Model Copy",
                color=self.colors['compute']
            )
            fig.add_shape(model_shape)
            fig.add_annotation(model_annotation)
            
            # Add mini-batch indicator
            fig.add_annotation(
                x=x_center,
                y=100 + device_height/6,
                text=f"Batch Fraction: {batch_size // num_devices}",
                showarrow=False,
                font=dict(size=10)
            )
        
        # Add parameter synchronization visualization
        param_sync_y = 100 + device_height + 50
        sync_shape, sync_annotation = self._create_tensor_box(
            x0=total_width/2 - 200,
            y0=param_sync_y,
            x1=total_width/2 + 200,
            y1=param_sync_y + 40,
            name="Parameter Synchronization",
            color=self.colors['communication']
        )
        fig.add_shape(sync_shape)
        fig.add_annotation(sync_annotation)
        
        # Add arrows for parameter synchronization
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            # Arrow going up (gradient)
            fig.add_shape(self._create_arrow(
                x0=x_center,
                y0=100 + device_height,
                x1=x_center,
                y1=param_sync_y,
                color=self.colors['communication'],
                dash="dot"
            ))
            # Arrow going down (updated params)
            fig.add_shape(self._create_arrow(
                x0=x_center,
                y0=param_sync_y + 40,
                x1=x_center,
                y1=100 + device_height + 10,
                color=self.colors['communication']
            ))
        
        # Add input data visualization
        input_shape, input_annotation = self._create_tensor_box(
            x0=total_width/2 - 200,
            y0=20,
            x1=total_width/2 + 200,
            y1=60,
            name=f"Input Batch (Size={batch_size})",
            color=self.colors['data']
        )
        fig.add_shape(input_shape)
        fig.add_annotation(input_annotation)
        
        # Add arrows for data distribution
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            fig.add_shape(self._create_arrow(
                x0=total_width/2,
                y0=60,
                x1=x_center,
                y1=100,
                color=self.colors['data']
            ))
        
        # Add annotations explaining data parallelism
        fig.add_annotation(
            x=total_width/2,
            y=param_sync_y + 80,
            text="Data Parallelism: Model replicated across devices, each processing a subset of data",
            showarrow=False,
            font=dict(size=16)
        )
        
        # Configure layout
        fig.update_layout(
            title="Data Parallelism Visualization",
            width=max(800, total_width + 100),
            height=param_sync_y + 120,
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(range=[0, total_width], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(range=[0, param_sync_y + 120], showticklabels=False, showgrid=False, zeroline=False)
        )
        
        return fig
    
    def visualize_tensor_parallelism(self, num_devices=4):
        """Visualize tensor parallelism with partitioned model operators."""
        # Create figure
        fig = go.Figure()
        
        # Define device dimensions
        device_width = 180
        device_height = 300
        device_spacing = 40
        total_width = num_devices * (device_width + device_spacing)
        
        # Define layer dimensions to visualize
        attn_height = 80
        mlp_height = 80
        layer_spacing = 30
        
        # Add devices with partitioned model components
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            
            # Device box
            device_shape, device_annotation = self._create_device_box(
                x0=x_center - device_width/2,
                y0=100,
                x1=x_center + device_width/2,
                y1=100 + device_height,
                name="GPU",
                device_id=i
            )
            fig.add_shape(device_shape)
            fig.add_annotation(device_annotation)
            
            # Attention head partitioning (assume heads are evenly distributed)
            heads_per_device = self.num_attention_heads // num_devices
            
            # Add attention partition
            attn_y = 120
            attn_shape, attn_annotation = self._create_tensor_box(
                x0=x_center - device_width * 0.4,
                y0=attn_y,
                x1=x_center + device_width * 0.4,
                y1=attn_y + attn_height,
                name=f"Attn Heads: {i*heads_per_device}-{(i+1)*heads_per_device-1}",
                color="rgba(255, 161, 90, 0.7)"
            )
            fig.add_shape(attn_shape)
            fig.add_annotation(attn_annotation)
            
            # MLP partitioning (assume hidden dim is partitioned)
            mlp_partition_size = self.mlp_hidden_dim // num_devices
            mlp_y = attn_y + attn_height + layer_spacing
            mlp_shape, mlp_annotation = self._create_tensor_box(
                x0=x_center - device_width * 0.4,
                y0=mlp_y,
                x1=x_center + device_width * 0.4,
                y1=mlp_y + mlp_height,
                name=f"MLP: {i*mlp_partition_size}-{(i+1)*mlp_partition_size-1}",
                color="rgba(99, 110, 250, 0.7)"
            )
            fig.add_shape(mlp_shape)
            fig.add_annotation(mlp_annotation)
            
            # Output partition
            output_y = mlp_y + mlp_height + layer_spacing
            output_height = 40
            hidden_partition_size = self.hidden_size // num_devices
            output_shape, output_annotation = self._create_tensor_box(
                x0=x_center - device_width * 0.4,
                y0=output_y,
                x1=x_center + device_width * 0.4,
                y1=output_y + output_height,
                name=f"Output: {i*hidden_partition_size}-{(i+1)*hidden_partition_size-1}",
                color="rgba(0, 204, 150, 0.7)"
            )
            fig.add_shape(output_shape)
            fig.add_annotation(output_annotation)
        
        # Add communication arrows for cross-attention
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            attn_y = 120 + attn_height/2
            
            # Add horizontal communication lines for attention (all-to-all)
            for j in range(num_devices):
                if i != j:
                    x_target = (j + 0.5) * (device_width + device_spacing)
                    fig.add_shape(self._create_arrow(
                        x0=x_center, 
                        y0=attn_y,
                        x1=x_target,
                        y1=attn_y,
                        color=self.colors['communication'],
                        dash="dot",
                        width=1
                    ))
        
        # Add communication arrows for final output reduction
        output_sync_y = 100 + device_height + 30
        sync_shape, sync_annotation = self._create_tensor_box(
            x0=total_width/2 - 120,
            y0=output_sync_y,
            x1=total_width/2 + 120,
            y1=output_sync_y + 30,
            name="All-Reduce Operation",
            color=self.colors['communication']
        )
        fig.add_shape(sync_shape)
        fig.add_annotation(sync_annotation)
        
        # Connect devices to all-reduce
        output_y = mlp_y + mlp_height + layer_spacing + output_height/2
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            fig.add_shape(self._create_arrow(
                x0=x_center,
                y0=output_y,
                x1=x_center,
                y1=output_sync_y + 15,
                color=self.colors['communication']
            ))
        
        # Add annotations explaining tensor parallelism
        fig.add_annotation(
            x=total_width/2,
            y=output_sync_y + 60,
            text="Tensor Parallelism: Model operators split across devices, communication during forward/backward pass",
            showarrow=False,
            font=dict(size=14)
        )
        
        # Configure layout
        fig.update_layout(
            title="Tensor Parallelism Visualization",
            width=max(800, total_width + 100),
            height=output_sync_y + 100,
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(range=[0, total_width], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(range=[0, output_sync_y + 100], showticklabels=False, showgrid=False, zeroline=False)
        )
        
        return fig
    
    def visualize_pipeline_parallelism(self, num_devices=4, pipeline_chunks=4):
        """Visualize pipeline parallelism with model layers distributed across devices."""
        # Create figure with subplots for animation frames
        fig = make_subplots(rows=1, cols=1)
        
        # Define device dimensions
        device_width = 120
        device_height = 200
        device_spacing = 40
        total_width = num_devices * (device_width + device_spacing)
        
        # Calculate layers per device
        layers_per_device = [self.num_layers // num_devices + (1 if i < self.num_layers % num_devices else 0) 
                            for i in range(num_devices)]
        
        # Create base figure with devices
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            
            # Device box
            device_shape = dict(
                type="rect",
                x0=x_center - device_width/2,
                y0=100,
                x1=x_center + device_width/2,
                y1=100 + device_height,
                line=dict(color=self.colors['device_border'], width=2),
                fillcolor=self.colors['device'],
                layer="below"
            )
            fig.add_shape(device_shape)
            
            # Device label
            fig.add_annotation(
                x=x_center,
                y=100 + device_height + 15,
                text=f"GPU {i}",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Layer allocation
            fig.add_annotation(
                x=x_center,
                y=100 + device_height/2,
                text=f"Layers {sum(layers_per_device[:i])}-{sum(layers_per_device[:i+1])-1}",
                showarrow=False,
                font=dict(size=12)
            )
        
        # Add pipeline visualization - use animation to show the bubbling pipeline execution
        
        # Define microbatch dimensions
        mb_width = device_width * 0.6
        mb_height = 30
        mb_spacing = 10
        
        # Define colors for different microbatches
        mb_colors = ['rgba(239, 85, 59, 0.7)', 
                    'rgba(0, 204, 150, 0.7)', 
                    'rgba(171, 99, 250, 0.7)', 
                    'rgba(255, 161, 90, 0.7)']
        
        # Define time steps for simulation
        time_steps = num_devices + pipeline_chunks # Approximation of pipeline steps
        
        # Create frames for animation
        frames = []
        
        for t in range(time_steps):
            frame_data = []
            frame_shapes = []
            frame_annotations = []
            
            # Current positions of microbatches
            for mb in range(pipeline_chunks):
                device_pos = t - mb  # Which device this microbatch is on at time t
                
                if 0 <= device_pos < num_devices:
                    x_center = (device_pos + 0.5) * (device_width + device_spacing)
                    y_pos = 100 + device_height/2 - mb_height - mb * (mb_height + mb_spacing)
                    
                    # Add microbatch box
                    mb_shape = dict(
                        type="rect",
                        x0=x_center - mb_width/2,
                        y0=y_pos,
                        x1=x_center + mb_width/2,
                        y1=y_pos + mb_height,
                        line=dict(color="rgba(0,0,0,0.5)", width=1),
                        fillcolor=mb_colors[mb % len(mb_colors)],
                        layer="above"
                    )
                    frame_shapes.append(mb_shape)
                    
                    # Add microbatch label
                    frame_annotations.append(dict(
                        x=x_center,
                        y=y_pos + mb_height/2,
                        text=f"MB {mb}",
                        showarrow=False,
                        font=dict(size=10, color="black")
                    ))
            
            # Add time step indicator
            frame_annotations.append(dict(
                x=total_width/2,
                y=50,
                text=f"Time Step: {t}",
                showarrow=False,
                font=dict(size=14)
            ))
            
            # Create frame
            frame = go.Frame(
                name=f"step_{t}",
                layout=go.Layout(
                    shapes=frame_shapes,
                    annotations=frame_annotations
                )
            )
            frames.append(frame)
        
        # Add initial state
        for s in frames[0].layout.shapes:
            fig.add_shape(s)
        for a in frames[0].layout.annotations:
            fig.add_annotation(a)
        
        # Add annotation explaining pipeline parallelism
        fig.add_annotation(
            x=total_width/2,
            y=350,
            text="Pipeline Parallelism: Model layers distributed across devices, microbatches flow through pipeline",
            showarrow=False,
            font=dict(size=14)
        )
        
        # Configure layout
        fig.update_layout(
            title="Pipeline Parallelism Visualization",
            width=max(800, total_width + 100),
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(range=[0, total_width], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(range=[0, 400], showticklabels=False, showgrid=False, zeroline=False),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 1000, "redraw": True}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "type": "buttons",
                "direction": "right",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Time Step: "
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f"step_{i}"], {"frame": {"duration": 0, "redraw": True}}],
                        "label": f"{i}",
                        "method": "animate"
                    } for i in range(time_steps)
                ]
            }]
        )
        
        # Add frames to figure
        fig.frames = frames
        
        return fig
    
    def visualize_zero_parallelism(self, num_devices=4):
        """
        Visualize ZeRO (Zero Redundancy Optimizer) parallelism with sharded 
        optimizer states, gradients, and possibly parameters.
        """
        # Create figure
        fig = go.Figure()
        
        # Define device dimensions
        device_width = 180
        device_height = 300
        device_spacing = 50
        total_width = num_devices * (device_width + device_spacing)
        
        # Add devices
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            
            # Device box
            device_shape, device_annotation = self._create_device_box(
                x0=x_center - device_width/2,
                y0=100,
                x1=x_center + device_width/2,
                y1=100 + device_height,
                name="GPU",
                device_id=i
            )
            fig.add_shape(device_shape)
            fig.add_annotation(device_annotation)
            
            # Model representation (equivalent copy of FP16 parameters)
            model_height = 70
            model_y = 120
            model_shape, model_annotation = self._create_tensor_box(
                x0=x_center - device_width * 0.4,
                y0=model_y,
                x1=x_center + device_width * 0.4,
                y1=model_y + model_height,
                name="Model (FP16)",
                color="rgba(99, 110, 250, 0.7)"
            )
            fig.add_shape(model_shape)
            fig.add_annotation(model_annotation)
            
            # Optimizer shard
            opt_height = 40
            opt_y = model_y + model_height + 20
            parameter_shard_size = int(self.num_layers * 12 * self.hidden_size // num_devices)  # Approximate
            opt_shape, opt_annotation = self._create_tensor_box(
                x0=x_center - device_width * 0.4,
                y0=opt_y,
                x1=x_center + device_width * 0.4,
                y1=opt_y + opt_height,
                name=f"Optimizer Shard {i+1}/{num_devices}",
                color="rgba(239, 85, 59, 0.7)"
            )
            fig.add_shape(opt_shape)
            fig.add_annotation(opt_annotation)
            
            # Gradient shard
            grad_height = 40
            grad_y = opt_y + opt_height + 20
            grad_shape, grad_annotation = self._create_tensor_box(
                x0=x_center - device_width * 0.4,
                y0=grad_y,
                x1=x_center + device_width * 0.4,
                y1=grad_y + grad_height,
                name=f"Gradient Shard {i+1}/{num_devices}",
                color="rgba(0, 204, 150, 0.7)"
            )
            fig.add_shape(grad_shape)
            fig.add_annotation(grad_annotation)
            
            # Parameter shard (ZeRO-3)
            param_height = 40
            param_y = grad_y + grad_height + 20
            param_shape, param_annotation = self._create_tensor_box(
                x0=x_center - device_width * 0.4,
                y0=param_y,
                x1=x_center + device_width * 0.4,
                y1=param_y + param_height,
                name=f"FP32 Param Shard {i+1}/{num_devices}",
                color="rgba(171, 99, 250, 0.7)"
            )
            fig.add_shape(param_shape)
            fig.add_annotation(param_annotation)
        
        # Add communication visualization for all-gather and reduce-scatter
        comm_y = 100 + device_height + 40
        comm_shape, comm_annotation = self._create_tensor_box(
            x0=total_width/2 - 200,
            y0=comm_y,
            x1=total_width/2 + 200,
            y1=comm_y + 50,
            name="All-Gather → Compute → Reduce-Scatter",
            color=self.colors['communication']
        )
        fig.add_shape(comm_shape)
        fig.add_annotation(comm_annotation)
        
        # Add arrows connecting devices to communication
        for i in range(num_devices):
            x_center = (i + 0.5) * (device_width + device_spacing)
            # Upward arrow
            fig.add_shape(self._create_arrow(
                x0=x_center,
                y0=100 + device_height,
                x1=x_center,
                y1=comm_y,
                color=self.colors['communication']
            ))
            # Downward arrow
            fig.add_shape(self._create_arrow(
                x0=x_center,
                y0=comm_y + 50,
                x1=x_center,
                y1=100 + device_height,
                color=self.colors['communication'],
                dash="dot"
            ))
        
        # Add annotations explaining ZeRO parallelism
        fig.add_annotation(
            x=total_width/2,
            y=comm_y + 80,
            text="ZeRO Parallelism: Optimizer states, gradients, and parameters sharded across devices",
            showarrow=False,
            font=dict(size=16)
        )
        
        fig.add_annotation(
            x=total_width/2,
            y=comm_y + 110,
            text="→ Reduced memory per device",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.add_annotation(
            x=total_width/2,
            y=comm_y + 130,
            text="→ Increased communication during training",
            showarrow=False,
            font=dict(size=14)
        )
        
        # Configure layout
        fig.update_layout(
            title="ZeRO (Zero Redundancy Optimizer) Parallelism Visualization",
            width=max(800, total_width + 100),
            height=comm_y + 170,
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(range=[0, total_width], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(range=[0, comm_y + 170], showticklabels=False, showgrid=False, zeroline=False)
        )
        
        return fig
    
    def visualize_hybrid_parallelism(self, num_pipeline_stages=2, num_tensor_parallel=2, num_data_parallel=2):
        """Visualize hybrid parallelism combining pipeline, tensor, and data parallelism."""
        # Total number of devices
        total_devices = num_pipeline_stages * num_tensor_parallel * num_data_parallel
        
        # Create figure
        fig = go.Figure()
        
        # Define device dimensions
        device_width = 140
        device_height = 160
        h_spacing = 30
        v_spacing = 80
        
        # Define total dimensions
        grid_width = num_tensor_parallel * (device_width + h_spacing)
        grid_height = num_pipeline_stages * (device_height + v_spacing)
        dp_spacing = 300  # Space between data parallel copies
        
        # Iterate through all devices
        for dp in range(num_data_parallel):
            dp_offset_x = dp * (grid_width + dp_spacing)
            
            # Add data parallel group label
            fig.add_annotation(
                x=dp_offset_x + grid_width/2,
                y=20,
                text=f"Data Parallel Group {dp+1}",
                showarrow=False,
                font=dict(size=16)
            )
            
            for pp in range(num_pipeline_stages):
                for tp in range(num_tensor_parallel):
                    # Calculate device position
                    x_center = dp_offset_x + (tp + 0.5) * (device_width + h_spacing)
                    y_center = 80 + (pp + 0.5) * (device_height + v_spacing)
                    
                    # Device box
                    device_id = dp * (num_pipeline_stages * num_tensor_parallel) + pp * num_tensor_parallel + tp
                    device_shape, device_annotation = self._create_device_box(
                        x0=x_center - device_width/2,
                        y0=y_center - device_height/2,
                        x1=x_center + device_width/2,
                        y1=y_center + device_height/2,
                        name="GPU",
                        device_id=device_id
                    )
                    fig.add_shape(device_shape)
                    fig.add_annotation(device_annotation)
                    
                    # Add device group labels
                    fig.add_annotation(
                        x=x_center,
                        y=y_center - device_height/4,
                        text=f"TP Rank: {tp}",
                        showarrow=False,
                        font=dict(size=10)
                    )
                    
                    fig.add_annotation(
                        x=x_center,
                        y=y_center,
                        text=f"PP Stage: {pp}",
                        showarrow=False,
                        font=dict(size=10)
                    )
                    
                    fig.add_annotation(
                        x=x_center,
                        y=y_center + device_height/4,
                        text=f"DP Rank: {dp}",
                        showarrow=False,
                        font=dict(size=10)
                    )
                    
                    # Add tensor parallel connections (horizontal)
                    if tp < num_tensor_parallel - 1:
                        next_x = dp_offset_x + (tp + 1.5) * (device_width + h_spacing)
                        fig.add_shape(self._create_arrow(
                            x0=x_center + device_width/2,
                            y0=y_center,
                            x1=next_x - device_width/2,
                            y1=y_center,
                            color="rgba(239, 85, 59, 0.7)"
                        ))
                    
                    # Add pipeline connections (vertical)
                    if pp < num_pipeline_stages - 1:
                        next_y = 80 + (pp + 1.5) * (device_height + v_spacing)
                        fig.add_shape(self._create_arrow(
                            x0=x_center,
                            y0=y_center + device_height/2,
                            x1=x_center,
                            y1=next_y - device_height/2,
                            color="rgba(0, 204, 150, 0.7)"
                        ))
        
        # Add data parallel connections (between corresponding devices in different DP groups)
        for pp in range(num_pipeline_stages):
            for tp in range(num_tensor_parallel):
                for dp in range(num_data_parallel-1):
                    # Calculate device positions
                    x1 = dp * (grid_width + dp_spacing) + (tp + 0.5) * (device_width + h_spacing) + device_width/2
                    x2 = (dp+1) * (grid_width + dp_spacing) + (tp + 0.5) * (device_width + h_spacing) - device_width/2
                    y = 80 + (pp + 0.5) * (device_height + v_spacing)
                    
                    fig.add_shape(self._create_arrow(
                        x0=x1,
                        y0=y,
                        x1=x2,
                        y1=y,
                        color="rgba(171, 99, 250, 0.7)",
                        dash="dot"
                    ))
        
        # Add legend for communication types
        legend_y = 80 + grid_height + 40
        fig.add_shape(
            type="rect",
            x0=grid_width/2 - 250,
            y0=legend_y,
            x1=grid_width/2 - 230,
            y1=legend_y + 20,
            line=dict(color="rgba(239, 85, 59, 1.0)", width=2),
            fillcolor="rgba(239, 85, 59, 0.7)",
            layer="above"
        )
        fig.add_annotation(
            x=grid_width/2 - 150,
            y=legend_y + 10,
            text="Tensor Parallelism Communication",
            showarrow=False,
            font=dict(size=12)
        )
        
        fig.add_shape(
            type="rect",
            x0=grid_width/2 + 0,
            y0=legend_y,
            x1=grid_width/2 + 20,
            y1=legend_y + 20,
            line=dict(color="rgba(0, 204, 150, 1.0)", width=2),
            fillcolor="rgba(0, 204, 150, 0.7)",
            layer="above"
        )
        fig.add_annotation(
            x=grid_width/2 + 100,
            y=legend_y + 10,
            text="Pipeline Parallelism Communication",
            showarrow=False,
            font=dict(size=12)
        )
        
        fig.add_shape(
            type="rect",
            x0=grid_width/2 + 250,
            y0=legend_y,
            x1=grid_width/2 + 270,
            y1=legend_y + 20,
            line=dict(color="rgba(171, 99, 250, 1.0)", width=2),
            fillcolor="rgba(171, 99, 250, 0.7)",
            layer="above"
        )
        fig.add_annotation(
            x=grid_width/2 + 350,
            y=legend_y + 10,
            text="Data Parallelism Communication",
            showarrow=False,
            font=dict(size=12)
        )
        
        # Add title and explanation
        fig.add_annotation(
            x=(num_data_parallel * grid_width + (num_data_parallel-1) * dp_spacing) / 2,
            y=legend_y + 60,
            text="Hybrid Parallelism: Combining Pipeline, Tensor, and Data Parallelism",
            showarrow=False,
            font=dict(size=16, color="black")
        )
        
        fig.add_annotation(
            x=(num_data_parallel * grid_width + (num_data_parallel-1) * dp_spacing) / 2,
            y=legend_y + 90,
            text=f"Total Devices: {total_devices} = {num_pipeline_stages} (PP) × {num_tensor_parallel} (TP) × {num_data_parallel} (DP)",
            showarrow=False,
            font=dict(size=14)
        )
        
        # Configure layout
        total_width = num_data_parallel * grid_width + (num_data_parallel-1) * dp_spacing
        fig.update_layout(
            title="Hybrid Parallelism Visualization",
            width=max(800, total_width + 100),
            height=legend_y + 130,
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(range=[0, total_width], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(range=[0, legend_y + 130], showticklabels=False, showgrid=False, zeroline=False)
        )
        
        return fig


def create_parallelism_visualization(config, parallelism_type, **kwargs):
    """Create a parallelization visualization based on the specified type."""
    visualizer = ParallelizationVisualizer(config)
    
    if parallelism_type == "batch":
        return visualizer.visualize_batch_parallelism(**kwargs)
    elif parallelism_type == "data":
        return visualizer.visualize_data_parallelism(**kwargs)
    elif parallelism_type == "tensor":
        return visualizer.visualize_tensor_parallelism(**kwargs)
    elif parallelism_type == "pipeline":
        return visualizer.visualize_pipeline_parallelism(**kwargs)
    elif parallelism_type == "zero":
        return visualizer.visualize_zero_parallelism(**kwargs)
    elif parallelism_type == "hybrid":
        return visualizer.visualize_hybrid_parallelism(**kwargs)
    else:
        raise ValueError(f"Unknown parallelism type: {parallelism_type}")