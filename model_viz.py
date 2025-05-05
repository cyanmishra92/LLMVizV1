# model_viz.py
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, Any

class ModelVisualizer:
    """Create visualizations of transformer model architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with model configuration."""
        self.config = config
        self.num_layers = config.get("num_layers", 12)
        self.hidden_size = config.get("hidden_size", 768)
        self.num_attention_heads = config.get("num_attention_heads", 12)
        self.mlp_hidden_dim = config.get("mlp_hidden_dim", 3072)
        self.vocab_size = config.get("vocab_size", 30000)
        
    def _create_layer_node(self, x, y, layer_name, width=100, height=30, color='lightblue'):
        """Helper to create a node for a single layer component."""
        return dict(
            type="rect",
            x0=x - width/2, y0=y - height/2,
            x1=x + width/2, y1=y + height/2,
            fillcolor=color,
            line=dict(color="black", width=2),
            label=dict(text=layer_name, font=dict(size=10, color="black")),
            layer="below"
        )
    
    def visualize_full_model(self):
        """Create a visualization of the entire model architecture."""
        # Set up figure
        fig = go.Figure()
        
        # Basic layout settings
        layer_spacing = 50
        total_height = (self.num_layers + 2) * layer_spacing  # +2 for input and output layers
        
        # Add input embedding layer
        fig.add_shape(self._create_layer_node(0, 0, "Input Embedding", width=150, color='lightgreen'))
        
        # Add transformer layers
        for i in range(self.num_layers):
            y_pos = (i + 1) * layer_spacing
            fig.add_shape(self._create_layer_node(0, y_pos, f"Transformer Layer {i+1}", width=150, color='lightskyblue'))
        
        # Add output layer
        fig.add_shape(self._create_layer_node(0, total_height - layer_spacing, "Output Layer", width=150, color='lightcoral'))
        
        # Add connecting lines between layers
        for i in range(self.num_layers + 1):
            y_start = i * layer_spacing
            y_end = (i + 1) * layer_spacing
            fig.add_shape(
                type="line",
                x0=0, y0=y_start,
                x1=0, y1=y_end,
                line=dict(color="black", width=1)
            )
        
        # Configure axes and layout
        fig.update_xaxes(range=[-100, 100], showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(range=[-50, total_height], showticklabels=False, showgrid=False, zeroline=False)
        
        fig.update_layout(
            title=f"Model Architecture: {self.num_layers} Layers",
            width=500,
            height=max(600, total_height + 100),
            showlegend=False,
            plot_bgcolor='white'
        )
        
        return fig
    
    def visualize_layer_details(self, layer_index=0):
        """Create a detailed visualization of a specific layer."""
        # Set up figure
        fig = go.Figure()
        
        # Component dimensions and positions
        width = 600
        height = 400
        center_x = 0
        center_y = 0
        spacing = 80
        
        # Add components within the layer
        components = [
            {"name": "Layer Norm 1", "y_pos": -3*spacing, "color": "lightgoldenrodyellow", "width": 120},
            {"name": "Multi-Head Attention", "y_pos": -2*spacing, "color": "lightblue", "width": 180},
            {"name": "Residual Connection", "y_pos": -spacing, "color": "lightgrey", "width": 160},
            {"name": "Layer Norm 2", "y_pos": 0, "color": "lightgoldenrodyellow", "width": 120},
            {"name": f"MLP ({self.mlp_hidden_dim})", "y_pos": spacing, "color": "lightgreen", "width": 180},
            {"name": "Residual Connection", "y_pos": 2*spacing, "color": "lightgrey", "width": 160}
        ]
        
        # Add attention detail box
        attention_detail = f"{self.num_attention_heads} Heads × {self.hidden_size // self.num_attention_heads} Dim"
        
        # Add each component
        for comp in components:
            fig.add_shape(self._create_layer_node(
                center_x, comp["y_pos"], comp["name"], 
                width=comp["width"], color=comp["color"]
            ))
            
            # Add attention details below the multi-head attention component
            if comp["name"] == "Multi-Head Attention":
                fig.add_annotation(
                    x=center_x, y=comp["y_pos"] + 20,
                    text=attention_detail,
                    showarrow=False,
                    font=dict(size=10)
                )
        
        # Add connecting arrows between components
        for i in range(len(components) - 1):
            y_start = components[i]["y_pos"]
            y_end = components[i+1]["y_pos"]
            fig.add_shape(
                type="line",
                x0=center_x, y0=y_start + 15,  # Adjust to connect at bottom of shape
                x1=center_x, y1=y_end - 15,    # Adjust to connect at top of shape
                line=dict(color="black", width=1, dash="dot")
            )
        
        # Add input/output arrows
        fig.add_shape(
            type="line",
            x0=center_x, y0=-3*spacing - 50,
            x1=center_x, y1=-3*spacing - 15,
            line=dict(color="black", width=2),
            layer="below"
        )
        fig.add_shape(
            type="line",
            x0=center_x, y0=2*spacing + 15,
            x1=center_x, y1=2*spacing + 50,
            line=dict(color="black", width=2),
            layer="below"
        )
        
        # Add annotations for input/output
        fig.add_annotation(
            x=center_x, y=-3*spacing - 60,
            text="Input",
            showarrow=False
        )
        fig.add_annotation(
            x=center_x, y=2*spacing + 60,
            text="Output",
            showarrow=False
        )
        
        # Configure axes and layout
        fig.update_xaxes(range=[-300, 300], showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(range=[-4*spacing - 80, 3*spacing + 80], showticklabels=False, showgrid=False, zeroline=False)
        
        fig.update_layout(
            title=f"Transformer Layer {layer_index+1} Detail",
            width=width,
            height=height,
            showlegend=False,
            plot_bgcolor='white'
        )
        
        return fig
    
    def visualize_attention_mechanism(self):
        """Create a visualization of the multi-head attention mechanism."""
        # Set up figure
        fig = go.Figure()
        
        # Basic settings
        width = 700
        height = 500
        num_heads_to_show = min(8, self.num_attention_heads)  # Limit for visual clarity
        head_dim = self.hidden_size // self.num_attention_heads
        
        # Define positions
        input_x = -250
        qkv_x = -100
        heads_x = 100
        output_x = 250
        
        # Input box
        fig.add_shape(self._create_layer_node(
            input_x, 0, f"Input\n({self.hidden_size})", 
            width=80, height=80, color='lightgreen'
        ))
        
        # QKV projections
        qkv_y_positions = [-60, 0, 60]
        qkv_labels = ["Query", "Key", "Value"]
        qkv_colors = ["lightskyblue", "lightpink", "lightgreen"]
        
        for i, (label, y_pos, color) in enumerate(zip(qkv_labels, qkv_y_positions, qkv_colors)):
            fig.add_shape(self._create_layer_node(
                qkv_x, y_pos, f"{label}\n({self.hidden_size})",
                width=80, height=40, color=color
            ))
            # Connect from input
            fig.add_shape(
                type="line",
                x0=input_x + 40, y0=0,
                x1=qkv_x - 40, y1=y_pos,
                line=dict(color="black", width=1)
            )
        
        # Attention heads
        head_spacing = 160 / num_heads_to_show
        head_y_positions = [head_spacing * (i - num_heads_to_show/2 + 0.5) for i in range(num_heads_to_show)]
        
        for i, y_pos in enumerate(head_y_positions):
            head_label = f"Head {i+1}\n({head_dim})" if i < num_heads_to_show - 1 else f"Head {i+1}\n+{self.num_attention_heads - num_heads_to_show} more" if self.num_attention_heads > num_heads_to_show else f"Head {i+1}\n({head_dim})"
            
            fig.add_shape(self._create_layer_node(
                heads_x, y_pos, head_label,
                width=70, height=30, color='lightsalmon'
            ))
            
            # Connect from QKV
            for j, qkv_y in enumerate(qkv_y_positions):
                fig.add_shape(
                    type="line",
                    x0=qkv_x + 40, y0=qkv_y,
                    x1=heads_x - 35, y1=y_pos,
                    line=dict(color="black", width=1, dash="dot")
                )
        
        # Output projection
        fig.add_shape(self._create_layer_node(
            output_x, 0, f"Output\n({self.hidden_size})",
            width=80, height=80, color='lightcoral'
        ))
        
        # Connect from heads to output
        for y_pos in head_y_positions:
            fig.add_shape(
                type="line",
                x0=heads_x + 35, y0=y_pos,
                x1=output_x - 40, y1=0,
                line=dict(color="black", width=1)
            )
        
        # Configure axes and layout
        fig.update_xaxes(range=[-350, 350], showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(range=[-150, 150], showticklabels=False, showgrid=False, zeroline=False)
        
        fig.update_layout(
            title=f"Multi-Head Attention Mechanism ({self.num_attention_heads} heads)",
            width=width,
            height=height,
            showlegend=False,
            plot_bgcolor='white'
        )
        
        return fig

    def visualize_models_from_config(self):
        """Generate all visualizations from a config file."""
        return {
            "full_model": self.visualize_full_model(),
            "layer_detail": self.visualize_layer_details(),
            "attention": self.visualize_attention_mechanism()
        }

def create_model_visualizations(config):
    """Helper function to create model visualizations from config."""
    visualizer = ModelVisualizer(config)
    return visualizer.visualize_models_from_config()