import numpy as np
import ml_dtypes  # noqa: F401
from safetensors import safe_open
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


def format_num(num: int) -> str:
    """Formats a number with commas and M/K suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


def create_bar(fraction: float, width: int = 20) -> str:
    """Creates a simple ASCII/Unicode bar."""
    filled = int(fraction * width)
    return "█" * filled + "░" * (width - filled)


def visualize_weights(path: str):
    console = Console()

    try:
        f = safe_open(path, framework="np", device="cpu")
    except Exception as e:
        console.print(f"[bold red]Error opening weights file:[/bold red] {e}")
        return

    params_by_layer = {}
    total_params = 0

    # First pass: calculate counts
    for key in f.keys():
        shape = f.get_tensor(key).shape
        count = np.prod(shape)
        total_params += count

        if key.startswith("model.layers_stacked."):
            # These are stacked across all layers
            num_layers = shape[0]
            count_per_layer = count // num_layers

            for i in range(num_layers):
                layer_key = f"Layer {i:02d}"
                if layer_key not in params_by_layer:
                    params_by_layer[layer_key] = 0
                params_by_layer[layer_key] += count_per_layer
        else:
            # Global parameters like embedding or final norm
            layer_key = key.replace("model.", "").replace(".weight", "").capitalize()
            if layer_key not in params_by_layer:
                params_by_layer[layer_key] = 0
            params_by_layer[layer_key] += count

    # Sort layers for display
    sorted_layers = sorted(params_by_layer.items())

    # Create Summary Table
    table = Table(
        title=f"Parameter Distribution: {path}", box=box.ROUNDED, show_footer=True
    )
    table.add_column("Component", footer="Total")
    table.add_column("Parameters", justify="right", footer=format_num(total_params))
    table.add_column("Percentage", justify="right")
    table.add_column("Visual", justify="left")

    for layer, count in sorted_layers:
        percentage = (count / total_params) * 100
        table.add_row(
            layer,
            format_num(count),
            f"{percentage:.2f}%",
            create_bar(count / max(params_by_layer.values())),
        )

    console.print(Panel(table, expand=False))

    # Detailed Breakdown for one Transformer Block (since they are identical in this stacked setup)
    block_params = {}
    for key in f.keys():
        if key.startswith("model.layers_stacked."):
            shape = f.get_tensor(key).shape
            count_per_layer = np.prod(shape) // shape[0]
            name = key.replace("model.layers_stacked.", "").replace(".weight", "")
            block_params[name] = count_per_layer

    if block_params:
        block_total = sum(block_params.values())
        detail_table = Table(
            title="Transformer Block Internal Breakdown (Per Layer)", box=box.SIMPLE
        )
        detail_table.add_column("Sub-component")
        detail_table.add_column("Parameters", justify="right")
        detail_table.add_column("Percentage", justify="right")

        for name, count in sorted(
            block_params.items(), key=lambda x: x[1], reverse=True
        ):
            detail_table.add_row(
                name, format_num(count), f"{(count / block_total) * 100:.1f}%"
            )

        console.print(Panel(detail_table, title="Layer Internal Details", expand=False))

    console.print(
        f"\n[bold green]Total Model Parameters:[/bold green] {format_num(total_params)}"
    )


if __name__ == "__main__":
    import sys

    weight_path = "model_stacked_it.safetensors"
    if len(sys.argv) > 1:
        weight_path = sys.argv[1]
    visualize_weights(weight_path)
