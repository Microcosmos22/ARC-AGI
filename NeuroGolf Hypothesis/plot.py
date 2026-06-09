import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_arc_task(train_pairs, test_input=None, hypothesis_name="Unknown"):
    """
    Plots the training input/output pairs and the test input side-by-side
    using the official ARC color scheme.
    """
    # Official ARC Color Palette Mapping (0-9)
    arc_colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]
    cmap = ListedColormap(arc_colors)

    num_train = len(train_pairs)
    # Total subplots = (num_train * 2 columns for Train Input/Output) + 1 column for Test Input
    fig, axes = plt.subplots(num_train + 1, 2, figsize=(8, 3 * (num_train + 1)))
    fig.suptitle(f"ARC Visualization - Found Hypothesis: {hypothesis_name}", fontsize=14, fontweight='bold')

    # 1. Plot Training Pairs
    for i, pair in enumerate(train_pairs):
        inp_grid = pair['input']
        out_grid = pair['output']

        # Train Input
        ax_inp = axes[i, 0]
        ax_inp.imshow(inp_grid, cmap=cmap, vmin=0, vmax=9)
        ax_inp.set_title(f"Train Input {i+1}")
        ax_inp.axis('off')

        # Train Output
        ax_out = axes[i, 1]
        ax_out.imshow(out_grid, cmap=cmap, vmin=0, vmax=9)
        ax_out.set_title(f"Train Output {i+1}")
        ax_out.axis('off')

    if test_input is not None:
        # 2. Plot Test Input
        ax_test = axes[num_train, 0]
        ax_test.imshow(test_input, cmap=cmap, vmin=0, vmax=9)
        ax_test.set_title("Test Input (Predict This)")
        ax_test.axis('off')

    # Empty placeholder for the missing Test Output column
    axes[num_train, 1].axis('off')
    axes[num_train, 1].text(0.5, 0.5, '?', fontsize=30, ha='center', va='center', color='gray')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # --- INTEGRATION EXAMPLE ---
    # Copy-paste your data variables right before calling it:
    train_data = [
        {'input': [[7, 9], [4, 3]], 'output': [[7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3], [9, 7, 9, 7, 9, 7], [3, 4, 3, 4, 3, 4], [7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3]]},
        {'input': [[8, 6], [6, 4]], 'output': [[8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4], [6, 8, 6, 8, 6, 8], [4, 6, 4, 6, 4, 6], [8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4]]}
    ]
    test_in = [[0, 2], [1, 0]]

    # Trigger the pop-up window
    plot_arc_task(train_data, test_in, hypothesis_name="rotate270")
