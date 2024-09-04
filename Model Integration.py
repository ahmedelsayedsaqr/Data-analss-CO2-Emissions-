import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Creating the figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Initialize the neural network graph
G_nn = nx.DiGraph()

# Add nodes for layers
input_color = "skyblue"
hidden_color = "lightgreen"
output_color = "salmon"

G_nn.add_nodes_from([
    ("Input 1", {"color": input_color}),
    ("Input 2", {"color": input_color}),
    ("Hidden 1", {"color": hidden_color}),
    ("Hidden 2", {"color": hidden_color}),
    ("Output 1", {"color": output_color})
])

# Position nodes in layers
pos_nn = {
    "Input 1": (-1, 0.5),
    "Input 2": (-1, -0.5),
    "Hidden 1": (0, 0.5),
    "Hidden 2": (0, -0.5),
    "Output 1": (1, 0)
}

# Draw the initial network without edges
colors = [G_nn.nodes[n]["color"] for n in G_nn.nodes()]
nx.draw(G_nn, pos_nn, ax=ax, with_labels=True, node_color=colors, node_size=3000,
        font_size=10, font_weight='bold')

ax.set_title("Animating Neural Network Connections")

# Edges to be added sequentially
edges = [
    ("Input 1", "Hidden 1"),
    ("Input 1", "Hidden 2"),
    ("Input 2", "Hidden 1"),
    ("Input 2", "Hidden 2"),
    ("Hidden 1", "Output 1"),
    ("Hidden 2", "Output 1")
]


def update(num):
    if num < len(edges):
        edge = edges[num]
        G_nn.add_edge(*edge)
        ax.clear()  # Clear the axis to redraw
        nx.draw(G_nn, pos_nn, ax=ax, with_labels=True, node_color=colors, node_size=3000,
                font_size=10, font_weight='bold', edge_color="gray")
        ax.set_title("Neural Network Schematic")


ani = FuncAnimation(fig, update, frames=len(
    edges)+1, interval=1000, repeat=False)

# To display in a Jupyter Notebook
HTML(ani.to_jshtml())

# To save as a GIF or video file
ani.save('neural_network_animation.gif', writer='imagemagick')