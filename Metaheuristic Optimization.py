import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph to represent the optimization flowchart
G = nx.DiGraph()

# Add nodes representing different parameters and stages of the optimization process
G.add_node("Initialize Parameters")
G.add_node("Evaluate Objective Function")
G.add_node("Update Parameters")
G.add_node("Check Convergence")
G.add_node("Refine Parameters")
G.add_node("Final Solution")

# Add edges to show the flow between different stages
G.add_edges_from([
    ("Initialize Parameters", "Evaluate Objective Function"),
    ("Evaluate Objective Function", "Update Parameters"),
    ("Update Parameters", "Check Convergence"),
    ("Check Convergence", "Refine Parameters"),
    ("Refine Parameters", "Evaluate Objective Function"),
    ("Check Convergence", "Final Solution")
])

# Set up the layout for the nodes
# Use spring layout for a dynamic appearance
pos = nx.spring_layout(G, seed=42)

# Draw the nodes, edges, and labels
plt.figure(figsize=(10, 8))

# Draw the nodes with a gear-like appearance
nx.draw_networkx_nodes(G, pos, node_size=3000,
    node_color='lightblue',node_shape='o', edgecolors='black')

# Draw the edges as arrows to indicate direction of flow
nx.draw_networkx_edges(G, pos, arrowstyle='-|>',
    arrowsize=20, edge_color='black')

# Add labels to the nodes
nx.draw_networkx_labels(G, pos, font_size=12,
                        font_color='black', font_weight='bold')

# Add a title to the plot
plt.title("Metaheuristic Optimization Flowchart", fontsize=16)

# Remove the axes for a cleaner look
plt.axis('off')

# Show the plot
plt.show()
