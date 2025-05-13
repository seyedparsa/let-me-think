import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

positions = {}
depth = 3
short = 3
long = 5
dead = 3

dark = False

petal_size = short + long + dead
petal_edges = []
petal_pos = {}
for i in range(short - 1):
    petal_edges.append((i, i + 1))
for i in range(short):
    petal_pos[i] = (i, 0)

radius = short/2
for i in range(long - 1):
    petal_edges.append((short + i, (short + i + 1) % (short + long - 1)))
    angle = np.pi * (i + 1) / long
    x = short/2 + radius * np.cos(angle)
    y = radius * np.sin(angle)
    petal_pos[short + i] = (x, y)


for i in range(dead):
    petal_edges.append((short + long + i - 1, (short + long + i) % (short + long + dead - 1)))
    petal_pos[short + long + i - 1] = (0, (dead-i)*-0.75)

if short > 0:
    petal_edges.append((short - 1, petal_size - 1))
if long > 0:
    petal_edges.append((short, petal_size - 1))
petal_pos[petal_size - 1] = (short, 0)
    

print(petal_edges)
print(petal_pos)

graph = nx.Graph()
for d in range(depth):
    for i in range(petal_size):
        if d > 0 and i == 0:
            continue
        node = f"{d}_{i}"
        x, y = petal_pos[i]
        positions[node] = (x + 3*d, y)
        graph.add_node(node)
    for u, v in petal_edges:
        du, uu = d, u
        dv, vv = d, v
        if d > 0 and u == 0:
            du = d - 1
            uu = petal_size - 1
        if d > 0 and v == 0:
            dv = d - 1
            vv = petal_size - 1        
        graph.add_edge(f"{du}_{uu}", f"{dv}_{vv}")

print(graph.nodes())
light_red = "#ff9999"
light_green = "#99ff99"
light_blue = "skyblue"
dark_blue = "#4682B4"
dark_green = "#8FBC8F"
node_colors = {node : dark_blue if dark else light_blue for node in graph.nodes() }
node_colors["0_0"] = light_red
node_colors[f"{depth-1}_{petal_size-1}"] = light_green
if dark:
    node_colors["0_1"] = light_blue
    node_colors["0_6"] = light_blue
    node_colors["0_9"] = light_blue
    node_colors[f"{depth-1}_{petal_size-1}"] = dark_green


positions_shifted = {node: (x + 1.25, y + 1) for node, (x, y) in positions.items()}
# Adjust colors to make them lighter for a background effect
lighter_node_colors = {node: "#d3d3d3" for node in graph.nodes()}  # Light gray for all nodes
lighter_node_colors[f"{depth-1}_{petal_size-1}"] = light_green
lighter_edge_color = "#cccccc"  # Light gray for edges

# Generate a random permutation of numbers from 1 to 62
num_nodes = len(graph.nodes())
# random_labels = random.sample(range(1, 2*num_nodes + 1), num_nodes)
random_labels = [15,2,57,23,7,3,38,55,58,4,27,37,44,46,41,12,33,43,31,19,29,54,18,11,32,16,30,26,17,6,1,5,60,49,24,34,47,9,42,61,48,14,56,51,52,35,59,25,20,36,45,0,22,10,21,40,39,13,28,50,53,8]
random_labels = random_labels[:num_nodes]

# Create a mapping from old labels to new random labels
node_mapping = {old_label: new_label for old_label, new_label in zip(graph.nodes(), random_labels)}

# Relabel the nodes in the graph
graph = nx.relabel_nodes(graph, node_mapping)

# Update positions and colors to reflect new labels
positions = {node_mapping[node]: pos for node, pos in positions.items()}
positions_shifted = {node_mapping[node]: pos for node, pos in positions_shifted.items()}
node_colors = {node_mapping[node]: color for node, color in node_colors.items()}
lighter_node_colors = {node_mapping[node]: color for node, color in lighter_node_colors.items()}

# plt.figure(figsize=(8, 6))
print(graph)
nx.draw(
    graph,
    pos=positions_shifted,
    with_labels=False,
    node_size=200,
    node_color=lighter_node_colors.values(),
    edge_color=lighter_edge_color,
    linewidths=0.5,  # Thinner edges for a lighter appearance
)

# Shift positions slightly for the second graph

nx.draw(
    graph,
    pos=positions,
    with_labels=True,
    node_size=300,
    node_color=node_colors.values(),
    linewidths=0.5,  # Thinner edges for a lighter appearance
    edgecolors="black",
    font_size=8  # Reduce font size for labels
)
plt.axis("equal")
plt.margins(0, 0)  # Remove side margins, keep small top and bottom margins
plt.show()
# plt.title(f"flower(d={depth}, s={short}, l={long}, b={dead}) graph")
save_path = f"figures/flower_d{depth}-s{short}-l{long}-b{dead}.png"
plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Save with small top and bottom padding