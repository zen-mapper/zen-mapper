"""
Accessing Clusterer Metadata in Zen Mapper
--------------------------------------------

This example demonstrates how to access and use metadata passed on from
clusterers. As our clusterer we will use the Affinity Propagation from scikit-learn, which provides metadata which we will use to generate an animation of mapper graphs.
"""
# %%
# Setting up the environment
# =========================
#
# Import libraries and generate a simple circular dataset..

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import imageio
import os
from matplotlib.patches import Rectangle
from sklearn.cluster import AffinityPropagation, affinity_propagation
from zen_mapper.cover import Width_Balanced_Cover
from zen_mapper.cluster import sk_learn
from zen_mapper import mapper
from zen_mapper.adapters import to_networkx

theta = np.linspace(0, 2 * np.pi, 100)
# Swap sin and cos for better indexing (makes affinity matrix look nice).
data = np.c_[np.sin(theta), np.cos(theta)]

# Use x-coordinate as the lens function
projection = data[:, 0]

# %%
# Configuring mapper and accessing clusterer metadata
# ==================================================
#
# The `sk_learn` wrapper in zen-mapper preserves all attributes from the sklearn
# clusterer, making them accessible after the mapper computation. For Affinity 
# Propagation, this includes `affinity_matrix_`, `cluster_centers_`, and `n_iter_`.

# Cover Scheme Parameters
n_elements = 4
percent_overlap = 0.2
cover_scheme = Width_Balanced_Cover(n_elements=n_elements, percent_overlap=percent_overlap)

# Create temp directory for storing animation frames
os.makedirs('frames', exist_ok=True)

# Define the range of preference values to explore
preference_values = np.arange(-60, 1, 0.25)
filenames = []

# Data boundaries plotting
x_min, x_max = data[:, 0].min(), data[:, 0].max()
y_min, y_max = data[:, 1].min(), data[:, 1].max()
margin = 0.1
x_min -= margin; x_max += margin
y_min -= margin; y_max += margin

# Maximum iterations for Affinity Propagation
max_iter = 500

# Calculate intervals for cover visualization
proj_min = np.min(projection)
proj_max = np.max(projection)
interval_width = (proj_max - proj_min) / (n_elements - (n_elements - 1) * percent_overlap)
intervals = []

for i in range(n_elements):
    start = proj_min + i * interval_width * (1 - percent_overlap)
    end = start + interval_width
    intervals.append((start, end))

# %%
# Visualizing mapper graphs with clusterer metadata
# ================================================
#
# For each preference value, we create a mapper graph using Affinity Propagation.
# We can access the clusterer's metadata through `result.cluster_metadata`. 
# This allows us to visualize the affinity matrices and set node positions for the graph.

# Generate frames for each preference value
for i, pref in enumerate(preference_values):
    try:
        # Affinity Propagation clusterer with current preference value
        sk = AffinityPropagation(preference=pref, damping=0.7, max_iter=max_iter, convergence_iter=100)

        # Wrap the sklearn clusterer - this preserves all metadata
        clusterer = sk_learn(base_clusterer=sk)

        # Compute mapper graph
        result = mapper(
            data=data,
            projection=projection,
            cover_scheme=cover_scheme,
            clusterer=clusterer,
            dim=1,
        )

        # Create figure
        fig = plt.figure(figsize=(12, 10))

        # Main axis for data and mapper graph
        main_ax = plt.subplot2grid((5, 5), (0, 0), colspan=5, rowspan=3)

        # Calculate positions for nodes based on cluster centers
        # Note how we access the cluster_centers_ attribute from the clusterer metadata
        pos = dict()
        for cover_element, cluster_meta in zip(result.cover, result.cluster_metadata):
            for node, center in zip(cover_element, cluster_meta.cluster_centers_):
                pos[node] = center

        # Current mapper graph
        G = to_networkx(result.nerve)

        # Plot original data
        main_ax.scatter(data[:, 0], data[:, 1], color='lightblue', s=30)

        # Draw mapper graph on top using positions obtained from metadata
        node_size = 200
        nx.draw_networkx_nodes(G, pos, node_color='blue', alpha=0.5, node_size=node_size, ax=main_ax)
        nx.draw_networkx_edges(G, pos, width=3, alpha=0.6, ax=main_ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=main_ax)

        # Dimensions for cover in visualization
        y_height = y_max - y_min
        proj_range = proj_max - proj_min

        # Create axes for the affinity matrices from each cover element
        affinity_axes = []
        for j, interval in enumerate(intervals):
            x_min_rect = interval[0]
            x_max_rect = interval[1]

            # Also, draw rectangular cover elements
            rectangle = Rectangle((x_min_rect, y_min), 
                                x_max_rect - x_min_rect, y_height,
                                facecolor=f'C{j}', alpha=0.15,
                                edgecolor='black', linestyle='--', linewidth=1)
            main_ax.add_patch(rectangle)

            # Dimensions for affinity matrix subplot
            matrix_width = (x_max_rect - x_min_rect) / proj_range * 5
            ax_width = max(0.5, min(matrix_width, 1.0))
            ax_left = (x_min_rect - proj_min) / proj_range * 5

            # Create subplot for this covering element's affinity matrix
            ax = fig.add_axes([ax_left/5, 0.15, ax_width/5, 0.15])
            affinity_axes.append(ax)

            # Display the affinity matrix from the clusterer metadata
            if j < len(result.cluster_metadata):
                cluster_meta = result.cluster_metadata[j]
                if hasattr(cluster_meta, 'affinity_matrix_'):
                    im = ax.imshow(cluster_meta.affinity_matrix_, cmap='viridis', aspect='auto')
                    ax.set_title(f"Cover {j+1}", fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.text(0.5, 0.5, "No matrix", ha='center', va='center', fontsize=8)

        main_ax.set_xlim(x_min, x_max)
        main_ax.set_ylim(y_min, y_max)
        main_ax.set_aspect('equal')
        main_ax.set_title("Affinity Propagation \n Mapper Graph")


        # Display Parameter Info:
        # Use the metadata to count clusters and calculate average iterations
        n_clusters = sum(1 for cm in result.cluster_metadata for _ in cm.cluster_centers_)
        avg_iterations = np.mean([cm.n_iter_ for cm in result.cluster_metadata]) if result.cluster_metadata else 0
        plt.figtext(0.5, 0.95, f"Preference = {pref:.1f}  |  Clusters: {n_clusters}  |  Avg. Iterations: {avg_iterations:.1f}", 
                  fontsize=14, ha='center')

        # Add colorbar for matrices
        if len(affinity_axes) > 0 and 'im' in locals():
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.15])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label('Affinity', fontsize=8)

        # Save the figure as a frame
        filename = f'frames/frame_{i:03d}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        filenames.append(filename)
        plt.close()

    except Exception as e:
        print(f"Error at preference = {pref:.1f}: {e}")
        continue

# %%
# Creating the animation
# ====================================================
#
# Finally, we combine all the generated frames into a GIF animation which
# shows how the mapper graph and clustering metadata evolve as the preference
# parameter in AffinityPropagation changes.
gif_filename= 'affinity_propagation_preference_mapper_evolution.gif'
if filenames:
    with imageio.get_writer(gif_filename, mode='I', duration=0.75) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"GIF created as {gif_filename}")
else:
    print("No frames created. Cannot build GIF.")

# %%
# Display Animation
from IPython.display import Image, display
display(Image(filename=gif_filename))
# %%
