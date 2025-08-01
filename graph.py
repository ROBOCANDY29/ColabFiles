import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from google.colab import files
import ast
import time
from tensorflow.keras.models import load_model

# Load the model
global model


def load_model_from_file():
    global model
    model = load_model("model.keras")
    # Display the model's summary to confirm it's loaded correctly
    model.summary()
    return model


def pad_grid(grid, target_shape):
    h, w = grid.shape
    th, tw = target_shape
    padded = np.zeros((th, tw), dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded


def prepare_single_set_for_prediction(iop1_in, iop1_out, iop2_in, iop2_out):
    # Use fixed target shape matching model input (here 30, 30)
    # Change 30 to your model's expected H, W if different
    global model
    target_shape = (30, 30)
    p_iop1_in = pad_grid(iop1_in, target_shape)
    p_iop1_out = pad_grid(iop1_out, target_shape)
    p_iop2_in = pad_grid(iop2_in, target_shape)
    p_iop2_out = pad_grid(iop2_out, target_shape)
    X = np.stack([p_iop1_in, p_iop1_out, p_iop2_in, p_iop2_out], axis=-1)
    X = np.expand_dims(X, axis=0)
    prediction = model.predict(X)[0][0]
    return prediction


class GraphNode:
    def __init__(self, input_matrix, output_matrix, functions):
        self.input_matrix = input_matrix
        self.output_matrix = output_matrix
        self.functions = functions  # list of functions or their names


def create_graph():
    uploaded = files.upload()
    for filename in uploaded.keys():
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            G = nx.Graph()
            nodes = []  # your nodes
            for id, a in df.iterrows():
                node = GraphNode(np.array(ast.literal_eval(a[0])), np.array(
                    ast.literal_eval(a[1])), np.array(ast.literal_eval(a[2])))
            nodes.append(node)
            # print(node.input_matrix,node.output_matrix)
            # break

            for idx, nodeA in enumerate(nodes):
                for jdx, nodeB in enumerate(nodes):
                    plt.figure(figsize=(4, 2))
                    plt.text(0.5, 0.5, f"{idx} with {jdx}",
                             fontsize=10, ha='center', va='center')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                    if jdx > idx:
                        sim = prepare_single_set_for_prediction(
                            nodeA.input_matrix, nodeA.output_matrix,
                            nodeB.input_matrix, nodeB.output_matrix
                        )
                        if sim > 0.5:
                            G.add_edge(idx, jdx, weight=sim,
                                       sim_score=sim)  # Store score

            # Export to GraphML after construction:
            nx.write_graphml(G, "similarity_graph.graphml")
            files.download("similarity_graph.graphml")
            return G
        else:
            print(f"File {filename} is not a CSV file. Skipping.")
            create_graph()


def load_graph():
    uploaded = files.upload()
    for filename in uploaded.keys():
        if filename.endswith('.graphml'):
            G = nx.read_graphml(filename)
            print(f"Graph loaded from {filename}")
            return G
        else:
            print(f"File {filename} is not a GraphML file. Skipping.")
            load_graph()
            return None
