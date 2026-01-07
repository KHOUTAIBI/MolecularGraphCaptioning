import os
import pickle


def inspect_graph_file(pkl_path, num_samples=3):
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return
    
    print(f"\nLoading: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    
    print(f"Total graphs: {len(data_list)}")
    
    if len(data_list) == 0:
        print("No graphs in this file")
        return
    
    first_graph = data_list[0]
    print(f"\nGraph type: {type(first_graph).__name__}")
    
    # Print all available attributes/features in the graph
    print(f"\nAll graph attributes:")
    if hasattr(first_graph, 'keys'):
        # PyTorch Geometric Data objects have a keys() method
        print(f"  {list(first_graph.keys())}")
    else:
        # Fallback to inspecting __dict__
        print(f"  {list(vars(first_graph).keys())}")
    
    if hasattr(first_graph, 'x'):
        print(f"Node features: {first_graph.x.shape}")
    if hasattr(first_graph, 'edge_index'):
        print(f"Edge indices: {first_graph.edge_index.shape}")
    if hasattr(first_graph, 'edge_attr'):
        print(f"Edge features: {first_graph.edge_attr.shape}")
    
    print(f"\nSample graph:")
    for i in range(min(num_samples, len(data_list))):
        graph = data_list[i]
        print(f"  Graph {i+1}:")
        print(f"    ID: {getattr(graph, 'id', 'N/A')}")
        print(f"    Nodes: {graph.x.shape[0] if hasattr(graph, 'x') else 'N/A'}")
        print(f"    Edges: {graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 'N/A'}")
        
        if hasattr(graph, 'x'):
            print(f"    Node features (first 3 nodes):")
            print(graph.x[:3])
        
        if hasattr(graph, 'edge_index'):
            print(f"    Edge indices (first 5 edges):")
            print(graph.edge_index[:, :5])
        
        if hasattr(graph, 'edge_attr'):
            print(f"    Edge attributes (first 5 edges):")
            print(graph.edge_attr[:5])
        
        if hasattr(graph, 'description'):
            desc = graph.description[:200] if len(graph.description) > 200 else graph.description
            print(f"    Description: {desc}...")
    
    if hasattr(data_list[0], 'x'):
        num_nodes = [g.x.shape[0] for g in data_list]
        print(f"\nNodes: min={min(num_nodes)}, max={max(num_nodes)}, avg={sum(num_nodes)/len(num_nodes):.1f}")
    
    if hasattr(data_list[0], 'edge_index'):
        num_edges = [g.edge_index.shape[1] for g in data_list]
        print(f"Edges: min={min(num_edges)}, max={max(num_edges)}, avg={sum(num_edges)/len(num_edges):.1f}")
    
    has_desc = sum(1 for g in data_list if hasattr(g, 'description') and g.description)
    print(f"Graphs with descriptions: {has_desc}/{len(data_list)}")


def main():
    base_path = "data"
    splits = ["train", "validation", "test"]
    
    print("=" * 100)
    print("INSPECTING GRAPH DATA FILES")
    print("=" * 100)
    
    for i, split in enumerate(splits):
        if i > 0:
            print("\n" + "=" * 100)
        pkl_path = f"{base_path}/{split}_graphs.pkl"
        inspect_graph_file(pkl_path, num_samples=1)
    
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()

