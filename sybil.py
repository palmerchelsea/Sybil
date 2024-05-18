import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import DBSCAN, KMeans
import community as community_louvain  # Louvain community detection

# Constants
MIN_CLUSTER_SIZE = 20  # Threshold for Sybil detection
CHUNK_SIZE = 10**6  # Size of chunks to read
MAX_WORKERS = os.cpu_count()  # Use the number of CPU cores available

# Known exchange wallets and contracts
EXCHANGE_WALLETS = set([
    # Add known exchange wallet addresses here
    "0xExchangeWallet1",
    "0xExchangeWallet2",
    # ...
])

# Step 1: Load and Clean Data in Chunks from Multiple Files
def load_and_clean_data(files):
    print("Starting data loading and cleaning...")
    required_columns = ['SENDER_WALLET', 'DESTINATION_CONTRACT', 'SOURCE_TIMESTAMP_UTC', 'NATIVE_DROP_USD']
    
    chunk_list = []
    
    for file_path in files:
        print(f"Processing file: {file_path}")
        for chunk in tqdm(pd.read_csv(file_path, compression='gzip', chunksize=CHUNK_SIZE), desc=f"Loading data from {file_path}"):
            chunk.dropna(subset=required_columns, inplace=True)
            chunk.drop_duplicates(inplace=True)

            # Ensure correct data types
            chunk['SOURCE_TIMESTAMP_UTC'] = pd.to_datetime(chunk['SOURCE_TIMESTAMP_UTC'], errors='coerce')
            chunk['NATIVE_DROP_USD'] = pd.to_numeric(chunk['NATIVE_DROP_USD'], errors='coerce')
            chunk.dropna(subset=['SOURCE_TIMESTAMP_UTC', 'NATIVE_DROP_USD'], inplace=True)
            
            chunk_list.append(chunk)
    
    df = pd.concat(chunk_list, axis=0)
    print("Data loading and cleaning completed.")
    return df

# Step 2: Construct a Comprehensive Transaction Graph in Parallel
def add_edges_to_graph(chunk):
    G = nx.Graph()
    for _, row in chunk.iterrows():
        G.add_edge(row['SENDER_WALLET'], row['DESTINATION_CONTRACT'], weight=row['NATIVE_DROP_USD'])
    return G

def construct_transaction_graph(df):
    print("Starting transaction graph construction...")
    chunks = [df[i:i + CHUNK_SIZE] for i in range(0, df.shape[0], CHUNK_SIZE)]
    G = nx.Graph()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(add_edges_to_graph, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Constructing graph"):
            G = nx.compose(G, future.result())

    print("Transaction graph construction completed.")
    return G

# Step 3: Detect Clusters using Louvain Community Detection
def detect_clusters_louvain(G):
    print("Starting cluster detection using Louvain method...")
    partition = community_louvain.best_partition(G)
    clusters = {c: [] for c in set(partition.values())}
    for node, com in partition.items():
        clusters[com].append(node)
    print("Cluster detection completed.")
    return [cluster for cluster in clusters.values() if len(cluster) >= MIN_CLUSTER_SIZE]

# Generate Visual Graph
def visualize_clusters(G, clusters, output_dir):
    print("Generating visual graphs for clusters...")
    for i, cluster in enumerate(clusters):
        subgraph = G.subgraph(cluster)
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, node_size=50, node_color='skyblue', edge_color='gray')
        plt.title(f'Cluster {i} Visualization')
        plt.savefig(os.path.join(output_dir, f'cluster_{i}_graph.png'))
        plt.close()
    print("Visual graph generation completed.")

# Step 4: Sybil Wallet Identification and Reporting
def identify_sybil_clusters(df):
    G = construct_transaction_graph(df)
    clusters = detect_clusters_louvain(G)
    return G, clusters

def generate_report(sybil_clusters, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G, clusters = sybil_clusters
    print("Generating cluster reports...")
    for i, cluster in enumerate(clusters):
        report_file = os.path.join(output_dir, f'cluster_{i}_report.csv')
        df_cluster = pd.DataFrame({'sybil_wallet': cluster, 'cluster_id': i})
        df_cluster.to_csv(report_file, index=False)
        print(f"Report generated: {report_file}")

    visualize_clusters(G, clusters, output_dir)

# Main Execution Flow
def run_sybil_detection(data_dir, output_dir):
    start_time = time.time()
    
    print(f"Loading files from directory: {data_dir}")
    files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv.gz')]
    df = load_and_clean_data(files)

    print("Constructing transaction graph and detecting clusters...")
    sybil_clusters = identify_sybil_clusters(df)

    print(f"Total Sybil clusters detected: {len(sybil_clusters[1])}")

    if sybil_clusters[1]:
        generate_report(sybil_clusters, output_dir)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    data_dir = r"C:\Users\User\Downloads\L0\snapshot1_transactions\transactions"
    output_dir = r"C:\Users\User\Downloads\L0\reports"
    run_sybil_detection(data_dir, output_dir)
