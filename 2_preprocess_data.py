import os
import pandas as pd
from scapy.all import rdpcap
from tqdm import tqdm  

# Step 1: Parse PCAP files and extract features
def parse_pcap(file_path, label):
    print(f"\nReading PCAP file: {file_path}")
    packets = rdpcap(file_path)
    print(f"Total packets in {os.path.basename(file_path)}: {len(packets)}")
    
    data = []
    for packet in tqdm(packets, desc=f"Processing {os.path.basename(file_path)}", unit="packets"):
        features = {
            'src_ip': packet['IP'].src if 'IP' in packet else '0.0.0.0',
            'dst_ip': packet['IP'].dst if 'IP' in packet else '0.0.0.0',
            'packet_size': len(packet),
            'timestamp': packet.time,
            'protocol': packet['IP'].proto if 'IP' in packet else 0,
            'label': label
        }
        data.append(features)
    print(f"Finished extracting features from {os.path.basename(file_path)}.")
    return pd.DataFrame(data)

# Step 2: Load and label the dataset
def load_dataset(base_path):
    print(f"\nLoading dataset from directory: {base_path}")
    datasets = []
    pcap_files = [f for f in os.listdir(base_path) if f.endswith('.pcap')]
    print(f"Found {len(pcap_files)} PCAP files in the directory.")
    
    for file_name in tqdm(pcap_files, desc="Processing PCAP files", unit="file"):
        file_path = os.path.join(base_path, file_name)
        label = 1 if 'injected' in file_name else 0  # 1 = attack, 0 = normal
        print(f"\nLabeling file '{file_name}' with label: {label}")
        df = parse_pcap(file_path, label)
        print(f"Shape of data extracted from '{file_name}': {df.shape}")
        datasets.append(df)
    combined_data = pd.concat(datasets, ignore_index=True)
    print(f"\nCombined dataset shape: {combined_data.shape}")
    return combined_data

# Step 3: Preprocess the data
def preprocess_data(df):
    print("\nPreprocessing the data...")
    print("Original dataset head:")
    print(df.head())

    # Convert categorical features to numerical
    print("\nConverting IP addresses to numerical values...")
    df['src_ip'] = df['src_ip'].apply(lambda x: int(x.replace('.', '')))
    df['dst_ip'] = df['dst_ip'].apply(lambda x: int(x.replace('.', '')))
    
    print("Dataset after IP address conversion:")
    print(df.head())

    # Normalize numerical features
    print("\nNormalizing numerical features...")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numerical_features = ['src_ip', 'dst_ip', 'packet_size', 'timestamp', 'protocol']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    print("Dataset after normalization:")
    print(df.head())

    return df

# Main function
def main():
    # Load dataset
    base_path = 'C:/Users/pmadh/Desktop/Major Project Execution/IDS Complete Execution(AEID)/AEID pcap files'  # Replace with the path to your AEID dataset
    print(f"\nStarting the script. Base path: {base_path}")
    df = load_dataset(base_path)

    # Preprocess data
    print("\nStarting data preprocessing...")
    df = preprocess_data(df)

    # Save preprocessed data to CSV
    output_file = 'preprocessed2_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nPreprocessed data successfully saved to '{output_file}'.")

if __name__ == '__main__':
    main()
