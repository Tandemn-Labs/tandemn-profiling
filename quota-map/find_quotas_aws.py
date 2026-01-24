#!/usr/bin/env python3
import pandas as pd
from sky.adaptors import aws as aws_adaptor
from concurrent.futures import ThreadPoolExecutor, as_completed


# Quota codes are at the FAMILY level, not instance level
# G family = Graphics instances, P family = GPU Compute instances
FAMILY_QUOTA_CODES = {
    'G': {'on-demand': 'L-DB2E81BA', 'spot': 'L-3819A6DF'},
    'P4_P3_P2': {'on-demand': 'L-417A185B', 'spot': 'L-7212CCBC'},
    'P5_P6': {'on-demand': 'L-417A185B', 'spot': 'L-C4BD4855'},
}

ALL_AWS_REGIONS = [
    'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
    'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1', 'eu-north-1', 
    'ap-south-1', 'ap-northeast-1', 'ap-northeast-2', 'ap-northeast-3',
    'ap-southeast-1', 'ap-southeast-2',
    'ca-central-1', 'sa-east-1', 
]


ALL_FAMILY_TYPES = ['G', 'P4_P3_P2', 'P5_P6']


def load_resource_table(csv_path):
    """Load resource table and filter for GPU family types only"""
    df = pd.read_csv(csv_path)
    gpu_families = df[df['Family_type'].isin(ALL_FAMILY_TYPES)]
    return gpu_families


def get_quota(quota_code, region):
    """Get quota value for a quota code in a region"""
    client = aws_adaptor.client('service-quotas', region_name=region)
    response = client.get_service_quota(
        ServiceCode='ec2',
        QuotaCode=quota_code
    )
    return int(response['Quota']['Value'])


def fetch_family_region_quota(family_type, region):
    """Fetch on-demand and spot quotas for a family in a region"""
    on_demand_code = FAMILY_QUOTA_CODES[family_type]['on-demand']
    spot_code = FAMILY_QUOTA_CODES[family_type]['spot']
    
    on_demand = get_quota(on_demand_code, region)
    
    # If spot code is N/A, return 0
    if spot_code == 'N/A':
        spot = 0
    else:
        spot = get_quota(spot_code, region)
    
    print(f"  {family_type} in {region}: on-demand={on_demand}, spot={spot}")
    
    return (family_type, region, on_demand, spot)


def fetch_all_quotas():
    """Fetch quotas for all family types across all regions (parallelized)"""
    print("Fetching quotas for all GPU families across all regions...")
    
    # Build all (family, region) combinations to check
    tasks = [(family, region) for family in ALL_FAMILY_TYPES for region in ALL_AWS_REGIONS]
    
    # Result structure: quotas[family_type][region] = {'on_demand': X, 'spot': Y}
    quotas = {family: {} for family in ALL_FAMILY_TYPES}
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_family_region_quota, f, r) for f, r in tasks]
        
        for future in as_completed(futures):
            family_type, region, on_demand, spot = future.result()
            quotas[family_type][region] = {'on_demand': on_demand, 'spot': spot}
    
    return quotas


def create_quota_map_by_region(resource_table_path, output_path):
    """Create quota map with on-demand and spot allocations per region"""
    
    quotas = fetch_all_quotas()
    print("\nLoading resource table...")
    resource_df = load_resource_table(resource_table_path)
    print(f"Found {len(resource_df)} GPU instance types (G and P families)")
    
    results = []
    
    for idx, row in resource_df.iterrows():
        family_type = row['Family_type']
        
        row_data = {
            'Family': row['Family'],
            'Instance_Type': row['GPU instance type'],
            'vCPU': row['vCPU req'],
            'GPU_Type': row['gpu type'],
            'VRAM_per_GPU': row['VRAM per GPU (GiB)'],
            'Total_VRAM': row['Total VRAM (GiB)'],
            'Family_Type': family_type
        }
        
        for region in ALL_AWS_REGIONS:
            row_data[f'{region}_on_demand'] = quotas[family_type][region]['on_demand']
            row_data[f'{region}_spot'] = quotas[family_type][region]['spot']
        
        results.append(row_data)
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    
    print(f"\nSaved quota map to {output_path}")
    return output_df


if __name__ == '__main__':
    resource_table_path = 'resource_table.csv'
    output_path = 'aws_gpu_quota_by_region.csv'
    
    quota_map = create_quota_map_by_region(resource_table_path, output_path)
    
    print("\nQuota map created successfully!")
    print(f"Total rows: {len(quota_map)}")
    print("\nSample output:")
    print(quota_map.head())
