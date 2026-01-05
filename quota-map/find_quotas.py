#!/usr/bin/env python3
import pandas as pd
from typing import Dict, List, Optional, Set
from sky.catalog import aws_catalog
from sky.adaptors import aws as aws_adaptor
from collections import defaultdict


# AWS Family-level quota codes (fallback mapping)
# These are standard AWS quota codes for EC2 instance families
FAMILY_QUOTA_CODES = {
    # GPU Graphics families - G instances
    'g4dn': {'on-demand': 'L-DB2E81BA', 'spot': 'L-3819A6DF'},  
    'g4ad': {'on-demand': 'L-DB2E81BA', 'spot': 'L-3819A6DF'},
    'g5': {'on-demand': 'L-DB2E81BA', 'spot': 'L-3819A6DF'},
    'g5g': {'on-demand': 'L-DB2E81BA', 'spot': 'L-3819A6DF'},
    'g6': {'on-demand': 'L-DB2E81BA', 'spot': 'L-3819A6DF'},
    'g6e': {'on-demand': 'L-DB2E81BA', 'spot': 'L-3819A6DF'},
    'g6f': {'on-demand': 'L-DB2E81BA', 'spot': 'L-3819A6DF'},
    'gr6': {'on-demand': 'L-DB2E81BA', 'spot': 'L-3819A6DF'},
    'gr6f': {'on-demand': 'L-DB2E81BA', 'spot': 'L-3819A6DF'},
    
    # GPU Compute families - P instances
    'p2': {'on-demand': 'L-417A185B', 'spot': 'L-7212CCBC'},  
    'p3': {'on-demand': 'L-417A185B', 'spot': 'L-7212CCBC'},
    'p3dn': {'on-demand': 'L-417A185B', 'spot': 'L-7212CCBC'},
    'p4d': {'on-demand': 'L-417A185B', 'spot': 'L-7212CCBC'},
    'p4de': {'on-demand': 'L-417A185B', 'spot': 'L-7212CCBC'},
    'p5': {'on-demand': 'L-417A185B', 'spot': 'L-C4BD4855'},
    'p5e': {'on-demand': 'L-417A185B', 'spot': 'L-C4BD4855'},
    'p5en': {'on-demand': 'L-417A185B', 'spot': 'L-C4BD4855'},
    'p6-b200': {'on-demand': 'L-417A185B', 'spot': 'L-7212CCBC'},
    'p6-b300': {'on-demand': 'L-417A185B', 'spot': 'L-7212CCBC'},
    
    # Inferentia families
    'inf1': {'on-demand': 'L-1945791B', 'spot': 'L-B5D1601B'},  
    'inf2': {'on-demand': 'L-1945791B', 'spot': 'L-B5D1601B'},
    
    # Trainium families
    'trn1': {'on-demand': 'L-2C3B7624', 'spot': 'L-6B0D517C'},  
    'trn1n': {'on-demand': 'L-2C3B7624', 'spot': 'L-6B0D517C'},
    'trn2': {'on-demand': 'L-2C3B7624', 'spot': 'L-6B0D517C'},
    
    # Gaudi (DL1)
    'dl1': {'on-demand': 'L-6E869C2A', 'spot': 'L-7212CCBC'}, }

def get_quota_code(instance_type: str, use_spot: bool) -> Optional[str]:
    """
    THis tries to get the quota code for the instance type, using skypilot,
    but then using these hardcoded values, it falls back to them.
    """
    quota_code = aws_catalog.get_quota_code(instance_type, use_spot)
    if quota_code:
        return quota_code
    
    # Fallback: use family-level quota codes
    print("Using Fallback, as I did not find any quota code for the instance type: ", instance_type)
    family = instance_type.split('.')[0]  #(e.g., 'g6e' from 'g6e.xlarge')
    mode = 'spot' if use_spot else 'on-demand'
    
    if family in FAMILY_QUOTA_CODES:
        return FAMILY_QUOTA_CODES[family][mode]
    
    return None

def get_all_aws_gpu_accelerators():
    accelerators = aws_catalog.list_accelerators(
        gpus_only=True,
        name_filter=None,
        region_filter=None,
        quantity_filter=None,
        all_regions=True
    )
    return accelerators


def get_all_gpu_instance_types() -> Set[str]:
    """
    Get all unique GPU instance types (e.g., 'g6e.xlarge', 'p4d.24xlarge').
    """
    accelerators = get_all_aws_gpu_accelerators()
    instance_types = set()
    
    for _, instance_list in accelerators.items():
        for instance_info in instance_list:
            if instance_info.instance_type:
                instance_types.add(instance_info.instance_type)
    
    return instance_types


def get_instance_types_by_gpu_family():
    """
    Group instance types by family (g5, g6, p3, p4, etc.).
    """
    instance_types = get_all_gpu_instance_types()
    families = defaultdict(set)
    
    for instance in instance_types:
        family = instance.split('.')[0] 
        families[family].add(instance)
    
    return families


def create_complete_quota_map(
    use_spot: bool = False,
    gpu_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a complete map of GPU quotas across all AWS regions.
    """
    print(f"🔍 Discovering all AWS GPU instance types...")

    # Get all unique instance types and regions
    instance_to_regions = defaultdict(set)
    
    # Use all_regions=True to get full region availability
    accelerators_all_regions = aws_catalog.list_accelerators(
        gpus_only=True,
        name_filter=gpu_filter,
        region_filter=None,
        quantity_filter=None,
        all_regions=True
    )
    
    for _, instance_list in accelerators_all_regions.items():
        for info in instance_list:
            instance_to_regions[info.instance_type].add(info.region)
    
    print(f"Found {len(instance_to_regions)} unique GPU instance types")
    
    # Check quotas
    results = []
    for instance_type, regions in instance_to_regions.items():
        family = instance_type.split('.')[0] 
        print(f"\n📊 Checking {instance_type} (family: {family}, {'spot' if use_spot else 'on-demand'})...")
        
        quota_code = get_quota_code(instance_type, use_spot)
        
        for region in sorted(regions):
            quota = None
            status = "Unknown"
            
            if quota_code:
                try:
                    client = aws_adaptor.client('service-quotas', region_name=region)
                    response = client.get_service_quota(
                        ServiceCode='ec2',
                        QuotaCode=quota_code
                    )
                    quota = response['Quota']['Value']
                    status = "Available" if quota > 0 else "Zero Quota"
                    if quota > 0:
                        print(f"  {region}: {quota} vCPUs")
                except Exception as e:
                    status = "Error"
                    error_msg = str(e)
                    print(f"  {region}: Error - {error_msg[:40]}")
            else:
                status = "No Quota Code"
            
            results.append({
                'InstanceType': instance_type,
                'Family': family,
                'Region': region,
                'QuotaCode': quota_code if quota_code else 'N/A',
                'Quota_vCPUs': quota if quota is not None else -1,
                'Status': status,
                'Mode': 'spot' if use_spot else 'on-demand'
            })
    
    return pd.DataFrame(results)


def print_gpu_summary():
    """Print a summary of all AWS GPUs and instance families."""
    print("="*80)
    print("AWS GPU ACCELERATORS SUMMARY")
    print("="*80)
    
    # Get all accelerators
    accelerators = get_all_aws_gpu_accelerators()
    
    print(f"\n📌 Total GPU Types: {len(accelerators)}")
    print("\nGPU Accelerators:")
    for i, gpu_name in enumerate(sorted(accelerators.keys()), 1):
        instance_count = len(accelerators[gpu_name])
        print(f"  {i:2d}. {gpu_name:15s} - {instance_count} instance types")
    
    print("\n" + "="*80)
    print("INSTANCE TYPE FAMILIES")
    print("="*80)
    
    families = get_instance_types_by_gpu_family()
    print(f"\n📌 Total Instance Families: {len(families)}")
    
    for family, instances in sorted(families.items()):
        print(f"\n{family} family ({len(instances)} types):")
        for instance in sorted(instances):
            print(f"  - {instance}")


# Example Usage
if __name__ == '__main__':
    import sys

    print_gpu_summary()
    
    print("\n" + "="*80)
    print("ALL AWS GPU TYPES")
    print("="*80)
    accelerators = get_all_aws_gpu_accelerators()
    for gpu in sorted(accelerators.keys()):
        print(f"  • {gpu}")
    
    # 3. Get all instance types
    print("\n" + "="*80)
    print("ALL GPU INSTANCE TYPES (sample)")
    print("="*80)
    instance_types = sorted(get_all_gpu_instance_types())
    print(f"Total: {len(instance_types)} instance types\n")
    
    # Show by family
    for family in ['g4dn', 'g5', 'g6', 'g6e', 'p3', 'p4d', 'p4de', 'p5']:
        matching = [it for it in instance_types if it.startswith(family + '.')]
        if matching:
            print(f"{family}: {', '.join(matching)}")
    
    if '--check-quotas' in sys.argv:
        # Determine if we want spot or on-demand quotas
        use_spot = '--spot' in sys.argv

        print("\n" + "="*80)
        mode_str = "SPOT" if use_spot else "ON-DEMAND"
        print(f"CHECKING QUOTAS ({mode_str} | This may take several minutes...)")
        print("="*80)
        
        quota_df = create_complete_quota_map(
            use_spot=use_spot,  # use_spot True if --spot, False otherwise
            gpu_filter=None  # None = all GPUs
        )
        
        print("\n📊 Quota Map (showing all results):")
        print(quota_df.to_string(index=False))
        
        # Filter to only save non-zero quotas
        quota_df_filtered = quota_df[quota_df['Quota_vCPUs'] > 0]
        
        print(f"\n📊 Summary:")
        print(f"  Total entries: {len(quota_df)}")
        print(f"  Available quotas (>0): {len(quota_df_filtered)}")
        print(f"  Zero quotas: {len(quota_df[quota_df['Quota_vCPUs'] == 0])}")
        print(f"  Failed checks: {len(quota_df[quota_df['Quota_vCPUs'] < 0])}")
        
        # Save only non-zero quotas
        output_filename = (
            'aws_gpu_quota_map_spot.csv' if use_spot else 'aws_gpu_quota_map.csv'
        )
        quota_df_filtered.to_csv(output_filename, index=False)
        print(f"\n✅ Saved available quotas to {output_filename}")