#!/usr/bin/env python3
"""
Complete AWS GPU Compute Space Mapper using SkyPilot
Discovers all GPU accelerators, instance types, and checks quotas across regions.
"""
import pandas as pd
from typing import Dict, List, Optional, Set
from sky.catalog import aws_catalog
from sky.adaptors import aws as aws_adaptor
from collections import defaultdict


def get_all_aws_gpu_accelerators() -> Dict[str, List]:
    """
    Get all GPU accelerators available on AWS using SkyPilot's catalog.
    
    Returns:
        Dict mapping accelerator names to list of instance type info
    """
    # gpus_only=True filters to only GPU accelerators
    # name_filter=None means get all
    # all_regions=True shows all regional availability
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
    
    Returns:
        Set of instance type strings
    """
    accelerators = get_all_aws_gpu_accelerators()
    instance_types = set()
    
    for gpu_name, instance_list in accelerators.items():
        for instance_info in instance_list:
            if instance_info.instance_type:
                instance_types.add(instance_info.instance_type)
    
    return instance_types


def get_instance_types_by_gpu_family() -> Dict[str, Set[str]]:
    """
    Group instance types by family (g5, g6, p3, p4, etc.).
    
    Returns:
        Dict mapping family name to set of instance types
    """
    instance_types = get_all_gpu_instance_types()
    families = defaultdict(set)
    
    for instance in instance_types:
        # Extract family (e.g., 'g6e' from 'g6e.xlarge')
        family = instance.split('.')[0]
        families[family].add(instance)
    
    return dict(families)


def get_detailed_gpu_info(accelerator_name: Optional[str] = None) -> pd.DataFrame:
    """
    Get detailed information about GPU accelerators and their instances.
    
    Args:
        accelerator_name: Filter by specific GPU (e.g., 'L40S', 'H100')
                         None = get all GPUs
    
    Returns:
        DataFrame with columns: GPU, InstanceType, Count, vCPUs, DeviceMem, 
                               HostMem, OnDemandPrice, SpotPrice, Region
    """
    accelerators = aws_catalog.list_accelerators(
        gpus_only=True,
        name_filter=accelerator_name,
        region_filter=None,
        quantity_filter=None,
        all_regions=True
    )
    
    rows = []
    for gpu_name, instance_list in accelerators.items():
        for info in instance_list:
            rows.append({
                'GPU': gpu_name,
                'InstanceType': info.instance_type,
                'GPUCount': info.accelerator_count,
                'vCPUs': info.cpu_count,
                'DeviceMemGB': info.device_memory,
                'HostMemGB': info.memory,
                'OnDemandPrice': info.price,
                'SpotPrice': info.spot_price,
                'Region': info.region,
            })
    
    return pd.DataFrame(rows)


def create_complete_quota_map(
    use_spot: bool = False,
    gpu_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a complete map of GPU quotas across all AWS regions.
    
    Args:
        use_spot: Check spot quotas (default: on-demand)
        gpu_filter: Filter to specific GPU (e.g., 'L40S', 'A100')
    
    Returns:
        DataFrame with quota information for all GPU instance types
    """
    print(f"🔍 Discovering all AWS GPU instance types...")
    
    # Get all accelerators
    accelerators = aws_catalog.list_accelerators(
        gpus_only=True,
        name_filter=gpu_filter,
        region_filter=None,
        quantity_filter=None,
        all_regions=False  # Get cheapest region per instance
    )
    
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
    
    for gpu_name, instance_list in accelerators_all_regions.items():
        for info in instance_list:
            instance_to_regions[info.instance_type].add(info.region)
    
    print(f"Found {len(instance_to_regions)} unique GPU instance types")
    
    # Check quotas
    results = []
    for instance_type, regions in instance_to_regions.items():
        print(f"\n📊 Checking {instance_type} ({'spot' if use_spot else 'on-demand'})...")
        
        quota_code = aws_catalog.get_quota_code(instance_type, use_spot)
        
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
                    status = "✅ Available" if quota > 0 else "❌ Zero Quota"
                    print(f"  {region}: {quota} vCPUs")
                except Exception as e:
                    status = "⚠️ Check Failed"
                    print(f"  {region}: Error - {str(e)[:40]}")
            else:
                status = "⚠️ No Quota Code"
            
            results.append({
                'InstanceType': instance_type,
                'Region': region,
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
    
    # 1. Print summary of all GPUs
    print_gpu_summary()
    
    # 2. List all AWS GPU types
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
    
    # 4. Detailed info for specific GPU
    print("\n" + "="*80)
    print("DETAILED INFO: L40S GPU")
    print("="*80)
    l40s_df = get_detailed_gpu_info('L40S')
    if not l40s_df.empty:
        print(l40s_df.to_string(index=False))
    
    # 5. Optional: Create full quota map (uncomment to run)
    # WARNING: This will make many API calls and take a while!
    if '--check-quotas' in sys.argv:
        print("\n" + "="*80)
        print("CHECKING QUOTAS (This may take several minutes...)")
        print("="*80)
        
        # Check quotas for L40S instances only (as example)
        quota_df = create_complete_quota_map(
            use_spot=False,
            gpu_filter=None
        )
        
        print("\n📊 Quota Map:")
        print(quota_df.to_string(index=False))
        
        # Save to CSV
        quota_df = quota_df[quota_df['Quota_vCPUs'] > 0.0]
        quota_df.to_csv('aws_gpu_quota_map.csv', index=False)
        print("\n✅ Saved to aws_gpu_quota_map.csv")