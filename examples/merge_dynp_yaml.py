#!/usr/bin/env python3
"""
Merge YAML configuration files into a single file.

Each input YAML file is identified by its (nodes, devices_per_node) configuration.
The script checks for duplicate configurations before merging.
"""

import os
import sys
import argparse
import glob
import yaml
from typing import Dict, List, Tuple, Any


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Merge YAML configuration files.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing YAML files to merge')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output merged YAML file')
    parser.add_argument('--dry_run', action='store_true',
                        help='Check for duplicates without merging files')
    return parser.parse_args()


def read_yaml_file(file_path: str) -> Dict[str, Any]:
    """Read a YAML file and return its contents.
    
    Uses FullLoader to handle Python-specific tags like !!python/tuple.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)


def get_config_id(config: Dict[str, Any]) -> Tuple[int, int]:
    """Extract the configuration ID (nodes, devices_per_node) from a config."""
    try:
        nodes = config.get('nodes')
        devices_per_node = config.get('devices_per_node')
        
        if nodes is None or devices_per_node is None:
            raise ValueError("Missing 'nodes' or 'devices_per_node' in config")
        
        return (nodes, devices_per_node)
    except Exception as e:
        print(f"Error extracting configuration ID: {e}")
        sys.exit(1)


def check_duplicate_configs(yaml_files: List[str]) -> Dict[Tuple[int, int], List[str]]:
    """
    Check for duplicate configurations across YAML files.
    
    Returns a dictionary mapping config IDs to lists of file paths that have that ID.
    """
    config_id_to_files = {}
    
    for file_path in yaml_files:
        config = read_yaml_file(file_path)
        config_id = get_config_id(config)
        
        if config_id not in config_id_to_files:
            config_id_to_files[config_id] = []
        
        config_id_to_files[config_id].append(file_path)
    
    # Filter to only include duplicates
    duplicates = {
        config_id: files 
        for config_id, files in config_id_to_files.items() 
        if len(files) > 1
    }
    
    return duplicates


def merge_yaml_files(yaml_files: List[str], output_file: str) -> None:
    """Merge multiple YAML files into a single file."""
    merged_configs = []
    
    for file_path in yaml_files:
        config = read_yaml_file(file_path)
        config_id = get_config_id(config)
        
        # Create entry with the format from Option 2
        merged_entry = {
            'nodes': config_id[0],
            'devices_per_node': config_id[1],
            'config': config
        }
        
        merged_configs.append(merged_entry)
    
    # Sort the configs by nodes, then devices_per_node for consistent output
    merged_configs.sort(key=lambda x: (x['nodes'], x['devices_per_node']))
    
    # Write the merged configurations to the output file
    try:
        with open(output_file, 'w') as file:
            # Use dump to preserve Python-specific types when writing
            yaml.dump(merged_configs, file, default_flow_style=False, sort_keys=False)
        print(f"Successfully merged {len(yaml_files)} YAML files into {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
        sys.exit(1)


def main():
    """Main function to merge YAML files."""
    args = parse_args()
    
    # Find all YAML files in the input directory
    yaml_files = glob.glob(os.path.join(args.input_dir, '*.yml'))
    yaml_files.extend(glob.glob(os.path.join(args.input_dir, '*.yaml')))
    
    if not yaml_files:
        print(f"No YAML files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(yaml_files)} YAML files in {args.input_dir}")
    
    # Check for duplicate configurations
    duplicates = check_duplicate_configs(yaml_files)
    
    if duplicates:
        print("Found duplicate configurations:")
        for config_id, files in duplicates.items():
            print(f"  Configuration {config_id} appears in:")
            for file in files:
                print(f"    - {file}")
        
        print("\nError: Cannot merge files with duplicate configurations.")
        sys.exit(1)
    
    # If this is just a dry run, exit now
    if args.dry_run:
        print("Dry run completed. No duplicates found.")
        sys.exit(0)
    
    # Merge the YAML files
    merge_yaml_files(yaml_files, args.output_file)


if __name__ == "__main__":
    main()