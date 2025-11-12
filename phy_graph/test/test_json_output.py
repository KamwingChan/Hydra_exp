#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_json_output():
    """Test the JSON output functionality"""
    print("=== JSON Output Test ===")
    
    # Check if output directory exists
    output_base_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    if not os.path.exists(output_base_dir):
        print("❌ Output directory does not exist yet")
        print("   This is normal if the service hasn't been run yet")
        return False
    
    # Find the most recent output directory
    subdirs = [d for d in os.listdir(output_base_dir) 
               if os.path.isdir(os.path.join(output_base_dir, d))]
    
    if not subdirs:
        print("❌ No timestamped output directories found")
        return False
    
    # Sort by name (which should be timestamp-based)
    subdirs.sort(reverse=True)
    latest_dir = os.path.join(output_base_dir, subdirs[0])
    
    print(f"✅ Found latest output directory: {latest_dir}")
    
    # Check for JSON files
    json_files = [f for f in os.listdir(latest_dir) if f.endswith('.json')]
    
    if not json_files:
        print("❌ No JSON files found in output directory")
        return False
    
    print(f"✅ Found {len(json_files)} JSON files")
    
    # Validate JSON files
    valid_files = 0
    for json_file in json_files:
        filepath = os.path.join(latest_dir, json_file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required fields
            required_fields = ["object_id", "label", "description", "friction_level", "pushable", "processing_time_ms"]
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                print(f"❌ {json_file}: Missing fields: {missing_fields}")
            else:
                print(f"✅ {json_file}: Valid JSON with all required fields")
                print(f"   - Object ID: {data['object_id']}")
                print(f"   - Label: {data['label']}")
                print(f"   - Description: {data['description']}")
                print(f"   - Friction Level: {data['friction_level']}")
                print(f"   - Pushable: {data['pushable']}")
                print(f"   - Processing Time: {data['processing_time_ms']}ms")
                valid_files += 1
                
        except json.JSONDecodeError as e:
            print(f"❌ {json_file}: Invalid JSON - {e}")
        except Exception as e:
            print(f"❌ {json_file}: Error reading file - {e}")
    
    print(f"\n📊 Summary: {valid_files}/{len(json_files)} valid JSON files")
    
    return valid_files > 0

if __name__ == "__main__":
    success = test_json_output()
    if success:
        print("\n🎉 JSON output functionality is working correctly!")
    else:
        print("\n⚠️  JSON output functionality needs to be tested with actual service calls")
    
    sys.exit(0 if success else 1)
