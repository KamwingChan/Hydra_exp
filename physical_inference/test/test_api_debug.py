#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
from PIL import Image, ImageDraw
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from physical_inference.physical_inference_lib.inferenceCore import PhysicalInference

def create_test_image():
    """Create a simple test image for debugging"""
    # Create a simple test image (a red square)
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill='red', outline='black', width=2)
    draw.text((75, 75), "TEST", fill='black')
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(temp_file.name)
    temp_file.close()
    
    return temp_file.name

def test_api():
    """Test the OpenAI API with debug information"""
    print("=== OpenAI API Debug Test ===")
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return False
    
    print(f"✅ API key found (length: {len(api_key)})")
    
    try:
        # Initialize inference tool
        inference_tool = PhysicalInference(api_key=api_key)
        print("✅ PhysicalInference initialized successfully")
        
        # Create test image
        test_image_path = create_test_image()
        print(f"✅ Test image created: {test_image_path}")
        
        # Test the API call
        print("\n--- Starting API call test ---")
        result = inference_tool.get_properties(test_image_path, "test_object")
        
        print(f"\n--- API call completed ---")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Clean up
        os.unlink(test_image_path)
        print(f"✅ Cleaned up test image")
        
        if "error" in result:
            print(f"❌ API call failed with error: {result['error']}")
            return False
        else:
            print("✅ API call successful!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
