#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import base64
from openai import OpenAI

class PhysicalInference:
    """A class to infer physical properties of an object from an image."""

    def __init__(self, api_key):
        """
        Initializes the PhysicalInference client.
        :param api_key: Your OpenAI API key.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=api_key)

    def _encode_image_to_base64(self, image_path):
        """Encodes an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _clean_json_response(self, content):
        """
        Clean the response content to extract JSON from markdown code blocks.
        
        :param content: Raw response content from OpenAI
        :return: Cleaned JSON string
        """
        if not content:
            return content
        
        # Remove leading and trailing whitespace
        content = content.strip()
        
        # Check if content is wrapped in markdown code blocks
        if content.startswith('```json'):
            # Find the end of the code block
            end_marker = content.find('```', 7)  # Start searching after '```json'
            if end_marker != -1:
                # Extract content between ```json and ```
                content = content[7:end_marker].strip()
        elif content.startswith('```'):
            # Handle generic code blocks
            first_newline = content.find('\n')
            if first_newline != -1:
                end_marker = content.find('```', first_newline)
                if end_marker != -1:
                    content = content[first_newline+1:end_marker].strip()
        
        return content

    def get_properties(self, image_path, label):
        """
        Analyzes an image to determine its physical properties.
        
        :param image_path: Path to the image file.
        :param label: The semantic label of the object in the image.
        :return: A dictionary with the object's properties.
        """
        print(f"[DEBUG] Processing image: {image_path} with label: {label}")
        
        try:
            base64_image = self._encode_image_to_base64(image_path)
            print(f"[DEBUG] Image encoded to base64, length: {len(base64_image)} characters")
        except Exception as e:
            print(f"[ERROR] Failed to encode image: {e}")
            return {"error": f"Image encoding failed: {str(e)}"}

        prompt_text = f"""
        Analyze the object in this image, which is a reprojected image from 3D reconstruction, and its quality may be degraded due to reconstruction artifacts. This object is labeled as a '{label}'. 
        Based on the visual information, provide its estimated physical properties.
        Return the information in a JSON object with the following keys and value types:
        - "description": A brief string description of the object.
        - "friction_level": An integer from 0 (very low friction, e.g., ice) to 2 (high friction, e.g., rubber).
        - "pushable": An integer, 1 if a standard mobile robot could likely push it, 0 otherwise.
        
        Do not include any text outside of the JSON object itself.
        """

        try:
            print("[DEBUG] Sending request to OpenAI API...")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            # print(f"[DEBUG] API response received. Response object type: {type(response)}")
            
            # Check if response has choices
            if not hasattr(response, 'choices') or not response.choices:
                print("[ERROR] No choices in API response")
                return {"error": "No choices in API response"}
            
            print(f"[DEBUG] Number of choices: {len(response.choices)}")
            
            # Check if first choice has message
            first_choice = response.choices[0]
            if not hasattr(first_choice, 'message') or not first_choice.message:
                print("[ERROR] No message in first choice")
                return {"error": "No message in API response"}
            
            # Check if message has content
            message_content = first_choice.message.content
            # print(f"[DEBUG] Raw message content: '{message_content}'")
            print(f"[DEBUG] Content type: {type(message_content)}, length: {len(message_content) if message_content else 0}")
            
            if not message_content:
                print("[ERROR] Empty message content")
                return {"error": "Empty response content from API"}
            
            # Clean the message content to remove markdown code blocks if present
            cleaned_content = self._clean_json_response(message_content)
            print(f"[DEBUG] Cleaned content: '{cleaned_content}'")
            
            # Try to parse JSON
            try:
                parsed_result = json.loads(cleaned_content)
                # print(f"[DEBUG] Successfully parsed JSON: {parsed_result}")
                return parsed_result
            except json.JSONDecodeError as json_error:
                print(f"[ERROR] JSON parsing failed: {json_error}")
                print(f"[ERROR] Original content: '{message_content}'")
                print(f"[ERROR] Cleaned content that failed to parse: '{cleaned_content}'")
                return {"error": f"Invalid JSON response: {str(json_error)}"}
        
        except Exception as e:
            # Log the exception or handle it as needed
            print(f"[ERROR] An error occurred while communicating with OpenAI API: {e}")
            print(f"[ERROR] Exception type: {type(e)}")
            return {"error": str(e)}
