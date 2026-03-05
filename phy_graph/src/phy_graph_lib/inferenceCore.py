#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import base64
import cv2
import time
from openai import OpenAI


class PhysicalInference:
    """A class to infer physical properties of an object from an image."""

    def __init__(self, api_key, base_url=None, model_name="openai/gpt-4o-mini"):
        """
        Initializes the PhysicalInference client.
        :param api_key: Your OpenAI API key.
        :param base_url: The base URL for the API, for use with services like OpenRouter.
        :param model_name: The model to use (e.g., 'gpt-4o', 'openai/gpt-4o' for OpenRouter).
        """
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _encode_image_to_base64(self, image_path):
        """Encodes an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _encode_cv_image_to_base64(self, cv_image):
        """Encodes an OpenCV image to a base64 string with compression."""
        # Set JPEG quality to 85 to reduce size without significant visual loss for VLM
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        success, buffer = cv2.imencode('.jpg', cv_image, encode_param)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode('utf-8')

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
    
    def get_properties_from_image(self, cv_image, label, max_retries=3) -> dict:
        """
        Infer physical properties from an image using VLM.
        :param cv_image: OpenCV image
        :param label: Object label
        :param max_retries: Number of retry attempts for API calls
        """
        print(f"[DEBUG] Processing image directly from memory with label: {label} using model: {self.model_name}")
        
        try:
            base64_image = self._encode_cv_image_to_base64(cv_image)
            print(f"[DEBUG] Image encoded to base64, length: {len(base64_image)} characters")
        except Exception as e:
            print(f"[ERROR] Failed to encode image: {e}")
            return {"error": f"Image encoding failed: {str(e)}"}

        prompt_text = f"""
        Analyze the object in this image. The image is reprojected from a 3D reconstruction pipeline and may contain visual artifacts.
        If a red bounding box is drawn in the image, the object to analyze is the one inside that box (ignore other objects in the crop).
        Otherwise, analyze the main object. The object is labeled as "{label}".

        Return a JSON object with the following keys and value types:

        - "description": A single string that includes:
            1) Visual appearance (color, shape, material),
            2) Spatial relations to nearby visible objects or surfaces,
            3) Optional brief  description context.
          Use the following format inside the string:
            "[appearance] ... ; [spatial] relation1(target), relation2(target) ; [context] ..."

          Spatial relation types must be chosen from:
          ["on", "under", "inside", "attached_to", "left_of", "right_of", "front_of", "behind", "near", "touching"].

        - "friction_level": Integer from 0 (very low friction) to 2 (high friction).

        - "pushable": Integer, 1 if a standard mobile robot could likely push it, 0 otherwise.

        - "weight_level": Integer from 0 (light) to 2 (heavy), defined to distinguish different robot payload capacities:
          * 0 (light): < 5kg - Manipulable by small embodied robots (e.g., Locobot, Spot Mini) - books, cups, bottles, small items
          * 1 (medium): 5-20kg - Manipulable by medium-sized robots (e.g., PR2, Fetch, HSR) - chairs, monitors, microwave, small boxes
          * 2 (heavy): > 20kg - Requires large/industrial robots or beyond typical manipulation - tables, sofas, large appliances, furniture
          Be realistic and conservative: most mobile manipulation robots have 5-15kg payload capacity.

        - "estimated_weight_kg": A string representing a realistic weight range in kilograms 
          (for example: "0.1-0.5", "0.5-2", "5-10", "20-50").
          This should be consistent with the weight_level classification above.

        Important rules:
        - Only include spatial relations that can be reasonably inferred from the image.
        - Do not hallucinate invisible objects.
        - If uncertain due to reconstruction artifacts, use conservative wording such as "likely".
        - Do not output any text outside the JSON object.
        """

        for attempt in range(max_retries):
            try:
                print(f"[DEBUG] Sending request to OpenAI/OpenRouter API (Attempt {attempt+1}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a robotic perception engine. You output ONLY raw JSON. No markdown formatting, no explanations."
                        },
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
                
                if not hasattr(response, 'choices') or not response.choices:
                    print("[ERROR] No choices in API response")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return {"error": "No choices in API response"}
                
                first_choice = response.choices[0]
                if not hasattr(first_choice, 'message') or not first_choice.message:
                    print("[ERROR] No message in first choice")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return {"error": "No message in API response"}
                
                message_content = first_choice.message.content
                
                if not message_content:
                    print("[ERROR] Empty message content")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return {"error": "Empty response content from API"}
                
                cleaned_content = self._clean_json_response(message_content)
                # print(f"[DEBUG] Cleaned content: '{cleaned_content}'")
                
                try:
                    parsed_result = json.loads(cleaned_content)
                    return parsed_result
                except json.JSONDecodeError as json_error:
                    print(f"[ERROR] JSON parsing failed: {json_error}")
                    # Only retry JSON errors if it's not the last attempt
                    if attempt < max_retries - 1:
                        print("[DEBUG] Retrying due to JSON error...")
                        time.sleep(1)
                        continue
                    return {"error": f"Invalid JSON response: {str(json_error)}"}
            
            except Exception as e:
                print(f"[ERROR] An error occurred while communicating with API: {e}")
                if attempt < max_retries - 1:
                    print(f"[DEBUG] Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    return {"error": str(e)}

        return {"error": "Max retries exceeded"}
