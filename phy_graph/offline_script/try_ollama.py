import ollama
from ollama import chat
import time
import json
import cv2
import base64
import time

class OllamaInference:
    def __init__(self, model_name="qwen3-vl:4b"):
        self.model_name = model_name
        self.result = None

    def _encode_cv_image_to_base64(self, cv_image):
        """Encodes an OpenCV image to a base64 string with compression."""
        # Set JPEG quality to 85 to reduce size without significant visual loss for VLM
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        success, buffer = cv2.imencode('.jpg', cv_image, encode_param)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode('utf-8')

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
            Analyze the object in this image, which is a reprojected image from 3D reconstruction, and its quality may be degraded due to reconstruction artifacts. This object is labeled as a '{label}'. 
            Based on the visual information, provide its estimated physical properties.
            IMPORTANT: Output ONLY a valid JSON object. Do not include any thinking, reasoning, explanation, or text outside the JSON object.
            Return the information in a JSON object with the following keys and value types:
            - "description": A brief string description of the object.
            - "friction_level": An integer from 0 (very low friction, e.g., ice) to 2 (high friction, e.g., rubber).
            - "pushable": An integer, 1 if a standard mobile robot could likely push it, 0 otherwise.
            - "weight_level": An integer from 0 (light, e.g., plastic bottle) to 2 (heavy, e.g., metal cabinet).
            - "estimated_weight_kg": A string representing the estimated weight range in kg (e.g., "0.5-2", "5-10", "20-50"). Be realistic based on the object type and apparent size.
            
            Do not include any text outside of the JSON object itself.
            """

            for attempt in range(max_retries):
                try:
                    print(f"[DEBUG] Sending request to Ollama API (Attempt {attempt+1}/{max_retries})...")
                    self.result = ollama.chat(model=self.model_name, 
                        messages=[{"role": "system", "content": "Directly answer the question without any internal thought process."},
                            {"role": "user", "content": prompt_text,"images": [base64_image]}],
                        think=False,
                        stream=False)
                    
                    if not self.result.message.content:
                        print("[ERROR] Empty message content")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        return {"error": "Empty response content from API"}
                    
                    return json.loads(self.result.message.content)
                
                except Exception as e:
                    print(f"[ERROR] An error occurred while communicating with API: {e}")
                    if attempt < max_retries - 1:
                        print(f"[DEBUG] Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        return {"error": str(e)}

            return {"error": "Max retries exceeded"}

if __name__ == "__main__":
    inference = OllamaInference(model_name="qwen3-vl:4b")
    image = cv2.imread("test.jpg")
    start_time = time.time()
    result = inference.get_properties_from_image(image, "swivel chair")
    end_time = time.time()
    print(result)
    print(f"Time taken: {end_time - start_time} seconds")