import json
import os
import time
from pathlib import Path


class PromptGenerator:
    def __init__(self):
        self.prompt_dir = Path(__file__).parent / "prompts"
        self.prompt_file = self.prompt_dir / "prompt.txt"
        self.prompt = self.prompt_file.read_text()
        self.scene_graph = None

    def load_scene_graph(self, scene_graph_path: str):
        with open(scene_graph_path, "r") as f:
            self.scene_graph = json.load(f)

    def generate_prompt(self) -> str:
        return self.prompt.format(scene_graph=self.scene_graph)