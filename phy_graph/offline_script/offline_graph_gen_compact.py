#!/usr/bin/env python3
"""
Compatibility entrypoint: offline_graph_gen_compact.py

This simply forwards to offline_graph_gen.py (compact DSG -> scene_graph_compact.json).
"""

from offline_graph_gen import main


if __name__ == "__main__":
    main()


