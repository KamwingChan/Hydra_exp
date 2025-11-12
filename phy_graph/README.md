# phy_graph

`phy_graph` is a ROS-based physical knowledge graph inference module.  
It generates structured scene graphs enriched with semantic and physical information by combining image analysis with physical reasoning models.

---

## 🚀 Overview
- Generate physical relationship graphs from images
- Integrate physical constraint reasoning modules
- Support multiple datasets (e.g., ADE20K, uHuman2)
- Provide ROS services and launch files
- Implemented with both Python and C++

---

## 🧩 Directory Structure
```
phy_graph/
├── config/           # Model and parameter configuration
├── include/          # Header files
├── launch/           # ROS launch files
├── src/              # Core source code
├── scripts/          # Supporting scripts
├── srv/              # ROS service definitions
├── test/             # Testing scripts
├── requirements.txt  # Python dependencies
└── setup.py          # Installation setup
```

---

## ⚙️ Requirements
- **ROS** (recommended: Noetic)
- **Python 3.8+**
- **CMake / catkin**
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## 🧠 Usage
Build the project:
```bash
cd ~/catkin_ws
catkin build phy_graph
source devel/setup.bash
```

Run the inference node:
```bash
roslaunch phy_graph inference.launch
```



