cmake_minimum_required(VERSION 3.0.2)
project(phy_graph)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
  cv_bridge
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  rospack
)

catkin_python_setup()

find_package(hydra REQUIRED)
find_package(hydra_ros REQUIRED)
find_package(hydra_msgs REQUIRED)
find_package(kimera_pgmo_msgs REQUIRED)

add_service_files(
  FILES
  GetProperties.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  kimera_pgmo_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs message_runtime kimera_pgmo_msgs cv_bridge tf2_ros tf2_eigen tf2_geometry_msgs rospack
  DEPENDS hydra hydra_ros hydra_msgs kimera_pgmo_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${hydra_INCLUDE_DIRS}
  ${hydra_ros_INCLUDE_DIRS}
  ${hydra_msgs_INCLUDE_DIRS}
  ${kimera_pgmo_msgs_INCLUDE_DIRS}
)

add_executable(phy_graph_node 
  src/nodeSub.cpp
  src/imageProcessor.cpp
)
add_dependencies(phy_graph_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(phy_graph_node
  ${catkin_LIBRARIES}
  ${hydra_LIBRARIES}  
  ${hydra_ros_LIBRARIES}  
  ${hydra_msgs_LIBRARIES}
  ${kimera_pgmo_msgs_LIBRARIES}
)

install(PROGRAMS scripts/module.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
