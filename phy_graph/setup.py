from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['phy_graph', 'phy_graph_lib'],
    package_dir={'': 'src'}
)

setup(**d)
