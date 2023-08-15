from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).resolve().parent
with (this_dir / "requirements.txt").open() as rf: 
    install_requires = [ 
        req.strip()
        for req in rf.readlines()
        if req.strip() and not req.startswith("#")
    ]   

setup(
    name="my_toolkit",
    version="0.0.0",
    author="Chaoyi",
    url="https://github.com/chaoyi-lyu/Deep_Learning_for_Particle_Physics",
    packages=find_packages(),
    description=""" A toolkit for applications of deep learning to particle physics """,
    install_requires=install_requires
)