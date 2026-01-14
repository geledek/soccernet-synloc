"""Setup script for synloc package."""

from setuptools import setup, find_packages

setup(
    name='synloc',
    version='0.1.0',
    description='Standalone baseline for SoccerNet SynLoc challenge',
    author='SoccerNet Contributors',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'Pillow>=9.5.0',
        'albumentations>=1.3.0',
        'opencv-python-headless>=4.8.0',
        'xtcocotools>=1.12',
        'tqdm>=4.65.0',
        'PyYAML>=6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'jupyter>=1.0.0',
            'matplotlib>=3.7.0',
            'wandb>=0.15.0',
        ],
    },
)
