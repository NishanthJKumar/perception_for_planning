#!/usr/bin/env python3
"""
Setup script for Gemini Robotics-ER Perception for Planning

This setup.py file installs all necessary dependencies for running the
Gemini Robotics-ER 1.5 object detection and bounding box examples.
"""

from setuptools import setup, find_packages

# Read README file for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Gemini Robotics-ER Perception for Planning"

setup(
    name="perception-for-planning",
    version="0.1.0",
    author="NishanthJKumar",
    description="TODO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NishanthJKumar/perception_for_planning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Google GenAI SDK for Gemini API
        "google-genai>=0.6.0",
        
        # Image processing
        "Pillow>=9.0.0",
        
        # HTTP requests and utilities
        "requests>=2.25.0",
        
        # Numerical and scientific computing
        "numpy>=1.21.0",
        "scipy>=1.6.0",
        
        # Visualization
        "matplotlib>=3.5.0",
        "supervision>=0.17.0",  # For segmentation visualization
        
        # 3D processing
        "open3d>=0.16.0",       # Point cloud processing
        "trimesh>=3.9.0",       # 3D mesh handling
        
        # API integrations
        "replicate>=0.18.0",    # For SAM-2 via Replicate API

        "segment-anything>=1.0",
    ],
    extras_require={
    },
    entry_points={
    },
    project_urls={
        "Bug Reports": "https://github.com/NishanthJKumar/perception_for_planning/issues",
        "Source": "https://github.com/NishanthJKumar/perception_for_planning",
        "Documentation": "https://ai.google.dev/gemini-api/docs/robotics-overview",
    },
)