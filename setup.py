from setuptools import setup, find_packages

setup(
    name="comed",
    version="2.1.7",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.2",
        "numpy>=1.26.0",
        "biopython>=1.85",
        "tqdm>=4.67.1",
        "openai>=1.65.1",
        "requests>=2.32.3",
    ],
    author="Studentiz",
    author_email="studentiz@live.com",
    description="CoMed: A comprehensive framework for analyzing drug co-medication risks using RAG, Chain-of-Thought reasoning, and multi-agent collaboration. FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY - NOT FOR CLINICAL USE.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/studentiz/comed",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",  # Compatible with Python 3.8 and above
)
