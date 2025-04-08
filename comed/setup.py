from setuptools import setup, find_packages

setup(
    name="comed",
    version="1.0.2",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.3",
        "numpy>=2.2.3",
        "biopython>=1.85",
        "tqdm>=4.67.1",
        "openai>=1.65.1",
        "requests>=2.32.3",
    ],
    author="Studentiz",
    author_email="studentiz@live.com",
    description="CoMed: A framework for analyzing co-medication risks using Chain-of-Thought reasoning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/studentiz/comed",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12.9",
)
