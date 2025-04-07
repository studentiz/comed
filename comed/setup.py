from setuptools import setup, find_packages

setup(
    name="comed",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "biopython",
        "tqdm",
        "openai>=1.0.0",
        "requests",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="CoMed: A framework for analyzing co-medication risks using Chain-of-Thought reasoning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/comed",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12.9",
)
