from setuptools import setup, find_packages

setup(
    name="causalbert",
    version="0.1.0",
    description="CausalBERT tools: dataset prep, training & inference",
    author="Your Name",
    url="https://github.com/norygami/causalbert",
    packages=find_packages(exclude=["__pycache__", "log*", "tests*"]),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "transformers>=4.0.0",
        "datasets>=1.0.0",
        "spacy>=3.0.0",
        "wandb>=0.10.0",
        "tqdm>=4.0.0",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
