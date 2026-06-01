from setuptools import setup, find_packages

setup(
    name="style",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.5.0",
        "sentence-transformers>=2.0.0",
        "rank-bm25>=0.2.2",
        "openai>=0.27.0",
        "wandb>=0.12.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.60.0",
        "python-dotenv>=0.19.0",
        "fastdtw>=0.3.4",
    ],
    python_requires=">=3.7",
    author="Arshia Ialty",
    author_email="ialtyarshia@gmail.com",
    description="STYLE: Improving Domain Transferability of Asking Clarification Questions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ArshiaIlaty/STYLE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
