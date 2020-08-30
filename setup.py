import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["tensorflow>=2.2.0"]

setuptools.setup(
    name="keras-kge",
    version="0.1.0",
    author="Stephan Baier",
    author_email="stephan.baier@gmail.com",
    description="Popular Knowledge Graph Embedding (KGE) models implemented with Keras.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baierst/keras-kge",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements
)