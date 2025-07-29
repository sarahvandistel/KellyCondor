from setuptools import setup, find_packages

setup(
    name="kellycondor",
    version="0.1.0",
    description="A live-paper-ready SPX 0DTE iron-condor engine using Kelly-criterion-based sizing",
    author="Sarah",
    author_email="sarah@example.com",
    url="https://github.com/YourUser/KellyCondor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0", 
        "backtrader>=1.9.78.123",
        "redis>=4.5.0",
        "databento>=0.32.0",
        "ibapi>=9.81.1",
        "dash>=2.14.0",
        "plotly>=5.17.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "kelly-replay=kellycondor.replay:main",
        ]
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
) 