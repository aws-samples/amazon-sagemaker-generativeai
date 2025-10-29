from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("sama_rl/requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="sama_rl",
    version="0.1.0",
    description="Simple API for Reinforcement Learning with Language Models",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    author="SAMA Team",
    author_email="team@sama.ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
