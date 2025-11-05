from setuptools import setup, find_packages

setup(
    name="ganglia-common",
    version="0.1.0",
    description="Shared utilities for GANGLIA ecosystem",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "google-cloud-texttospeech>=2.14.1",
        "google-cloud-storage>=2.10.0",
        "gTTS>=2.5.0",
        "requests>=2.31.0",
    ],
)


