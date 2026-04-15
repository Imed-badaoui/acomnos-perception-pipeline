# ACOMNOS – Real-Time AI Perception Pipeline

## Overview
AI-driven sequence-to-text translation system using computer vision and deep learning. Built for real-time inference with GPU acceleration and memory-constrained deployment.

## Key Features
- Real-time perception pipeline with MediaPipe + TensorFlow
- GPU acceleration via CUDA (NVIDIA RTX 4060)
- Memory-optimized data loading (~40% reduction in peak RAM usage)
- Edge deployment exploration: TensorFlow Lite + post-training quantization
- Production-ready backend: FastAPI + Docker + CI/CD (GitHub Actions)

## Tech Stack
- **ML/CV**: TensorFlow, MediaPipe, OpenCV, NumPy
- **Backend**: FastAPI, Python, Docker
- **DevOps**: Git, GitHub Actions, CI/CD
- **Optimization**: CUDA, TFLite, Quantization, Memory Mapping

## Getting Started
```bash
# Clone the repository
git clone https://github.com/Imed-badaoui/acomnos-perception-pipeline.git
cd acomnos-perception-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the backend (requires GPU for optimal performance)
uvicorn app.main:app --reload
