# Contributing to DentaVision

Welcome to the DentaVision team! To maintain high code quality and clinical compliance, please follow these guidelines.

## Branching Strategy
- **main**: Production-ready code only. Do not push directly here.
- **develop**: Integration branch for features.
- **feature/[member-name]/[task]**: Your personal work branch (e.g., `feature/bilal/r2-unet-training`).

## Pull Request (PR) Process
1. **Sync**: Ensure your branch is up-to-date with `develop` before opening a PR.
2. **Review**: Every PR requires at least **one approval** from a relevant Code Owner (e.g., Bilal for AI changes).
3. **Tests**: Ensure all local tests pass and the FastAPI server starts without errors.
4. **DICOM Safety**: Double-check that any new data handling logic includes PII stripping.

## Coding Standards
- **Python**: Follow PEP 8. Use type hints for all FastAPI endpoints.
- **AI**: All model changes must be logged in Weights & Biases (W&B).
- **Frontend**: Use functional components and Tailwind CSS for styling.

## Clinical Standards
All contributors must ensure that logic aligns with the **STS-Tooth (2025)** research paper and **ANSI/ADA 1110-1:2025** guidelines.
