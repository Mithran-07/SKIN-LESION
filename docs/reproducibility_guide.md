# Reproducibility Guide

## Environment
- Python virtual environment
- Dependencies from `requirements.txt`
- Apple Silicon workstation used only for research and static validation

## Steps
1. Activate the environment.
2. Run the test suite.
3. Validate model shapes.
4. Run the model analysis self-test.
5. Run the inference help command.
6. Generate publication figures.

## Recommended Validation Commands
- pytest for the test suite
- model analysis self-test for parameter and FLOP estimates
- inference help for CLI parsing

## Notes
No training or dataset downloads are performed on the MacBook workstation.
