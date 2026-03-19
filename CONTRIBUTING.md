# Contributing to BurmeseGLUE

Thank you for your interest in contributing to BurmeseGLUE! This document provides guidelines for contributing to the project.

## How to Contribute

### 1. Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Check existing issues before creating a new one
- Provide clear descriptions with:
  - Steps to reproduce (for bugs)
  - Expected vs actual behavior
  - Environment details (Python version, OS, etc.)

### 2. Code Contributions

#### Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/burmeseglue.git
   cd burmeseglue
   ```
3. Create a development branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

#### Making Changes

1. Write clear, documented code
2. Follow existing code style (PEP 8)
3. Add tests for new features
4. Update documentation as needed

#### Code Style

We use:
- **black** for code formatting
- **flake8** for linting
- **pytest** for testing

Before committing, run:
```bash
black burmeseglue/ scripts/ tests/
flake8 burmeseglue/ scripts/ tests/
pytest tests/ -v
```

#### Commit Guidelines

- Use clear, descriptive commit messages
- Follow conventional commits format:
  ```
  feat: Add sentence similarity dataset loader
  fix: Correct tokenization in NER task
  docs: Update README with installation instructions
  test: Add tests for metrics module
  ```

### 3. Adding New Tasks

To add a new task to BurmeseGLUE:

1. **Create dataset loader** (`burmeseglue/datasets/your_task.py`):
   - Inherit from `BurmeseGLUEDataset`
   - Implement `download()`, `load_split()`, `get_labels()`
   - Follow existing format (JSON Lines)

2. **Create task processor** (`burmeseglue/tasks/your_task.py`):
   - Inherit from `Task`
   - Implement required methods
   - Add data formatting for models

3. **Add metrics** (`burmeseglue/metrics/your_metrics.py`):
   - Implement task-specific metrics
   - Use sklearn or custom implementations
   - Add unit tests

4. **Update configuration**:
   - Add task config to `configs/baseline.yaml` and `configs/transformer.yaml`
   - Update `__init__.py` files

5. **Add documentation**:
   - Update README with task description
   - Add example usage
   - Document data format

6. **Add tests**:
   - Test dataset loading
   - Test task processors
   - Test metrics

### 4. Adding New Datasets

If you have a Burmese dataset to contribute:

1. Ensure proper licensing and permissions
2. Follow the standardized format (JSON Lines)
3. Create train/val/test splits (70/15/15)
4. Document data source and citation
5. Implement loader following existing patterns

### 5. Submitting Leaderboard Results

To submit benchmark results:

1. Run full benchmark:
   ```bash
   python scripts/benchmark.py --model your-model --output results.json
   ```
2. Create a PR with:
   - Results JSON file
   - Model description
   - Training details
   - Reproducibility information

### 6. Pull Request Process

1. Update your branch with latest main:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. Push to your fork:
   ```bash
   git push origin your-branch
   ```

3. Create a Pull Request:
   - Provide clear description
   - Link related issues
   - Ensure CI passes

4. Address review feedback

5. Once approved, maintainers will merge

## Development Guidelines

### Code Quality

- Write docstrings for all public functions/classes
- Use type hints where appropriate
- Keep functions focused and modular
- Avoid deep nesting (max 3 levels)

### Testing

- Write tests for new features
- Maintain >80% code coverage
- Test edge cases and error conditions
- Use pytest fixtures for common setups

### Documentation

- Update README for user-facing changes
- Add docstrings for API documentation
- Include examples for new features
- Keep documentation in sync with code

## Project Structure

```
burmeseglue/
├── datasets/       # Data loaders (inherit from BurmeseGLUEDataset)
├── tasks/          # Task processors (inherit from Task)
├── models/         # Model implementations
├── metrics/        # Evaluation metrics
├── training/       # Training utilities
├── evaluation/     # Evaluation pipeline
└── utils/          # Common utilities
```

## Communication

- **GitHub Issues**: For bugs and feature requests
- **Pull Requests**: For code contributions
- **Discussions**: For questions and ideas

## Code of Conduct

Be respectful and inclusive:
- Use welcoming language
- Respect differing viewpoints
- Accept constructive criticism
- Focus on what's best for the community

## Questions?

If you have questions:
1. Check existing issues and documentation
2. Open a GitHub Discussion
3. Contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to BurmeseGLUE!
