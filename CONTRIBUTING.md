# Contributing to Agentic RAG System

Thank you for your interest in contributing to the Agentic RAG System! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Adding Features](#adding-features)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's coding standards

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Bdm_Project_IITM.git
   cd Bdm_Project_IITM
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/LalitIITM/Bdm_Project_IITM.git
   ```

4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Run tests to verify setup:
   ```bash
   python -m unittest discover tests
   ```

## Making Changes

### Types of Changes

- **Bug fixes**: Fix existing functionality
- **Features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve tests
- **Refactoring**: Improve code without changing functionality

### Branch Naming

- Features: `feature/short-description`
- Bug fixes: `fix/short-description`
- Documentation: `docs/short-description`
- Tests: `test/short-description`

### Commit Messages

Follow this format:
```
<type>: <short summary>

<detailed description (optional)>

<footer (optional)>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `style`: Code style changes
- `chore`: Maintenance tasks

Examples:
```
feat: Add sentiment analysis tool to agent registry

Added a sentiment analysis tool that agents can use to analyze
the emotional tone of text. Includes positive, negative, and
neutral classification.
```

```
fix: Resolve memory leak in agent orchestrator

Fixed memory accumulation issue by properly clearing agent
memory after processing. Closes #123
```

## Testing

### Running Tests

Run all tests:
```bash
python -m unittest discover tests
```

Run specific test file:
```bash
python -m unittest tests.test_agents
```

Run specific test:
```bash
python -m unittest tests.test_agents.TestBaseAgent.test_agent_initialization
```

### Writing Tests

1. Create test file in `tests/` directory
2. Follow naming convention: `test_<module>.py`
3. Use unittest framework
4. Mock external dependencies

Example:
```python
import unittest
from unittest.mock import Mock
from app.agents.base_agent import BaseAgent

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.agent = MyAgent()
    
    def test_my_feature(self):
        """Test description."""
        result = self.agent.my_method()
        self.assertEqual(result, expected_value)
```

### Test Coverage

Aim for:
- All new code: 80%+ coverage
- Critical paths: 100% coverage
- Edge cases: Well tested

## Submitting Changes

1. Ensure all tests pass:
   ```bash
   python -m unittest discover tests
   ```

2. Update documentation if needed

3. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request:
   - Go to GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template
   - Submit

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Test addition

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Manual testing done

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes
```

## Code Style

### Python Style Guide

Follow PEP 8 with these specifics:

**Imports:**
```python
# Standard library
import os
import sys

# Third-party
from flask import Flask, request
from langchain import LLMChain

# Local
from app.agents.base_agent import BaseAgent
from app.agents.tools import tool_registry
```

**Naming Conventions:**
```python
# Classes: PascalCase
class RetrievalAgent:
    pass

# Functions/Methods: snake_case
def process_query(query):
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3

# Private: _leading_underscore
def _internal_method():
    pass
```

**Documentation:**
```python
def complex_function(param1: str, param2: int) -> dict:
    """
    Short description.
    
    Longer description with more details about what the
    function does and how it works.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
    """
    pass
```

**Type Hints:**
```python
from typing import Dict, List, Optional

def get_documents(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, str]] = None
) -> List[Document]:
    pass
```

### Code Quality Tools

Run these before submitting:

```bash
# Format code
black backend/

# Check style
flake8 backend/

# Type checking (if mypy installed)
mypy backend/
```

## Adding Features

### Adding a New Agent

1. Create agent file in `backend/app/agents/`:
   ```python
   from .base_agent import BaseAgent
   
   class MyAgent(BaseAgent):
       def perceive(self, input_data):
           # Implementation
           pass
       
       def reason(self, perception):
           # Implementation
           pass
       
       def act(self, reasoning):
           # Implementation
           pass
   ```

2. Add to `__init__.py`:
   ```python
   from .my_agent import MyAgent
   __all__ = [..., 'MyAgent']
   ```

3. Write tests in `tests/test_agents.py`:
   ```python
   class TestMyAgent(unittest.TestCase):
       # Test cases
       pass
   ```

4. Update documentation:
   - Add to `backend/app/agents/README.md`
   - Add examples to `backend/app/agents/examples.py`

### Adding a New Tool

1. Create tool function:
   ```python
   def my_tool(input_data):
       """Tool description."""
       # Implementation
       return result
   ```

2. Register in `tools.py`:
   ```python
   self.register_tool(
       "my_tool",
       self._my_tool,
       "Tool description"
   )
   ```

3. Write tests:
   ```python
   def test_my_tool(self):
       tool = self.registry.get_tool("my_tool")
       result = tool("test input")
       self.assertEqual(result, expected)
   ```

4. Document usage in examples

### Adding an Endpoint

1. Add route in `main.py`:
   ```python
   @app.route('/my_endpoint', methods=['POST'])
   def my_endpoint():
       try:
           data = request.json
           result = process_data(data)
           return jsonify({"status": "success", "result": result})
       except Exception as e:
           logger.error(f"Error: {e}")
           return jsonify({"status": "error", "message": str(e)})
   ```

2. Add tests:
   ```python
   def test_my_endpoint(self):
       response = self.client.post('/my_endpoint', json={...})
       self.assertEqual(response.status_code, 200)
   ```

3. Update API documentation in README

## Common Tasks

### Adding Dependencies

1. Install package:
   ```bash
   pip install package-name
   ```

2. Update requirements.txt:
   ```bash
   pip freeze > requirements.txt
   ```

3. Document why dependency is needed

### Updating Documentation

1. Update relevant files:
   - `README.md` - Main documentation
   - `QUICKSTART.md` - Getting started
   - `ARCHITECTURE.md` - Architecture details
   - `backend/app/agents/README.md` - Agent documentation

2. Ensure examples are up to date

3. Check for broken links

### Debugging Issues

1. Check logs:
   ```python
   logger.info("Debug message")
   logger.error("Error message")
   ```

2. Use Python debugger:
   ```python
   import pdb; pdb.set_trace()
   ```

3. Run specific tests:
   ```bash
   python -m unittest tests.test_agents.TestClass.test_method -v
   ```

## Review Process

1. **Automated Checks**: CI runs tests automatically
2. **Code Review**: Maintainers review code
3. **Discussion**: Address feedback and questions
4. **Approval**: At least one maintainer approval needed
5. **Merge**: Changes merged to main branch

## Questions?

- Open an issue for questions
- Check existing documentation
- Review closed issues and PRs

## Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Git commit history

Thank you for contributing! 🎉
