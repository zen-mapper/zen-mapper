# List the tasks by default
@_default:
	just --list

alias fmt := format

# Run code formatters
format:
	ruff format src
	ruff check src --fix
	nix fmt

# Check for style issues
lint:
	ruff format --check src
	ruff check src

# Run the test suite
test:
	pytest .

# Build the docs
docs:
	sphinx-apidoc -f -o docs/source src '**/test_*.py'
	cd docs && make html

