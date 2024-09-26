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
	cd docs && make dirhtml

# Build the package
build:
	hatch build

# Clean up the working directory
clean:
	rm -rf dist/ docs/build/ result
	find src/ -type f -name "*.pyc" -delete
	find src/ -type d -name "__pycache__" -delete
