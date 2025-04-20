# List the tasks by default
@_default:
	just --list

alias fmt := format

# Run code formatters
format:
	ruff format .
	ruff check src --fix
	nix fmt

# Check for style issues
lint:
	ruff format --check .
	ruff check src
	alejandra --check .

# Run the test suite
test:
	pytest .

# Run the nix checks
check check:
	nix build .#checks.x86_64-linux.{{check}}

# Build the docs
doc-build:
	cd docs && make dirhtml

# Serve the docs
doc-serve: doc-build
	python -m http.server -d docs/build/dirhtml

# Build the package
build:
	uv build

# Clean up the working directory
clean:
	rm -rf dist/ docs/build/ result
	find src/ -type f -name "*.pyc" -delete
	find src/ -type d -name "__pycache__" -delete
