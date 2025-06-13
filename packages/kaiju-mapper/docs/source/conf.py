# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "kaiju mapper"
copyright = "2025, Ethan Rooke, Jacob Miller"
author = "Ethan Rooke, Jacob Miller"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for Api documentation --------------------------------------------
autoapi_dirs = ["../../src/kaiju_mapper/"]
autoapi_ignore = ["**/test*.py"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
