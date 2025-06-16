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
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for Api documentation --------------------------------------------
autoapi_dirs = ["../../src/kaiju_mapper/"]
autoapi_ignore = ["**/test*.py"]
autoapi_root = "api"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_css_files = ["css/custom.css"]

# -- Options for example gallery ----------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": ["../../examples/datasets/", "../../examples/pipelines/"],
    "gallery_dirs": ["dataset-examples", "pipeline-examples"],
    "filename_pattern": "/*.py",
    "remove_config_comments": True,
    "download_all_examples": False,
}
