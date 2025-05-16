import sphinxcontrib.katex as katex

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Zen Mapper"
description = "Mapper without the noise"
copyright = "2023, Ethan Rooke"
author = "Ethan Rooke"
release = "0.2.0"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "autoapi.extension",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.katex",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Other documentation -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#confval-intersphinx_mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_theme_options = {
    "github_user": "zen-mapper",
    "github_repo": "zen-mapper",
}
html_static_path = ["_static"]
html_sidebars = {
    "**": [
        "about.html",
        "searchfield.html",
        "navigation.html",
    ]
}

# -- Options for Api documentation --------------------------------------------
autoapi_dirs = ["../../src/zen_mapper"]
autoapi_ignore = ["**/test*.py"]

# -- Options for example gallery ----------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "examples",
    "filename_pattern": "/*.py",
}

# -- Options for KaTeX rendering ----------------------------------------------
# https://sphinxcontrib-katex.readthedocs.io
katex_css_path = "https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css"
katex_js_path = "katex.min.js"
katex_autorender_path = "auto-render.min.js"
katex_inline = [r"\(", r"\)"]
katex_display = [r"\[", r"\]"]

# Prerender requires nodeJS and KaTeX executable on PATH,
# currently disabled for the above reason.
katex_prerender = False

# The following allows for the use of macros in HTML Sphinx builder
# and in LaTeX based builders.
latex_macros = r"""
    \def \RR            #1{\mathbb{R}}
    \def \RRn            #1{\mathbb{R}^{#1}}
"""
# Translates LaTeX macros to KaTeX macros/options
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = "macros: {" + katex_macros + "}"
# Adds LaTeX macros for LaTeX builder
latex_elements = {"preamble": latex_macros}
