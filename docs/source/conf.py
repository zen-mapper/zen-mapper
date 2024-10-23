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
release = "0.1.4"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "autoapi.extension",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode", # Added viewcode for source
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Include LaTeX Support ---------------------------------------------------
latex_engine = 'xelatex'
latex_elements = {
    'passoptionstopackages': r'''
\PassOptionsToPackage{svgnames}{xcolor}
''',
    'fontpkg': r'''
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
''',
    'preamble': r'''
\usepackage[titles]{tocloft}
\cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
\setlength{\cftchapnumwidth}{0.75cm}
\setlength{\cftsecindent}{\cftchapnumwidth}
\setlength{\cftsecnumwidth}{1.25cm}
''',
    'sphinxsetup': 'TitleColor=DarkGoldenrod',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
}
latex_show_urls = 'footnote'

# -- Other documentation -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#confval-intersphinx_mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    #Want to add link to networkx but can't find one that works
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,  # Enable scroll-following ToC
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
