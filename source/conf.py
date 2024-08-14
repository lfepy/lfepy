# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lfepy'
copyright = '2024, Prof. Dr. Khalid M. Hosny, B.Sc. Mahmoud A. Mohamed'
author = 'Prof. Dr. Khalid M. Hosny, B.Sc. Mahmoud A. Mohamed'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # Automatically extract documentation from docstrings
    'sphinx.ext.napoleon',   # Supports Google-style and NumPy-style docstrings
]

templates_path = ['_templates']
exclude_patterns = []

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('..'))