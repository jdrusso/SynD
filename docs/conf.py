# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import synd

# -- Project information -----------------------------------------------------

project = 'SynD'
copyright = '2022, John Russo'
author = 'John Russo'


# -- General configuration ---------------------------------------------------
# Workaround for nbsphinx not reading directories outside of docs root
#   See: https://github.com/spatialaudio/nbsphinx/issues/170
print("Copy example notebooks into docs/_examples")
import shutil
def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        if os.path.isfile(os.path.join(dir,c)) and (not c.endswith(".ipynb")):
            result += [c]
    return result

project_root = os.path.abspath('../')
shutil.rmtree(os.path.join(project_root, "docs/_examples"), ignore_errors=True)
shutil.copytree(os.path.join(project_root, "examples"),
                os.path.join(project_root, "docs/_examples"),
                ignore=all_but_ipynb)



# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon',
              'sphinx.ext.todo', 'sphinx.ext.autosummary', 'nbsphinx', 'sphinx_autodoc_typehints']

autodoc_default_options = {
    'members':           True,
    'undoc-members':     True,
    'member-order':      'bysource',
}
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True
napoleon_include_init_with_doc = True
autodoc_typehints = 'signature'
# autodoc_typehints = 'none'
autodoc_type_aliases = {
    'Iterable': 'Iterable',
    'ArrayLike': 'ArrayLike',
    'npt.ArrayLike': 'ArrayLike'
}

# autoclass_content = 'both'
autodoc_member_order = 'bysource'

# autodoc_default_flags = ['methods']
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = synd.__version__
# The full version, including alpha/beta/rc tags.
release = synd.__version__

# The language for content autogenerated by Sphinx.
language = "en"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
mathjax_path = ("https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?"
                "config=TeX-AMS-MML_HTMLorMML")
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'msm_wedoc'

# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'synd.tex',
     'synd Documentation',
     'John Russo', 'manual'),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'synd',
     'synd Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'synd',
     'synd Documentation',
     author,
     'synd',
     'One line description of project.',
     'Miscellaneous'),
]
