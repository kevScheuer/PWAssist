import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "src"))

project = "PWAssist"
author = "Kevin Scheuer"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx.ext.todo",
]
todo_include_todos = True
templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- napoleon: parse Google-style docstrings ---------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_include_init_with_doc = True

# -- autodoc / autosummary ----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"  # keeps signatures clean, types shown in the body
autosummary_generate = True  # build stub pages for every module in api/*.rst

# -- theme ---------------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/kevScheuer/PWAssist",
    "use_repository_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs/source",
}
