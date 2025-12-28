library(reticulate)

# Pick a brand-new env name (don???t reuse the old one)
env <- "C:/Users/user/Documents/.virtualenvs/r-tf-clean"

# Make sure Python won???t read your user site-packages (this often leaks NumPy 2.x)
Sys.setenv(PYTHONNOUSERSITE = "1")

# Create a fresh venv
virtualenv_create(env)

# Point reticulate to it (BEFORE anything else imports Python)
Sys.setenv(RETICULATE_PYTHON = file.path(env, "Scripts", "python.exe"))
use_virtualenv(env, required = TRUE)

py <- py_config()$python
pip <- c(py, "-m", "pip")

# Fresh pip toolchain
system2(pip[1], c(pip[-1], "install", "--upgrade", "pip", "setuptools", "wheel"))
# Uninstall until it says "WARNING: Skipping numpy as it is not installed."
repeat {
  out <- system2(pip[1], c(pip[-1], "uninstall", "-y", "numpy"), stdout = TRUE)
  if (any(grepl("Skipping numpy", out))) break
}

# Install the exact versions
system2(pip[1], c(pip[-1], "install", "numpy==1.26.4", "ml-dtypes==0.3.2"))
py_run_string("
import sys, numpy, site, pkgutil
print('[PY]', sys.executable)
print('[NUMPY]', numpy.__version__, '->', numpy.__file__)
print('[USER_SITE_ENABLED]', getattr(site, 'ENABLE_USER_SITE', 'n/a'))
print('[PATH_HEAD]', sys.path[:5])
")
system2(pip[1], c(pip[-1], "install", "tensorflow==2.17.1"))
py_run_string("
import tensorflow as tf, numpy as np
print('[OK] TF', tf.__version__, '| NumPy', np.__version__)
")
