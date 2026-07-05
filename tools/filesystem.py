# Filesystem tool for CriderGPT Engine
import os

def list_files(directory):
    """Lists files in the specified directory."""
    try:
        return os.listdir(directory)
    except Exception as e:
        return str(e)
