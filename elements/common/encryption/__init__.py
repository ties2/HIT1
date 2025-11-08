"""
---------------------
**encryption** module
---------------------

This module contains the public encryption functionality.

>>> from common.elements.encryption import *

"""


from .cryptutils import (
    decrypt, encrypt,
    decrypt_file, encrypt_file,
    encrypt_files_folder,
    generate_key
)