import os

from cryptography.fernet import Fernet
from glob import glob


def encrypt(message: bytes, key: bytes) -> bytes:
    return Fernet(key).encrypt(message)


def decrypt(token: bytes, key: bytes) -> bytes:
    return Fernet(key).decrypt(token)


def encrypt_file(filename, key):
    if not os.path.isfile(filename):
        raise RuntimeError("File not found {filename}")
    with open(filename, "rb") as f:
        content = f.read()
    return encrypt(content, key)


def decrypt_file(filename, key):
    if not os.path.isfile(filename):
        raise RuntimeError("File not found {filename}")
    with open(filename, "rb") as f:
        content = f.read()
    return decrypt(content, key)


def generate_key() -> bytes:
    return Fernet.generate_key().decode()


def encrypt_files_folder(input_folder: str, output_folder: str, key: bytes, file_filter: str = "*.xml"):
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    i = 0
    for file in glob(os.path.join(input_folder, file_filter)):
        content = encrypt_file(file, key)
        out_filename = os.path.basename(file) + ".crypt"
        out_file = os.path.join(output_folder, out_filename)
        with open(out_file, "wb") as f_out:
            f_out.write(content)
        if not os.path.isfile(out_file):
            raise RuntimeError(f"Failed writing file {out_file}")
        i = i + 1
    print(f"Encrypted {i} files")

