import os

import hashlib
from sklearn.preprocessing import LabelEncoder


def md5_hash(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

    
def encode_categoricals(X, columns):
    for col in columns:
        X[col] = LabelEncoder.fit_transform(X[col].astype(str), y=X[col].astype(str))
    return X


def get_filepaths(dirpath='.', extensions=['.py','.yaml','yml']):
    files=[]
    for r, d, f in os.walk(dirpath):
        for file in f:
            if any(file.endswith(ext) for ext in extensions):
                files.append(os.path.join(r,file))
    return files
        
