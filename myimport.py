
from deslib.base import BaseDS
from sklearn.utils.validation import validate_data

# TODO: update deslib.BaseDS
def _validate_data(obj, *aaa, **bbb):
    return validate_data(obj, *aaa, **bbb)

BaseDS._validate_data = _validate_data
