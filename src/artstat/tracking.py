from box import Box
import hashlib

def params_tostr(params):
    pairs = sorted(["%s=%s" % (k,v) for k,v in params.items()])
    return ",".join(pairs)

def hash_params(params):
    h = hashlib.sha1()
    h.update(params_tostr(params).encode())
    return h.hexdigest()
