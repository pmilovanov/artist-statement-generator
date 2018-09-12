from artstat import tracking as t

def test_params_tostr():
    a = {'0a': 'oo', 'aa': 9, 'p0': 45, 'p6': 'hello'}
    assert '0a=oo,aa=9,p0=45,p6=hello' == t.params_tostr(a)

