import numpy as np

from ellalgo.oracles.ldlt_mgr import LDLTMgr


def test_chol1():
    l1 = [[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
    m1 = np.array(l1)
    ldlt_mgr = LDLTMgr(len(m1))
    assert ldlt_mgr.factorize(m1)


def test_chol2():
    l2 = [
        [18.0, 22.0, 54.0, 42.0],
        [22.0, -70.0, 86.0, 62.0],
        [54.0, 86.0, -174.0, 134.0],
        [42.0, 62.0, 134.0, -106.0],
    ]
    m2 = np.array(l2)
    ldlt_mgr = LDLTMgr(len(m2))
    assert not ldlt_mgr.factorize(m2)
    ldlt_mgr.witness()
    assert ldlt_mgr.pos == (0, 2)
    # assert ep == 1.0


def test_chol3():
    l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
    m3 = np.array(l3)
    ldlt_mgr = LDLTMgr(len(m3))
    assert not ldlt_mgr.factorize(m3)
    ep = ldlt_mgr.witness()
    assert ldlt_mgr.pos == (0, 1)
    assert ldlt_mgr.wit[0] == 1.0
    assert ep == 0.0


def test_chol4():
    l1 = [[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
    m1 = np.array(l1)
    Q1 = LDLTMgr(len(m1))
    # Q1.allow_semidefinite = True
    assert Q1.factorize(m1)


def test_chol5():
    l2 = [
        [18.0, 22.0, 54.0, 42.0],
        [22.0, -70.0, 86.0, 62.0],
        [54.0, 86.0, -174.0, 134.0],
        [42.0, 62.0, 134.0, -106.0],
    ]
    m2 = np.array(l2)
    ldlt_mgr = LDLTMgr(len(m2))
    # ldlt_mgr.allow_semidefinite = True
    assert not ldlt_mgr.factorize(m2)
    ldlt_mgr.witness()
    assert ldlt_mgr.pos == (0, 2)
    # assert ep == 1.0


def test_chol6():
    l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
    m3 = np.array(l3)
    ldlt_mgr = LDLTMgr(len(m3))
    # ldlt_mgr.allow_semidefinite = True
    assert ldlt_mgr.factor_with_allow_semidefinite(lambda i, j: m3[i, j])


def test_chol7():
    l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, -20.0]]
    m3 = np.array(l3)
    ldlt_mgr = LDLTMgr(len(m3))
    # ldlt_mgr.allow_semidefinite = True
    # assert not ldlt_mgr.factorize(m3)
    assert not ldlt_mgr.factor_with_allow_semidefinite(lambda i, j: m3[i, j])
    ep = ldlt_mgr.witness()
    assert ep == 20.0


def test_chol8():
    l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 20.0]]
    m3 = np.array(l3)
    ldlt_mgr = LDLTMgr(len(m3))
    # ldlt_mgr.allow_semidefinite = False
    assert not ldlt_mgr.factorize(m3)


def test_chol9():
    l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 20.0]]
    m3 = np.array(l3)
    ldlt_mgr = LDLTMgr(len(m3))
    # ldlt_mgr.allow_semidefinite = True
    # assert ldlt_mgr.factorize(m3)
    assert ldlt_mgr.factor_with_allow_semidefinite(lambda i, j: m3[i, j])


def test_ldlt_mgr_sqrt():
    A = np.array([[1.0, 0.5, 0.5], [0.5, 1.25, 0.75], [0.5, 0.75, 1.5]])
    ldlt_mgr = LDLTMgr(3)
    ldlt_mgr.factor(lambda i, j: A[i, j])
    assert ldlt_mgr.is_spd()
    R = ldlt_mgr.sqrt()
    assert np.allclose(R, np.array([[1.0, 0.5, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]))
