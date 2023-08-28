import numpy as np
from pytest import approx

from ellalgo.ldlt import ldlt_rank1update


def test_ldlt1():
    """[summary]"""
    n = 4
    low = np.eye(n)
    vec = np.ones(n)
    inv_diag = 0.5 * np.ones(n)
    old = np.diag(1 / inv_diag) + 4 * np.outer(vec, vec)
    ldlt_rank1update(low, inv_diag, 4, vec)
    new = low.dot(np.diag(1 / inv_diag)).dot(low.transpose())
    assert old == approx(new)

    vec = np.ones(n)
    old = old + 9 * np.outer(vec, vec)
    ldlt_rank1update(low, inv_diag, 9, vec)
    new = low.dot(np.diag(1 / inv_diag)).dot(low.transpose())
    assert old == approx(new)


# def test_chol2():
#     """[summary]"""
#     l2 = [
#         [18.0, 22.0, 54.0, 42.0],
#         [22.0, -70.0, 86.0, 62.0],
#         [54.0, 86.0, -174.0, 134.0],
#         [42.0, 62.0, 134.0, -106.0],
#     ]
#     m2 = np.array(l2)
#     ldlt_mgr = LDLTMgr(len(m2))
#     assert not ldlt_mgr.factorize(m2)
#     ldlt_mgr.witness()
#     assert ldlt_mgr.pos == (0, 2)
#     # assert ep == 1.0
#
#
# def test_chol3():
#     """[summary]"""
#     l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
#     m3 = np.array(l3)
#     ldlt_mgr = LDLTMgr(len(m3))
#     assert not ldlt_mgr.factorize(m3)
#     ep = ldlt_mgr.witness()
#     assert ldlt_mgr.pos == (0, 1)
#     assert ldlt_mgr.v[0] == 1.0
#     assert ep == 0.0
#
#
# def test_chol4():
#     """[summary]"""
#     l1 = [[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
#     m1 = np.array(l1)
#     Q1 = LDLTMgr(len(m1))
#     Q1.allow_semidefinite = True
#     assert Q1.factorize(m1)
#
#
# def test_chol5():
#     """[summary]"""
#     l2 = [
#         [18.0, 22.0, 54.0, 42.0],
#         [22.0, -70.0, 86.0, 62.0],
#         [54.0, 86.0, -174.0, 134.0],
#         [42.0, 62.0, 134.0, -106.0],
#     ]
#     m2 = np.array(l2)
#     ldlt_mgr = LDLTMgr(len(m2))
#     ldlt_mgr.allow_semidefinite = True
#     assert not ldlt_mgr.factorize(m2)
#     ldlt_mgr.witness()
#     assert ldlt_mgr.pos == (0, 2)
#     # assert ep == 1.0
#
#
# def test_chol6():
#     """[summary]"""
#     l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
#     m3 = np.array(l3)
#     ldlt_mgr = LDLTMgr(len(m3))
#     ldlt_mgr.allow_semidefinite = True
#     assert ldlt_mgr.factorize(m3)
#
#
# #     [v, ep] = ldlt_mgr.witness2()
# #     assert len(v) == 1
# #     assert v[0] == 1.0
# #     assert ep == 0.0
#
#
# def test_chol7():
#     """[summary]"""
#     l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, -20.0]]
#     m3 = np.array(l3)
#     ldlt_mgr = LDLTMgr(len(m3))
#     ldlt_mgr.allow_semidefinite = True
#     assert not ldlt_mgr.factorize(m3)
#     ep = ldlt_mgr.witness()
#     assert ep == 20.0
#
#
# def test_chol8():
#     """[summary]"""
#     """[summary]
#     """
#     l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 20.0]]
#     m3 = np.array(l3)
#     ldlt_mgr = LDLTMgr(len(m3))
#     ldlt_mgr.allow_semidefinite = False
#     assert not ldlt_mgr.factorize(m3)
#
#
# def test_chol9():
#     """[summary]"""
#     """[summary]
#     """
#     l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 20.0]]
#     m3 = np.array(l3)
#     ldlt_mgr = LDLTMgr(len(m3))
