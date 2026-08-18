import numpy as np
from scipy.spatial.transform import Rotation

def decode_fp3_row_rot6d(v):
    v = np.asarray(v, dtype=np.float64).reshape(6)
    r0 = v[:3]
    r1 = v[3:]
    r0 = r0 / np.linalg.norm(r0)
    r1 = r1 - np.dot(r0, r1) * r0
    r1 = r1 / np.linalg.norm(r1)
    r2 = np.cross(r0, r1)
    return np.stack((r0, r1, r2), axis=0)

def test_row_rot6d_roundtrip():
    original = Rotation.from_euler("XYZ", [0.35, -0.4, 1.2]).as_matrix()
    v = original[:2, :].reshape(6)
    recovered = decode_fp3_row_rot6d(v)
    assert np.allclose(original, recovered, atol=1e-7)
