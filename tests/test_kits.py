import numpy as np

from mtah.kits import recover_keeper, slot_labels, torso_features

PITCH = (105.0, 68.0)


def test_torso_features_ignore_grass():
    img = np.zeros((200, 100, 3), np.uint8)
    img[:] = (40, 140, 40)                         # grass green (BGR)
    img[60:110, 40:60] = (30, 30, 200)             # a red shirt in the torso band
    f = torso_features(img, x=50, y=100, w=40, h=200)
    assert f is not None
    import cv2
    red_lab = cv2.cvtColor(np.uint8([[[30, 30, 200]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    assert np.allclose(f[:3], red_lab, atol=3)


def test_torso_features_none_when_only_grass():
    img = np.zeros((200, 100, 3), np.uint8)
    img[:] = (40, 140, 40)
    assert torso_features(img, x=50, y=100, w=40, h=200) is None


def test_slot_labels_need_one_clear_slot():
    px = np.array([[10.0, 10.8, 50.0, 90.0]])      # slots 0 and 1 are 0.8 m apart
    py = np.array([[30.0, 30.0, 30.0, 30.0]])
    team = np.array([0, 1, 0, 1], np.int8)
    boxes = np.array([[10.1, 30.0], [50.2, 30.1], [90.0, 31.5]])
    lab = slot_labels(boxes, np.repeat(px, 3, 0), np.repeat(py, 3, 0), team, {0: 2, 1: 3})
    assert lab.tolist() == [-1, 2, -1]             # ambiguous; keeper slot 2 -> class 2; 1.5 m away -> none


def test_recover_keeper_ignores_off_pitch_and_wrong_half():
    X = np.array([3.0, -2.0, 80.0, 10.0])
    Y = np.array([34.0, 34.0, 34.0, 0.5])
    p = np.array([0.6, 0.99, 0.99, 0.99])
    # box 1 is behind the goal line, box 2 in the other half, box 3 on the touchline strip
    assert recover_keeper(X, Y, p, own_half_low_x=True, pitch=PITCH) == 0
    assert recover_keeper(X[1:], Y[1:], p[1:], own_half_low_x=True, pitch=PITCH) == -1
