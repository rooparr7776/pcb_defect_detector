"""Automatic image registration and homography alignment for PCB inspection.

Supports:
- ORB (Oriented FAST and Rotated BRIEF) - ultra-fast binary descriptor matching
- SIFT (Scale-Invariant Feature Transform) - robust to larger scale and rotation
- ECC (Enhanced Correlation Coefficient) - subpixel intensity-based fine alignment
"""

import cv2
import numpy as np


def simulate_misalignment(image, shift_x=12, shift_y=-8, angle_deg=2.5, scale=1.0):
    """Simulate real-world camera vibration or board misplacement."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, scale)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    warped = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return warped, matrix


def align_images_orb(test_bgr, template_bgr, max_features=2000, match_ratio=0.8, ransac_threshold=4.0):
    """Align test image to template using ORB feature detection and RANSAC homography."""
    h, w = template_bgr.shape[:2]
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY) if test_bgr.ndim == 3 else test_bgr
    temp_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY) if template_bgr.ndim == 3 else template_bgr

    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=12)
    kp_test, des_test = orb.detectAndCompute(test_gray, None)
    kp_temp, des_temp = orb.detectAndCompute(temp_gray, None)

    if des_test is None or des_temp is None or len(kp_test) < 4 or len(kp_temp) < 4:
        return test_bgr.copy(), np.eye(3), 0, [], kp_test, kp_temp

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_test, des_temp, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < match_ratio * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        # Fallback to cross-check matcher
        cc_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good_matches = cc_matcher.match(des_test, des_temp)

    if len(good_matches) < 4:
        return test_bgr.copy(), np.eye(3), 0, good_matches, kp_test, kp_temp

    src_pts = np.float32([kp_test[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_temp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    if homography is None:
        return test_bgr.copy(), np.eye(3), 0, good_matches, kp_test, kp_temp

    num_inliers = int(np.sum(inliers)) if inliers is not None else 0
    aligned = cv2.warpPerspective(test_bgr, homography, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return aligned, homography, num_inliers, good_matches, kp_test, kp_temp


def align_images_sift(test_bgr, template_bgr, max_features=1500, match_ratio=0.75, ransac_threshold=4.0):
    """Align test image to template using SIFT feature detection and RANSAC homography."""
    h, w = template_bgr.shape[:2]
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY) if test_bgr.ndim == 3 else test_bgr
    temp_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY) if template_bgr.ndim == 3 else template_bgr

    sift = cv2.SIFT_create(nfeatures=max_features)
    kp_test, des_test = sift.detectAndCompute(test_gray, None)
    kp_temp, des_temp = sift.detectAndCompute(temp_gray, None)

    if des_test is None or des_temp is None or len(kp_test) < 4 or len(kp_temp) < 4:
        return test_bgr.copy(), np.eye(3), 0, [], kp_test, kp_temp

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = matcher.knnMatch(des_test, des_temp, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < match_ratio * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        return test_bgr.copy(), np.eye(3), 0, good_matches, kp_test, kp_temp

    src_pts = np.float32([kp_test[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_temp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    if homography is None:
        return test_bgr.copy(), np.eye(3), 0, good_matches, kp_test, kp_temp

    num_inliers = int(np.sum(inliers)) if inliers is not None else 0
    aligned = cv2.warpPerspective(test_bgr, homography, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return aligned, homography, num_inliers, good_matches, kp_test, kp_temp


def align_images_ecc(test_bgr, template_bgr, warp_mode=cv2.MOTION_HOMOGRAPHY, max_iters=50, eps=1e-4):
    """Align test image to template using Enhanced Correlation Coefficient (ECC) maximization."""
    h, w = template_bgr.shape[:2]
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY) if test_bgr.ndim == 3 else test_bgr
    temp_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY) if template_bgr.ndim == 3 else template_bgr

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)

    try:
        _, warp_matrix = cv2.findTransformECC(
            temp_gray, test_gray, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5
        )
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(test_bgr, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
            homography = np.linalg.inv(warp_matrix)
        else:
            aligned = cv2.warpAffine(test_bgr, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
            homography = np.vstack([warp_matrix, [0, 0, 1]])
        return aligned, homography, 1, [], [], []
    except cv2.error:
        # If ECC fails to converge, fall back to ORB
        return align_images_orb(test_bgr, template_bgr)


def auto_register_pair(test_bgr, template_bgr, method="orb"):
    """Unified registration dispatcher."""
    method_lower = method.lower()
    if method_lower == "orb":
        return align_images_orb(test_bgr, template_bgr)
    elif method_lower == "sift":
        return align_images_sift(test_bgr, template_bgr)
    elif method_lower == "ecc":
        return align_images_ecc(test_bgr, template_bgr)
    else:
        # None / Identity
        return test_bgr.copy(), np.eye(3), 0, [], [], []


def draw_registration_matches(test_bgr, kp_test, template_bgr, kp_temp, matches, max_draw=40):
    """Draw top inlier matches between test and template images."""
    if not matches or not kp_test or not kp_temp:
        h = max(test_bgr.shape[0], template_bgr.shape[0])
        w = test_bgr.shape[1] + template_bgr.shape[1]
        combo = np.zeros((h, w, 3), dtype=np.uint8)
        combo[:test_bgr.shape[0], :test_bgr.shape[1]] = test_bgr
        combo[:template_bgr.shape[0], test_bgr.shape[1]:] = template_bgr
        return combo

    sorted_matches = sorted(matches, key=lambda m: m.distance)[:max_draw]
    draw_params = dict(
        matchColor=(0, 255, 128),
        singlePointColor=(255, 120, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    vis = cv2.drawMatches(test_bgr, kp_test, template_bgr, kp_temp, sorted_matches, None, **draw_params)
    return vis
