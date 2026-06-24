import cv2
import numpy as np

# 1. ARUCO SETUP (Matches your 7x7 marker)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
parameters = cv2.aruco.DetectorParameters()

# 2. CAMERA CALIBRATION (Standard Laptop HD)
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((4,1))

# 3. DEFINE ACRIMSAT GEOMETRY (Hexagonal Body + 4 Tilted Wings)
s = 0.05 
# Hexagonal Body (6 points bottom, 6 points top)
angles = np.linspace(0, 2*np.pi, 7)[:-1]
body_bot = np.array([[np.cos(a)*s, np.sin(a)*s, 0] for a in angles])
body_top = np.array([[np.cos(a)*s, np.sin(a)*s, -s*1.8] for a in angles])

# 4 Tilted Solar Panels
panel_w, panel_h = s*1.6, s*1.4
tilt = np.deg2rad(35)
panels = []
for i in range(4):
    angle = i * np.pi/2
    p = np.array([[s, -panel_w/2, -s], [s+panel_h, -panel_w/2, -s-panel_h*np.sin(tilt)], 
                  [s+panel_h, panel_w/2, -s-panel_h*np.sin(tilt)], [s, panel_w/2, -s]])
    c, sh = np.cos(angle), np.sin(angle)
    rot_m = np.array([[c, -sh, 0], [sh, c, 0], [0, 0, 1]])
    panels.append(p @ rot_m.T)

cap = cv2.VideoCapture(0)
print("🛰️ NASA ACRIMSAT Tracker Live. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Estimate 3D Pose
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            # Project Body & Panels to 2D
            img_bot, _ = cv2.projectPoints(body_bot, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            img_top, _ = cv2.projectPoints(body_top, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            img_bot, img_top = np.int32(img_bot).reshape(-1, 2), np.int32(img_top).reshape(-1, 2)

            # Draw Yellow Hexagonal Body (Thick lines for quality)
            cv2.drawContours(frame, [img_bot], -1, (0, 255, 255), 4)
            cv2.drawContours(frame, [img_top], -1, (0, 255, 255), 4)
            for j in range(6): cv2.line(frame, tuple(img_bot[j]), tuple(img_top[j]), (0, 255, 255), 4)

            # Draw Blue Tilted Panels
            for p_pts in panels:
                img_p, _ = cv2.projectPoints(p_pts, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                cv2.drawContours(frame, [np.int32(img_p).reshape(-1, 2)], -1, (255, 100, 0), 3)

            # HUD Display
            dist = np.linalg.norm(tvecs[i]) * 100
            cv2.putText(frame, f"ACRIMSAT LOCKED | DIST: {dist:.1f}cm", (40, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("NASA ACRIMSAT Project", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

