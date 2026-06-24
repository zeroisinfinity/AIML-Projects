import cv2
import cv2.aruco as aruco
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import trimesh  # Make sure to pip install trimesh

# --- CONSTANTS & SETUP ---
MARKER_SIZE = 0.05  # 5cm (Measure your phone screen width!)
MODEL_SCALE = 0.002  # CRITICAL: Shrinks the model to fit the marker. Adjust this if it's too small/big.
FORCE_7X7 = True  # We detected this in your logs

# Load your mesh using trimesh
try:
    mesh = trimesh.load("assets/asset.glb")  # Update path
    # If it's a scene, merge it
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
except:
    print("⚠️ Model not found, using fallback cube.")
    mesh = None


def draw_mesh():
    """ Renders the loaded GLB model or a fallback cube """
    if mesh:
        glEnable(GL_BLEND)  # Enable transparency if needed
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Render the mesh vertices
        # Note: For production, use Vertex Arrays (VBOs). This is 'immediate mode' for simplicity.
        glBegin(GL_TRIANGLES)
        glColor3f(1.0, 1.0, 1.0)  # White tint
        for face in mesh.faces:
            for vertex_i in face:
                v = mesh.vertices[vertex_i]
                glVertex3f(v[0], v[1], v[2])
        glEnd()
    else:
        # Fallback Cube
        glutSolidCube(0.05)


# --- CAMERA MATH ---
def get_camera_matrix(w, h):
    # Approximate focal length if uncalibrated
    fov = 60
    f = w / (2 * np.tan(np.radians(fov / 2)))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32), np.zeros(5)


# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
w, h = 640, 480
cap.set(3, w);
cap.set(4, h)

# Pygame / OpenGL Init
pygame.init()
pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)

# Setup Perspective
glViewport(0, 0, w, h)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(60, (w / h), 0.1, 100.0)  # FOV must match camera estimation
glMatrixMode(GL_MODELVIEW)

# ArUco Setup (LOCKED TO 7x7 BASED ON YOUR LOGS)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

mtx, dist = get_camera_matrix(w, h)

# 3D Points of the marker corners
marker_pts = np.array([[-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                       [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                       [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                       [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]], dtype=np.float32)

print("🚀 AR Started. Look at the marker.")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: pygame.quit(); cap.release(); exit()

    ret, frame = cap.read()
    if not ret: break

    # 1. Detect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    # 2. Render Background (The Video Feed)
    bg_img = cv2.flip(frame, 0)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_data = bg_img.tobytes()

    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1, 1)
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity()
    glRasterPos2i(0, 0)
    glDrawPixels(w, h, GL_RGB, GL_UNSIGNED_BYTE, bg_data)
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW)

    # 3. Render 3D Object
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)  # Clear depth so object sits on top

    if ids is not None:
        # Solve Pose
        ret, rvec, tvec = cv2.solvePnP(marker_pts, corners[0], mtx, dist)

        if ret:
            # Build Matrix
            R, _ = cv2.Rodrigues(rvec)
            view_matrix = np.eye(4)
            view_matrix[:3, :3] = R
            view_matrix[:3, 3] = tvec.squeeze()

            # Flip Coordinate System (OpenCV -> OpenGL)
            cv_to_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            view_matrix = cv_to_gl @ view_matrix

            glLoadMatrixd(view_matrix.T)

            # --- CRITICAL FIX FOR BLACK BARS ---
            # Scale the model down drastically
            glScalef(MODEL_SCALE, MODEL_SCALE, MODEL_SCALE)

            # Rotate to stand upright if needed (Common for GLB files)
            glRotatef(90, 1, 0, 0)

            draw_mesh()

    pygame.display.flip()
