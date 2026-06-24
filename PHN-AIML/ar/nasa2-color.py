import cv2
import cv2.aruco as aruco
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import trimesh

# ==========================================
# 1. CONFIGURATION
# ==========================================
# MARKER SETUP
MARKER_SIZE = 0.05  # 5cm (The physical size of the marker on your phone)
DICT_TO_USE = cv2.aruco.DICT_7X7_50  # Detected from your logs

# MODEL SETUP
MODEL_PATH = "assets/asset.glb"
MODEL_SCALE = 0.002  # CRITICAL: Shrinks the 1.0m model to fit the 5cm marker.
# If too small, change to 0.01. If too big, 0.001.

# ==========================================
# 2. MESH LOADER (With Color & Normals)
# ==========================================
try:
    mesh = trimesh.load(MODEL_PATH)
    if isinstance(mesh, trimesh.Scene):
        # Concatenate all parts of the scene into one mesh
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    print(f"✅ Model Loaded: {len(mesh.faces)} faces")
except Exception as e:
    print(f"⚠️ Model Load Failed: {e}")
    print("   (Using Fallback Cube)")
    mesh = None


def draw_mesh():
    """ Renders the mesh with Lighting and Color """
    if mesh:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glBegin(GL_TRIANGLES)

        # Check if the model has baked colors
        has_color = hasattr(mesh.visual, 'face_colors') and len(mesh.visual.face_colors) > 0

        for i, face in enumerate(mesh.faces):
            # 1. Apply Color
            if has_color:
                c = mesh.visual.face_colors[i]
                # Trimesh colors are 0-255, OpenGL wants 0.0-1.0
                glColor3f(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            else:
                # Default "Tech Cyan" if no color found
                glColor3f(0.0, 0.8, 0.8)

                # 2. Apply Normal (Critical for Lighting!)
            # The normal tells the light how to bounce off this face
            n = mesh.face_normals[i]
            glNormal3f(n[0], n[1], n[2])

            # 3. Draw Vertices
            for vertex_i in face:
                v = mesh.vertices[vertex_i]
                glVertex3f(v[0], v[1], v[2])

        glEnd()
    else:
        # Fallback Red Cube
        glDisable(GL_LIGHTING)  # Simple color for fallback
        glColor3f(1.0, 0.0, 0.0)
        glutSolidCube(0.05)


# ==========================================
# 3. OPENGL & CAMERA SETUP
# ==========================================
def init_gl(w, h):
    # Viewport
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, (w / h), 0.01, 100.0)  # FOV 60 matches our camera est.
    glMatrixMode(GL_MODELVIEW)

    # --- LIGHTING ENGINE ---
    glEnable(GL_LIGHTING)  # Master Switch
    glEnable(GL_LIGHT0)  # Sun
    glEnable(GL_COLOR_MATERIAL)  # Allow object color to show through light
    glEnable(GL_NORMALIZE)  # Fix lighting scaling issues

    # Light Position (x, y, z, w) - Up and to the right
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 5.0, 10.0, 1.0))
    # Light Color
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))  # White light
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))  # Soft shadow fill


def get_camera_matrix(w, h):
    # Approximate intrinsics (Focal length ~ Width for 60 deg FOV)
    f = w / (2 * np.tan(np.radians(60 / 2)))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32), np.zeros(5)


# ==========================================
# 4. MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(0)
w, h = 640, 480
cap.set(3, w);
cap.set(4, h)

pygame.init()
pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)
init_gl(w, h)

# ArUco
aruco_dict = aruco.getPredefinedDictionary(DICT_TO_USE)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

mtx, dist = get_camera_matrix(w, h)

# Marker Corners in 3D Space (Centered at 0,0,0)
marker_pts = np.array([
    [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

print("🚀 Running AR... Press 'q' on the window to quit.")

clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: pygame.quit(); cap.release(); exit()

    ret, frame = cap.read()
    if not ret: break

    # 1. Detect Markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    # 2. Render Video Background
    # We draw the webcam feed as a flat 2D image behind everything
    bg_img = cv2.flip(frame, 0)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_data = bg_img.tobytes()

    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)  # Lighting off for background
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

    # 3. Render 3D Model
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)  # Clear depth buffer so 3D sits on top of video

    if ids is not None:
        # Calculate Pose
        ret, rvec, tvec = cv2.solvePnP(marker_pts, corners[0], mtx, dist)

        if ret:
            # --- MATRIX MAGIC ---
            # 1. Get Rotation Matrix
            R, _ = cv2.Rodrigues(rvec)

            # 2. Build View Matrix
            view_matrix = np.eye(4)
            view_matrix[:3, :3] = R
            view_matrix[:3, 3] = tvec.squeeze()

            # 3. Convert OpenCV (Y-Down) to OpenGL (Y-Up)
            cv_to_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            view_matrix = cv_to_gl @ view_matrix

            # 4. Load into OpenGL (Transpose needed for memory layout)
            glLoadMatrixd(view_matrix.T)

            # --- TRANSFORMATIONS ---
            # Scale: Fixes the "Black Bars" issue (Model too big)
            glScalef(MODEL_SCALE, MODEL_SCALE, MODEL_SCALE)

            # Rotate: Fixes orientation (GLB often comes in lying down)
            glRotatef(90, 1, 0, 0)

            draw_mesh()

    pygame.display.flip()
    clock.tick(30)

