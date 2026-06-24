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
a = "assets/asset.glb"
b = 'assets/Advanced Crew Escape Suit.glb'
MARKER_SIZE = 0.05  # 5cm (Matches your phone screen)
DICT_TO_USE = cv2.aruco.DICT_7X7_50
MODEL_PATH = b
MODEL_SCALE = 0.002  # Scale adjustment
SMOOTHING = 0.5  # 0.5 = Balanced smoothing

# ==========================================
# 2. GPU ACCELERATED LOADER
# ==========================================
model_list_id = None  # Pointer to GPU memory


def load_model_to_gpu():
    """ Compiles the mesh into a Display List for instant rendering. """
    global model_list_id
    print("⏳ Loading Model to GPU...")

    try:
        mesh = trimesh.load(MODEL_PATH)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    except:
        print("⚠️ Model not found, using Cube.")
        mesh = None

    # Create a Display List (Compiles commands to GPU)
    model_list_id = glGenLists(1)
    glNewList(model_list_id, GL_COMPILE)

    if mesh:
        # Optimization: Normalize normals for correct lighting scaling
        glEnable(GL_NORMALIZE)
        glBegin(GL_TRIANGLES)

        # Check for colors
        has_color = hasattr(mesh.visual, 'face_colors') and len(mesh.visual.face_colors) > 0

        for i, face in enumerate(mesh.faces):
            # 1. Color
            if has_color:
                c = mesh.visual.face_colors[i]
                glColor3f(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            else:
                glColor3f(0.0, 0.8, 0.9)  # Default Cyan

            # 2. Normal (Crucial for Lighting shadows)
            n = mesh.face_normals[i]
            glNormal3f(n[0], n[1], n[2])

            # 3. Vertices
            for vertex_i in face:
                v = mesh.vertices[vertex_i]
                glVertex3f(v[0], v[1], v[2])
        glEnd()
    else:
        # Fallback Cube
        glColor3f(1, 0, 0)
        glutSolidCube(0.05)

    glEndList()
    print(f"✅ Model compiled to GPU (ID: {model_list_id})")


# ==========================================
# 3. TRACKING STABILIZER
# ==========================================
class PoseFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.rvec = None
        self.tvec = None

    def update(self, rvec, tvec, found):
        if found:
            if self.rvec is None:
                self.rvec, self.tvec = rvec, tvec
            else:
                # Linear Interpolation to remove jitter
                self.rvec = self.rvec * (1 - self.alpha) + rvec * self.alpha
                self.tvec = self.tvec * (1 - self.alpha) + tvec * self.alpha
            return True, self.rvec, self.tvec
        return False, None, None


# ==========================================
# 4. MAIN SYSTEM
# ==========================================
def init_gl(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity()
    gluPerspective(60, (w / h), 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW)

    # High Quality Lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST)
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 10.0, 10.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.2, 1.2, 1.2, 1.0))


def main():
    cap = cv2.VideoCapture(0)
    w, h = 640, 480
    cap.set(3, w);
    cap.set(4, h)

    pygame.init()
    pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)
    init_gl(w, h)

    # --- STEP 1: LOAD TO GPU ONCE ---
    load_model_to_gpu()

    # Setup ArUco
    aruco_dict = aruco.getPredefinedDictionary(DICT_TO_USE)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    marker_pts = np.array([
        [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0], [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
        [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0], [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
    ], dtype=np.float32)

    # Camera Intrinsics
    f = w / (2 * np.tan(np.radians(60 / 2)))
    mtx = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5)

    filter = PoseFilter(alpha=SMOOTHING)
    clock = pygame.time.Clock()

    print("🚀 AR Running Smoothly. Press 'q' to quit.")

    while True:
        # Event Handling (Prevents "Not Responding")
        for event in pygame.event.get():
            if event.type == QUIT: pygame.quit(); return

        ret, frame = cap.read()
        if not ret: break

        # Detect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        found = False
        rvec_raw, tvec_raw = None, None

        if ids is not None:
            ret_pnp, rvec_raw, tvec_raw = cv2.solvePnP(marker_pts, corners[0], mtx, dist)
            found = ret_pnp

        # Smooth
        should_render, rvec, tvec = filter.update(rvec_raw, tvec_raw, found)

        # --- RENDER ---
        # 1. Background Video
        bg_data = cv2.flip(frame, 0).tobytes()
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, w, 0, h, -1, 1)
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity()
        glRasterPos2i(0, 0)
        # GL_BGR matches OpenCV format naturally
        glDrawPixels(w, h, GL_BGR, GL_UNSIGNED_BYTE, bg_data)
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW)

        # 2. 3D Model
        if should_render:
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_LIGHTING);
            glClear(GL_DEPTH_BUFFER_BIT)

            R, _ = cv2.Rodrigues(rvec)
            view_matrix = np.eye(4)
            view_matrix[:3, :3] = R
            view_matrix[:3, 3] = tvec.squeeze()

            # Convert Coords
            cv_to_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            view_matrix = cv_to_gl @ view_matrix

            glLoadMatrixd(view_matrix.T)

            # Transforms
            glScalef(MODEL_SCALE, MODEL_SCALE, MODEL_SCALE)
            glRotatef(90, 1, 0, 0)

            # --- FAST RENDER CALL ---
            # Draws the entire model in 1 command
            glCallList(model_list_id)

        pygame.display.flip()
        clock.tick(60)  # Lock to 60 FPS

    cap.release()


if __name__ == "__main__":
    main()

# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 nasa-2-gpu.py