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
MARKER_SIZE = 0.05  # 5cm
DICT_TO_USE = cv2.aruco.DICT_7X7_50
MODEL_PATH = "assets/asset.glb"
MODEL_SCALE = 0.002
SMOOTHING = 0.5

# ==========================================
# 2. GPU LOADER (WITH TEXTURE FIX)
# ==========================================
model_list_id = None


def load_model_to_gpu():
    """ Loads model and bakes textures into vertex colors for OpenGL """
    global model_list_id
    print("⏳ Loading Model & Baking Colors...")

    try:
        mesh = trimesh.load(MODEL_PATH)

        # 1. Merge Scene
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        # 2. CRITICAL: Convert Textures to Vertex Colors
        # This takes the image wrap (Gold/Solar Panels) and paints it onto the geometry
        if hasattr(mesh.visual, 'to_color'):
            print("   → Baking textures to colors (this might take 2 seconds)...")
            mesh.visual = mesh.visual.to_color()

    except Exception as e:
        print(f"⚠️ Model Error: {e}")
        mesh = None

    # Compile to GPU
    model_list_id = glGenLists(1)
    glNewList(model_list_id, GL_COMPILE)

    if mesh:
        glEnable(GL_NORMALIZE)
        glBegin(GL_TRIANGLES)

        # Get colors from the visual object
        # (trimesh stores these as 0-255 RGBA bytes)
        colors = mesh.visual.vertex_colors
        vertices = mesh.vertices
        normals = mesh.vertex_normals
        faces = mesh.faces

        for face in faces:
            # For every corner of the triangle
            for vertex_i in face:
                # Apply Color (Convert 0-255 to 0.0-1.0)
                c = colors[vertex_i]
                glColor3f(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)

                # Apply Normal
                n = normals[vertex_i]
                glNormal3f(n[0], n[1], n[2])

                # Draw Vertex
                v = vertices[vertex_i]
                glVertex3f(v[0], v[1], v[2])
        glEnd()
    else:
        # Fallback Cube
        glColor3f(1, 0, 0)
        glutSolidCube(0.05)

    glEndList()
    print(f"✅ Model Ready (ID: {model_list_id})")


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
                self.rvec = self.rvec * (1 - self.alpha) + rvec * self.alpha
                self.tvec = self.tvec * (1 - self.alpha) + tvec * self.alpha
            return True, self.rvec, self.tvec
        return False, None, None


# ==========================================
# 4. MAIN LOOP
# ==========================================
def init_gl(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity()
    gluPerspective(60, (w / h), 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW)

    # Lighting Setup for "Gold" appearance
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST)

    # Light coming from top-right to highlight edges
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 5.0, 5.0, 1.0))
    # Slightly warm light to enhance gold
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.9, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))
    # Make the material shiny (Metallic effect)
    glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))  # White reflection
    glMaterialf(GL_FRONT, GL_SHININESS, 100.0)  # 0=Dull, 128=Mirror


def main():
    cap = cv2.VideoCapture(0)
    w, h = 640, 480
    cap.set(3, w);
    cap.set(4, h)

    pygame.init()
    pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)
    init_gl(w, h)

    load_model_to_gpu()

    aruco_dict = aruco.getPredefinedDictionary(DICT_TO_USE)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    marker_pts = np.array([
        [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0], [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
        [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0], [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
    ], dtype=np.float32)

    f = w / (2 * np.tan(np.radians(60 / 2)))
    mtx = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5)

    filter = PoseFilter(alpha=SMOOTHING)
    clock = pygame.time.Clock()

    print("🚀 AR Running. Press 'q' to quit.")

    while True:
        for event in pygame.event.get():
            if event.type == QUIT: pygame.quit(); return

        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        found = False
        rvec_raw, tvec_raw = None, None

        if ids is not None:
            ret_pnp, rvec_raw, tvec_raw = cv2.solvePnP(marker_pts, corners[0], mtx, dist)
            found = ret_pnp

        should_render, rvec, tvec = filter.update(rvec_raw, tvec_raw, found)

        # Background
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
        glDrawPixels(w, h, GL_BGR, GL_UNSIGNED_BYTE, bg_data)
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW)

        # Model
        if should_render:
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_LIGHTING);
            glClear(GL_DEPTH_BUFFER_BIT)

            R, _ = cv2.Rodrigues(rvec)
            view_matrix = np.eye(4)
            view_matrix[:3, :3] = R
            view_matrix[:3, 3] = tvec.squeeze()
            cv_to_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            view_matrix = cv_to_gl @ view_matrix

            glLoadMatrixd(view_matrix.T)
            glScalef(MODEL_SCALE, MODEL_SCALE, MODEL_SCALE)
            glRotatef(90, 1, 0, 0)

            glCallList(model_list_id)

        pygame.display.flip()
        clock.tick(60)

    cap.release()


if __name__ == "__main__":
    main()
