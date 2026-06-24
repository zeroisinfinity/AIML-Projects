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
# 2. GPU LOADER (TEXTURE FIX)
# ==========================================
model_list_id = None


def load_model_to_gpu():
    """ Loads model and bakes textures into vertex colors """
    global model_list_id
    print("⏳ Loading Model & Baking Colors...")

    try:
        mesh = trimesh.load(MODEL_PATH)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        if hasattr(mesh.visual, 'to_color'):
            print("   → Baking textures...")
            mesh.visual = mesh.visual.to_color()

    except Exception as e:
        print(f"⚠️ Model Error: {e}")
        mesh = None

    model_list_id = glGenLists(1)
    glNewList(model_list_id, GL_COMPILE)

    if mesh:
        glEnable(GL_NORMALIZE)
        glBegin(GL_TRIANGLES)
        colors = mesh.visual.vertex_colors
        vertices = mesh.vertices
        normals = mesh.vertex_normals
        faces = mesh.faces

        for face in faces:
            for vertex_i in face:
                c = colors[vertex_i]
                glColor3f(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
                n = normals[vertex_i]
                glNormal3f(n[0], n[1], n[2])
                v = vertices[vertex_i]
                glVertex3f(v[0], v[1], v[2])
        glEnd()
    else:
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

    # Lighting for Gold/Metal Look
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST)
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 5.0, 5.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.9, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))

    # Shiny Specular Highlights
    glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
    glMaterialf(GL_FRONT, GL_SHININESS, 100.0)


def main():
    # --- FULL SCREEN SETUP ---
    pygame.init()

    # Get Monitor Resolution
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h

    # Create Full Screen Window
    pygame.display.set_mode((screen_w, screen_h), FULLSCREEN | DOUBLEBUF | OPENGL)

    # Setup Camera (Request HD 1280x720 for better quality on big screen)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    init_gl(screen_w, screen_h)
    load_model_to_gpu()

    # ArUco
    aruco_dict = aruco.getPredefinedDictionary(DICT_TO_USE)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    marker_pts = np.array([
        [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0], [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
        [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0], [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
    ], dtype=np.float32)

    # Recalculate Focal Length for the Screen Width
    f = screen_w / (2 * np.tan(np.radians(60 / 2)))
    mtx = np.array([[f, 0, screen_w / 2], [0, f, screen_h / 2], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5)

    filter = PoseFilter(alpha=SMOOTHING)
    clock = pygame.time.Clock()

    print("🚀 Full Screen AR. Press 'ESC' or 'q' to quit.")

    while True:
        # Event Handling
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit();
                return
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_q:
                    pygame.quit();
                    return

        ret, frame = cap.read()
        if not ret: break

        # Resize webcam frame to match screen (Stretched)
        # Note: For perfect aspect ratio, we'd crop, but stretching fills the screen.
        frame_resized = cv2.resize(frame, (screen_w, screen_h))

        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        found = False
        rvec_raw, tvec_raw = None, None

        if ids is not None:
            ret_pnp, rvec_raw, tvec_raw = cv2.solvePnP(marker_pts, corners[0], mtx, dist)
            found = ret_pnp

        should_render, rvec, tvec = filter.update(rvec_raw, tvec_raw, found)

        # --- RENDER ---
        # 1. Background (Resized Video)
        bg_data = cv2.flip(frame_resized, 0).tobytes()
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, screen_w, 0, screen_h, -1, 1)
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity()
        glRasterPos2i(0, 0)
        glDrawPixels(screen_w, screen_h, GL_BGR, GL_UNSIGNED_BYTE, bg_data)
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW)

        # 2. Model
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
