import cv2
import cv2.aruco as aruco
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import trimesh

# ==========================================
# 1. CONFIGURATION (UPDATED)
# ==========================================
# CHANGED: Pointing to a new model from your list
MODEL_PATH = "assets/space-stat/columbus.glb"

MARKER_SIZE = 0.05  # 5cm (Physical marker size)
DICT_TO_USE = cv2.aruco.DICT_7X7_50

# CHANGED: Increased from 0.05 to 0.20
# 0.05 = Same size as marker (Small)
# 0.20 = 4x larger than marker (Big)
MODEL_SCALE = 0.20
SMOOTHING = 0.5

# ==========================================
# 2. UNIVERSAL MODEL LOADER
# ==========================================
model_list_id = None


def load_model_to_gpu():
    global model_list_id
    print(f"⏳ Loading: {MODEL_PATH}")

    try:
        mesh = trimesh.load(MODEL_PATH)
        if isinstance(mesh, trimesh.Scene):
            print("   → Merging scene geometry...")
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        # 1. CENTER IT (Fixes offset issues)
        mesh.apply_translation(-mesh.centroid)

        # 2. NORMALIZE IT (Fixes size issues)
        # We force the model to be exactly 1.0 meter max dimension
        max_span = np.max(mesh.extents)
        if max_span > 0:
            mesh.apply_scale(1.0 / max_span)

        # 3. BAKE COLORS (Fixes black textures)
        if hasattr(mesh.visual, 'to_color'):
            print("   → Baking textures...")
            mesh.visual = mesh.visual.to_color()

    except Exception as e:
        print(f"⚠️ Load Error: {e}")
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

                if len(normals) > 0:
                    n = normals[vertex_i]
                    glNormal3f(n[0], n[1], n[2])

                v = vertices[vertex_i]
                glVertex3f(v[0], v[1], v[2])
        glEnd()
    else:
        glColor3f(1, 0, 0)
        glutSolidCube(1.0)

    glEndList()
    print(f"✅ Model Ready (ID: {model_list_id})")


# ==========================================
# 3. TRACKING & FILTER
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

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST)
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 5.0, 5.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.9, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))
    glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
    glMaterialf(GL_FRONT, GL_SHININESS, 100.0)


def main():
    pygame.init()
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h

    pygame.display.set_mode((screen_w, screen_h), FULLSCREEN | DOUBLEBUF | OPENGL)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    init_gl(screen_w, screen_h)
    load_model_to_gpu()

    aruco_dict = aruco.getPredefinedDictionary(DICT_TO_USE)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    marker_pts = np.array([
        [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0], [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
        [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0], [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
    ], dtype=np.float32)

    f = screen_w / (2 * np.tan(np.radians(60 / 2)))
    mtx = np.array([[f, 0, screen_w / 2], [0, f, screen_h / 2], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5)

    filter = PoseFilter(alpha=SMOOTHING)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); return
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE: pygame.quit(); return

        ret, frame = cap.read()
        if not ret: break

        frame_resized = cv2.resize(frame, (screen_w, screen_h))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        found = False
        rvec_raw, tvec_raw = None, None

        if ids is not None:
            ret_pnp, rvec_raw, tvec_raw = cv2.solvePnP(marker_pts, corners[0], mtx, dist)
            found = ret_pnp

        should_render, rvec, tvec = filter.update(rvec_raw, tvec_raw, found)

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
