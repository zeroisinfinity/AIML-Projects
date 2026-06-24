import cv2
import cv2.aruco as aruco
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import trimesh
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
ASSET_FOLDER = "assets"
MARKER_SIZE = 0.05
DICT_TO_USE = cv2.aruco.DICT_7X7_50
MODEL_SCALE = 0.15
SMOOTHING = 0.6

# ==========================================
# 2. AUTO-SCANNER
# ==========================================
available_models = []
current_model_idx = 0


def scan_assets():
    global available_models
    print(f"🔍 Scanning '{ASSET_FOLDER}' for 3D models...")
    for root, dirs, files in os.walk(ASSET_FOLDER):
        for file in files:
            if file.lower().endswith(('.glb', '.gltf', '.obj')):
                full_path = os.path.join(root, file)
                available_models.append(full_path)

    if len(available_models) == 0:
        print("❌ No models found. Add .glb files to 'assets/'")
        exit()
    print(f"✅ Found {len(available_models)} models!")


# ==========================================
# 3. UNIVERSAL LOADER
# ==========================================
model_list_id = None


def load_current_model():
    global model_list_id, current_model_idx
    path = available_models[current_model_idx]
    print(f"\n⬇️  Loading: {os.path.basename(path)}")

    if model_list_id is not None:
        glDeleteLists(model_list_id, 1)

    try:
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        mesh.apply_translation(-mesh.centroid)
        max_span = np.max(mesh.extents)
        if max_span > 0:
            mesh.apply_scale(1.0 / max_span)

        if hasattr(mesh.visual, 'to_color'):
            try:
                mesh.visual = mesh.visual.to_color()
            except:
                pass

    except Exception as e:
        print(f"⚠️ Load Error: {e}")
        mesh = None

    model_list_id = glGenLists(1)
    glNewList(model_list_id, GL_COMPILE)

    if mesh:
        glEnable(GL_NORMALIZE)
        glBegin(GL_TRIANGLES)
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors
        else:
            colors = None

        vertices = mesh.vertices
        normals = mesh.vertex_normals
        faces = mesh.faces

        for face in faces:
            for vertex_i in face:
                if colors is not None:
                    c = colors[vertex_i]
                    glColor3f(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
                else:
                    glColor3f(0.9, 0.9, 0.9)  # Bright White default

                if len(normals) > 0:
                    n = normals[vertex_i]
                    glNormal3f(n[0], n[1], n[2])

                v = vertices[vertex_i]
                glVertex3f(v[0], v[1], v[2])
        glEnd()
    else:
        glColor3f(1, 0, 0);
        glutSolidCube(1.0)

    glEndList()


# ==========================================
# 4. TRACKING
# ==========================================
class PoseFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.rvec = None;
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
# 5. STUDIO LIGHTING SETUP
# ==========================================
def init_gl(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity()
    gluPerspective(60, (w / h), 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW)

    # --- LIGHTING ENGINE UPGRADE ---
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)  # Key Light (Sun)
    glEnable(GL_LIGHT1)  # Fill Light (Sky)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)  # Smooth shading for rounded metal

    # 1. KEY LIGHT (Strong White from Right)
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 5.0, 5.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.2, 1.2, 1.2, 1.0))  # Overdriven Brightness
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))  # Sharp Highlights

    # 2. FILL LIGHT (Soft Blue from Left - Kills shadows)
    glLightfv(GL_LIGHT1, GL_POSITION, (-5.0, 2.0, 5.0, 1.0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.5, 0.5, 0.6, 1.0))  # Cool tone

    # 3. GLOBAL AMBIENT (Base brightness)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.6, 0.6, 0.6, 1.0))

    # 4. MATERIAL (High Polish Metal)
    glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
    glMaterialf(GL_FRONT, GL_SHININESS, 128.0)  # 0-128 (Max Shine)


def main():
    global current_model_idx, MODEL_SCALE
    scan_assets()

    pygame.init()
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h
    pygame.display.set_mode((screen_w, screen_h), FULLSCREEN | DOUBLEBUF | OPENGL)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280);
    cap.set(4, 720)

    init_gl(screen_w, screen_h)
    load_current_model()

    aruco_dict = aruco.getPredefinedDictionary(DICT_TO_USE)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    marker_pts = np.array([[-MARKER_SIZE / 2, MARKER_SIZE / 2, 0], [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                           [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0], [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]],
                          dtype=np.float32)

    f = screen_w / (2 * np.tan(np.radians(60 / 2)))
    mtx = np.array([[f, 0, screen_w / 2], [0, f, screen_h / 2], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5)

    filter = PoseFilter(alpha=SMOOTHING)
    clock = pygame.time.Clock()

    print("\n🎮 CONTROLS: [N] Next | [P] Prev | [+/-] Size | [ESC] Quit")

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); return
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit(); return
                elif event.key == K_n:
                    current_model_idx = (current_model_idx + 1) % len(available_models)
                    load_current_model()
                elif event.key == K_p:
                    current_model_idx = (current_model_idx - 1) % len(available_models)
                    load_current_model()
                elif event.key == K_EQUALS or event.key == K_PLUS:
                    MODEL_SCALE += 0.02
                elif event.key == K_MINUS:
                    MODEL_SCALE = max(0.01, MODEL_SCALE - 0.02)

        ret, frame = cap.read()
        if not ret: break
        frame_resized = cv2.resize(frame, (screen_w, screen_h))

        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        found, rvec, tvec = False, None, None
        if ids is not None:
            found, rvec, tvec = cv2.solvePnP(marker_pts, corners[0], mtx, dist)

        should_render, rvec, tvec = filter.update(rvec, tvec, found)

        bg_data = cv2.flip(frame_resized, 0).tobytes()
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, screen_w, 0, screen_h, -1, 1)
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
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
            view_matrix = np.eye(4);
            view_matrix[:3, :3] = R;
            view_matrix[:3, 3] = tvec.squeeze()
            view_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ view_matrix
            glLoadMatrixd(view_matrix.T)
            glScalef(MODEL_SCALE, MODEL_SCALE, MODEL_SCALE)
            glRotatef(90, 1, 0, 0)
            glCallList(model_list_id)

        pygame.display.flip()
        clock.tick(60)

    cap.release()


if __name__ == "__main__":
    main()
