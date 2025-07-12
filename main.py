
import sys
import math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout, QMessageBox, QFileDialog, QGroupBox,
    QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Physics Engine ---
class ProjectilePhysics:
    GRAVITY = 9.81  # m/s^2

    def __init__(self, v0, angle_deg, h0, air_resistance=False):
        self.v0 = v0
        self.angle_deg = angle_deg
        self.h0 = h0
        self.air_resistance = air_resistance

        # Calculated quantities
        self.vx = None
        self.vy = None
        self.t_flight = None
        self.max_height = None
        self.range = None
        self.x_points = None
        self.y_points = None

        self.calculate()

    def calculate(self):
        theta = math.radians(self.angle_deg)
        self.vx = self.v0 * math.cos(theta)
        self.vy = self.v0 * math.sin(theta)

        # Time of flight (quadratic solution)
        discrim = self.vy**2 + 2 * self.GRAVITY * self.h0
        self.t_flight = (self.vy + math.sqrt(discrim)) / self.GRAVITY

        # Max height
        self.max_height = self.h0 + (self.vy**2) / (2 * self.GRAVITY)

        # Range
        self.range = self.vx * self.t_flight

        # Trajectory points
        t = np.linspace(0, self.t_flight, num=500)
        self.x_points = self.vx * t
        self.y_points = self.h0 + self.vy * t - 0.5 * self.GRAVITY * t**2

    def get_times_for_y(self, y_target):
        """
        Returns all positive times t where the projectile is at y_target.
        """
        a = -0.5 * self.GRAVITY
        b = self.vy
        c = self.h0 - y_target
        disc = b**2 - 4*a*c
        if disc < 0:
            return []
        t1 = (-b + math.sqrt(disc)) / (2*a)
        t2 = (-b - math.sqrt(disc)) / (2*a)
        times = []
        for t in (t1, t2):
            if t >= 0 and t <= self.t_flight:
                times.append(t)
        return times

    def is_target_reachable(self, x_target, y_target):
        """
        Returns (bool, details) whether (x_target, y_target) is on the trajectory.
        """
        # Find times when y(t) == y_target
        times = self.get_times_for_y(y_target)
        for t in times:
            x = self.vx * t
            if abs(x - x_target) < 1e-2:  # Allow for floating point tolerance
                return True, {"t": t, "actual_x": x, "actual_y": y_target}
        return False, None

    def get_closest_point_to(self, x_target, y_target):
        """
        For a target (x, y), find the closest point on the trajectory and its distance.
        Returns (min_distance, point_x, point_y, t).
        """
        distances = np.sqrt((self.x_points - x_target)**2 + (self.y_points - y_target)**2)
        idx = np.argmin(distances)
        return distances[idx], self.x_points[idx], self.y_points[idx], idx * self.t_flight / (len(self.x_points)-1)

# --- Matplotlib Plot Widget ---
class TrajectoryPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setParent(parent)
        self.setStyleSheet("background: transparent;")

    def plot_trajectory(self, x, y, title, target_x=None, target_y=None, hit=None, closest=None):
        self.ax.clear()
        self.ax.plot(x, y, color="#0099FF", linewidth=3, label="Projectile Path", zorder=2)
        self.ax.scatter([x[0]], [y[0]], color="#3DDC97", s=80, zorder=3, label="Launch")
        self.ax.scatter([x[-1]], [0], color="#FF6363", s=80, zorder=3, label="Impact")

        # Draw target
        if target_x is not None and target_y is not None:
            self.ax.scatter([target_x], [target_y], color="#FFB800", marker="*", s=180, zorder=4, label="🎯 Target")
            # Optionally draw connecting line and annotate
            if hit:
                self.ax.annotate(f"Hit!\n({target_x:.2f}, {target_y:.2f})",
                                 xy=(target_x, target_y), xytext=(target_x+3, target_y+5),
                                 arrowprops=dict(arrowstyle="->", color="#FFB800"), fontsize=12,
                                 color="#C75146")
            elif closest:
                dist, cx, cy, t = closest
                self.ax.plot([target_x, cx], [target_y, cy], color="#FFB800", linestyle="--", linewidth=1.2, zorder=1)
                self.ax.scatter([cx], [cy], color="#B25DFF", marker="o", s=60, zorder=4, label="Closest")
                self.ax.annotate(f"Closest: ({cx:.2f}, {cy:.2f})",
                                 xy=(cx, cy), xytext=(cx+3, cy+5),
                                 arrowprops=dict(arrowstyle="->", color="#B25DFF"), fontsize=11,
                                 color="#B25DFF")
        self.ax.set_xlabel("Distance (m)", fontsize=13, color="#222")
        self.ax.set_ylabel("Height (m)", fontsize=13, color="#222")
        self.ax.set_title(title, fontsize=15, color="#222", pad=12, fontweight='bold')
        self.ax.grid(True, linestyle="--", alpha=0.25)
        self.ax.legend(loc="upper right", fontsize=11, frameon=False)
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(bottom=0)
        self.fig.patch.set_facecolor("#F8FAFB")
        self.ax.set_facecolor("#F8FAFB")
        self.draw()

    def save_plot(self, filename):
        self.fig.savefig(filename, bbox_inches="tight", facecolor="#F8FAFB")

# --- Modern Styled Main GUI ---
class ProjectileSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 Projectile Motion Simulator")
        self.setWindowIcon(QIcon.fromTheme("applications-education"))
        self.setGeometry(120, 80, 1080, 670)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F2F6FC;
            }
        """)
        self.physics = None
        self.target_x = None
        self.target_y = None

        # --- Input Panel ---
        input_panel = QGroupBox("🎛️ Launch Conditions")
        input_panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 18px; color: #005377;
                border: 2px solid #e0e6ed;
                border-radius: 10px;
                background: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                margin-top: 0px;
                background: transparent;
            }
        """)
        form = QFormLayout()
        font_input = QFont("Segoe UI", 12)
        self.input_speed = QLineEdit("30")
        self.input_speed.setFont(font_input)
        self.input_angle = QLineEdit("45")
        self.input_angle.setFont(font_input)
        self.input_height = QLineEdit("0")
        self.input_height.setFont(font_input)
        self.input_target_x = QLineEdit("")
        self.input_target_y = QLineEdit("")
        self.input_target_x.setFont(font_input)
        self.input_target_y.setFont(font_input)
        self.input_target_x.setPlaceholderText("e.g. 60 (meters)")
        self.input_target_y.setPlaceholderText("e.g. 5 (meters)")
        self.input_speed.setPlaceholderText("e.g. 30")
        self.input_angle.setPlaceholderText("e.g. 45")
        self.input_height.setPlaceholderText("e.g. 0")
        form.addRow("Initial Speed (m/s):", self.input_speed)
        form.addRow("Launch Angle (deg):", self.input_angle)
        form.addRow("Initial Height (m):", self.input_height)
        form.addRow("🎯 Target X (m):", self.input_target_x)
        form.addRow("🎯 Target Y (m):", self.input_target_y)
        input_panel.setLayout(form)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.btn_simulate = QPushButton("Simulate")
        self.btn_simulate.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.btn_simulate.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #37A2FF, stop:1 #0099FF);
                color: white; border-radius: 8px; padding: 10px 28px;
                font-size: 16px; margin-top: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0099FF, stop:1 #37A2FF);
                color: #fff;
                box-shadow: 0 2px 8px #b8eaff;
            }
        """)
        self.btn_save = QPushButton("Save Graph")
        self.btn_save.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: #F0F4F9; color: #0078D7; border-radius: 8px; padding: 10px 18px;
                font-size: 15px; border: 1.5px solid #cce6ff;
            }
            QPushButton:hover {
                background: #eaf7ff; color: #0099FF;
            }
        """)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_simulate)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addStretch()

        # --- Info Panel ---
        info_panel = QGroupBox("📊 Results")
        info_panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 18px; color: #005377;
                border: 2px solid #e0e6ed; border-radius: 10px; margin-top: 10px;
                background: #F8FAFB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                margin-top: 0px;
                background: transparent;
            }
        """)
        font_info = QFont("Segoe UI", 13)
        self.lbl_tof = QLabel("Time of Flight: -")
        self.lbl_tof.setFont(font_info)
        self.lbl_tof.setStyleSheet("color:#007862;")
        self.lbl_height = QLabel("Max Height: -")
        self.lbl_height.setFont(font_info)
        self.lbl_height.setStyleSheet("color:#005377;")
        self.lbl_range = QLabel("Total Range: -")
        self.lbl_range.setFont(font_info)
        self.lbl_range.setStyleSheet("color:#C75146;")
        self.lbl_target = QLabel("Target: -")
        self.lbl_target.setFont(font_info)
        self.lbl_target.setStyleSheet("color:#FFB800;")
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.lbl_tof)
        info_layout.addWidget(self.lbl_height)
        info_layout.addWidget(self.lbl_range)
        info_layout.addWidget(self.lbl_target)
        info_panel.setLayout(info_layout)

        # --- Plot Widget ---
        self.plot = TrajectoryPlot(self)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot.setMinimumWidth(560)

        # --- Left Layout ---
        left_panel = QVBoxLayout()
        left_panel.addWidget(input_panel)
        left_panel.addSpacing(10)
        left_panel.addLayout(btn_layout)
        left_panel.addWidget(info_panel)
        left_panel.addStretch()

        # --- Divider ---
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        divider.setLineWidth(2)
        divider.setStyleSheet("color:#dee4ea; background:#dee4ea; min-width:2px;")

        # --- Main Layout ---
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel, stretch=2)
        main_layout.addWidget(divider)
        main_layout.addWidget(self.plot, stretch=5)

        # --- Container Widget ---
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # --- Connections ---
        self.btn_simulate.clicked.connect(self.on_simulate)
        self.btn_save.clicked.connect(self.on_save_graph)

        # --- Set modern palette for better look ---
        self.set_palette()

    def set_palette(self):
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("#F2F6FC"))
        pal.setColor(QPalette.Base, QColor("#FFFFFF"))
        pal.setColor(QPalette.AlternateBase, QColor("#F8FAFB"))
        pal.setColor(QPalette.Button, QColor("#0099FF"))
        pal.setColor(QPalette.ButtonText, QColor("#FFFFFF"))
        self.setPalette(pal)

    def on_simulate(self):
        try:
            v0 = float(self.input_speed.text())
            angle = float(self.input_angle.text())
            h0 = float(self.input_height.text())
            if v0 < 0 or angle < 0 or angle > 90 or h0 < 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers:\n- Speed ≥ 0\n- Angle 0–90°\n- Height ≥ 0")
            return

        # Target x, y
        self.target_x = None
        self.target_y = None
        hit = False
        closest = None
        target_x_str = self.input_target_x.text().strip()
        target_y_str = self.input_target_y.text().strip()
        if target_x_str or target_y_str:
            try:
                tx = float(target_x_str) if target_x_str else None
                ty = float(target_y_str) if target_y_str else None
                if tx is not None and tx < 0:
                    raise ValueError
                if ty is not None and ty < 0:
                    raise ValueError
                self.target_x = tx if tx is not None else None
                self.target_y = ty if ty is not None else None
            except Exception:
                QMessageBox.warning(self, "Invalid Target", "Please enter valid target X,Y (meters, ≥ 0) or leave blank.")
                return

        self.physics = ProjectilePhysics(v0, angle, h0)

        if self.target_x is not None and self.target_y is not None:
            # Check if the target is hit (on the trajectory)
            hit, details = self.physics.is_target_reachable(self.target_x, self.target_y)
            if hit:
                self.lbl_target.setText(f"🎯 Target HIT: (X={self.target_x:.2f}, Y={self.target_y:.2f})")
            else:
                # Not exactly hit, show closest point
                closest = self.physics.get_closest_point_to(self.target_x, self.target_y)
                dist, cx, cy, t = closest
                self.lbl_target.setText(f"Closest to Target: ({cx:.2f}, {cy:.2f}), Δ={dist:.2f} m")
        elif self.target_x is not None:
            # Only X entered; show y on trajectory at that x
            y_at_x = self.physics.get_y_at_x(self.target_x)
            if y_at_x is not None:
                self.lbl_target.setText(f"Target at X={self.target_x:.2f}: Y={y_at_x:.2f} m")
            else:
                self.lbl_target.setText(f"Target X={self.target_x:.2f} m is out of trajectory range")
        elif self.target_y is not None:
            # Only Y entered; show all times projectile is at Y
            times = self.physics.get_times_for_y(self.target_y)
            if times:
                xs = [self.physics.vx * t for t in times]
                xtext = ", ".join([f"{x:.2f}" for x in xs])
                self.lbl_target.setText(f"At Y={self.target_y:.2f}: X={xtext} m")
            else:
                self.lbl_target.setText(f"Target Y={self.target_y:.2f} m is not reached")

        else:
            self.lbl_target.setText("Target: -")

        # Update info panel
        self.lbl_tof.setText(f"⏱️ Time of Flight: <b>{self.physics.t_flight:.2f} s</b>")
        self.lbl_height.setText(f"🗻 Max Height: <b>{self.physics.max_height:.2f} m</b>")
        self.lbl_range.setText(f"📏 Total Range: <b>{self.physics.range:.2f} m</b>")

        # Plot
        title = f"Projectile Trajectory\n(v₀={v0} m/s, θ={angle}°, h₀={h0} m)"
        self.plot.plot_trajectory(
            self.physics.x_points, self.physics.y_points, title,
            target_x=self.target_x,
            target_y=self.target_y,
            hit=hit, closest=closest
        )

    def on_save_graph(self):
        if not self.physics:
            QMessageBox.information(self, "Nothing to Save", "Please run a simulation first.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Graph", "trajectory.png", "PNG Files (*.png);;PDF Files (*.pdf)")
        if filename:
            self.plot.save_plot(filename)
            QMessageBox.information(self, "Saved", f"Graph saved to:\n{filename}")

# --- Main Entry Point ---
def main():
    app = QApplication(sys.argv)
    win = ProjectileSimulator()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()