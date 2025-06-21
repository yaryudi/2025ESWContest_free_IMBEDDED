from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QScrollArea, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox, QPushButton, QGridLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QTimer
import sys
import os
from functools import partial

class WebMapTest(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polytopia Map Generator (Web + PyQt Panel)")
        self.resize(1850, 1200)
        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        # --- 왼쪽 패널 (mapGeneration.py의 컨트롤) ---
        self.controls_frame = QFrame()
        self.controls_frame.setFrameShape(QFrame.StyledPanel)
        self.controls_frame.setMaximumWidth(350)
        ctr = QVBoxLayout(self.controls_frame)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_w = QWidget()
        form = QFormLayout(form_w)

        self.map_size_input = QSpinBox();    self.map_size_input.setRange(5,50);    self.map_size_input.setValue(30)
        self.initial_land_input = QDoubleSpinBox(); self.initial_land_input.setRange(0.0,1.0); self.initial_land_input.setSingleStep(0.01); self.initial_land_input.setValue(0.55)
        self.smoothing_input = QSpinBox();     self.smoothing_input.setRange(0,10);     self.smoothing_input.setValue(3)
        self.relief_input = QSpinBox();        self.relief_input.setRange(1,8);       self.relief_input.setValue(4)
        self.tribes_input = QLineEdit("Ai-mo Aquarion Bardur Imperius Oumaji Quetzali Vengir Xin-xi Yadakk")
        self.fill_tribe_input = QLineEdit()
        self.no_biomes_checkbox = QCheckBox("Don't generate biomes")
        self.no_resources_checkbox = QCheckBox("Don't generate resources")

        form.addRow("Map Size:", self.map_size_input)
        form.addRow("Initial Land:", self.initial_land_input)
        form.addRow("Smoothing:", self.smoothing_input)
        form.addRow("Relief:", self.relief_input)
        form.addRow("Tribes:", self.tribes_input)
        form.addRow("Fill Tribe:", self.fill_tribe_input)
        form.addRow(self.no_biomes_checkbox)
        form.addRow(self.no_resources_checkbox)

        scroll.setWidget(form_w)
        ctr.addWidget(scroll)

        # 기물 이동 컨트롤 추가
        unit_control = QFormLayout()
        self.unit_id_input = QSpinBox()
        self.unit_id_input.setRange(1, 4)
        self.unit_id_input.setValue(1)
        unit_control.addRow("Unit ID:", self.unit_id_input)

        direction_layout = QGridLayout()
        self.btn_unit_up = QPushButton('↑')
        self.btn_unit_down = QPushButton('↓')
        self.btn_unit_left = QPushButton('←')
        self.btn_unit_right = QPushButton('→')
        self.btn_unit_ul = QPushButton('↖')
        self.btn_unit_ur = QPushButton('↗')
        self.btn_unit_dl = QPushButton('↙')
        self.btn_unit_dr = QPushButton('↘')

        direction_layout.addWidget(self.btn_unit_ul, 0, 0)
        direction_layout.addWidget(self.btn_unit_up, 0, 1)
        direction_layout.addWidget(self.btn_unit_ur, 0, 2)
        direction_layout.addWidget(self.btn_unit_left, 1, 0)
        direction_layout.addWidget(self.btn_unit_right, 1, 2)
        direction_layout.addWidget(self.btn_unit_dl, 2, 0)
        direction_layout.addWidget(self.btn_unit_down, 2, 1)
        direction_layout.addWidget(self.btn_unit_dr, 2, 2)

        unit_control.addRow("Unit Direction:", direction_layout)
        ctr.addLayout(unit_control)

        # 기물 이동 버튼 연결
        self.btn_unit_up.clicked.connect(lambda: self.move_unit('up'))
        self.btn_unit_down.clicked.connect(lambda: self.move_unit('down'))
        self.btn_unit_left.clicked.connect(lambda: self.move_unit('left'))
        self.btn_unit_right.clicked.connect(lambda: self.move_unit('right'))
        self.btn_unit_ul.clicked.connect(lambda: self.move_unit('up-left'))
        self.btn_unit_ur.clicked.connect(lambda: self.move_unit('up-right'))
        self.btn_unit_dl.clicked.connect(lambda: self.move_unit('down-left'))
        self.btn_unit_dr.clicked.connect(lambda: self.move_unit('down-right'))

        zoom_layout = QHBoxLayout()
        self.btn_zoom_in = QPushButton('+')
        self.btn_zoom_out = QPushButton('-')
        zoom_layout.addWidget(self.btn_zoom_in)
        zoom_layout.addWidget(self.btn_zoom_out)
        ctr.addLayout(zoom_layout)

        # 이동/줌 상태 변수
        self.offset_x = 0
        self.offset_y = 0
        self.tile_size = 48  # 기본값, JS에서 tile_size 변수와 동기화 필요

        def zoom_map(delta):
            self.tile_size = max(16, min(128, self.tile_size + delta))
            js = f'''
                window.tile_size = {self.tile_size};
                if (typeof map !== 'undefined' && typeof display_map === 'function') {{
                    window.tile_size = {self.tile_size};
                    display_map(map);
                }}
            '''
            self.webview.page().runJavaScript(js)

        self.btn_zoom_in.clicked.connect(partial(zoom_map, 8))
        self.btn_zoom_out.clicked.connect(partial(zoom_map, -8))

        btn = QPushButton("Generate Map (웹에서)")
        btn.clicked.connect(self.reload_web_map)
        ctr.addWidget(btn)

        main_layout.addWidget(self.controls_frame)

        # --- 오른쪽: QWebEngineView로 index.html 표시 ---
        self.webview = QWebEngineView()
        html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "index.html"))
        self.webview.load(QUrl.fromLocalFile(html_path))
        self.webview.setMinimumWidth(800)  # 최소 너비 설정
        self.webview.setMinimumHeight(600)  # 최소 높이 설정
        main_layout.addWidget(self.webview, 1)

        # 페이지 로드 완료 후 맵 생성
        self.webview.loadFinished.connect(self.on_load_finished)

        QTimer.singleShot(200, self.force_resize)

    def move_unit(self, direction):
        unit_id = self.unit_id_input.value()
        js = f'''
            if (typeof move_unit === 'function') {{
                move_unit({unit_id}, "{direction}");
            }}
        '''
        self.webview.page().runJavaScript(js)

    def reload_web_map(self):
        js = '''
            if (typeof generate === 'function') {
                generate();
            }
        '''
        self.webview.page().runJavaScript(js)

    def on_load_finished(self):
        # 페이지가 로드되면 자동으로 맵 생성
        QTimer.singleShot(100, self.reload_web_map)  # 약간의 지연을 주어 페이지가 완전히 로드되도록 함

    def force_resize(self):
        self.resize(self.width()+1, self.height())
        self.resize(self.width()-1, self.height())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WebMapTest()
    win.show()
    sys.exit(app.exec_())
