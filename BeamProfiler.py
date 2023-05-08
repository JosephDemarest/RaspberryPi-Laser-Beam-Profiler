# Vyir
# Vyirtech.com

# required imports
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import QThread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import datetime
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import math
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

# ignore command line warnings
import warnings

warnings.filterwarnings("ignore")

# Gaussian function that takes x values, amplitude (a),
# center position (x0), and standard deviation (sigma) as input


def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


# Fit a Gaussian curve to the given `data` and return the optimized
# parameters: amplitude (a), center position (x0), and standard deviation (sigma)


def fit_gaussian(data):
    x = np.arange(len(data))
    mean = np.sum(x * data) / np.sum(data)
    sigma = np.sqrt(np.sum(data * (x - mean) ** 2) / np.sum(data))
    popt, _ = curve_fit(
        gaussian, x, data, p0=[np.max(data), mean, sigma], maxfev=100000
    )
    return popt


# Calculate and return the full width at half maximum (FWHM) for
# a Gaussian curve given its standard deviation (sigma)


def full_width_half_maximum(sigma):
    return sigma * np.sqrt(8 * np.log(2))


# main GUI window definition
class Ui_MainWindow(object):
    # set camera resolution which will be passed through the whole program

    W, H = 640, 480

    # setup UI elements

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Beam GUI")
        # Set the fixed size of the MainWindow for convenient display on a Raspberry Pi desktop
        MainWindow.setFixedSize(1655, 1066)

        # Create central widget for the main window and set its object name
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Create a tab widget and set its geometry and object name
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(14, 8, 1600, 1066))
        self.tabWidget.setObjectName("tabWidget")

        # Create the first tab (Camera view) and set its object name
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        # Create a line edit widget for displaying the root directory, set its geometry and object name
        self.lineEdit = QtWidgets.QLineEdit(MainWindow)
        self.lineEdit.setGeometry(QtCore.QRect(156, 44, 539, 25))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("Root directory: " + os.getcwd())

        # Create a label for the line edit widget, set its geometry and object name
        self.label = QtWidgets.QLabel(MainWindow)
        self.label.setGeometry(QtCore.QRect(122, 45, 65, 21))
        self.label.setObjectName("label")

        # Create the first push button for starting the image acquisition, set its geometry and object name
        self.pushButton = QtWidgets.QPushButton(MainWindow)
        self.pushButton.setGeometry(QtCore.QRect(703, 45, 55, 23))
        self.pushButton.setObjectName("pushButton")

        # Create a text edit widget for tab 1, set its geometry and object name
        self.textEdit_2 = QtWidgets.QTextEdit(self.tab)
        self.textEdit_2.setGeometry(QtCore.QRect(52, 660, 801, 100))
        self.textEdit_2.setObjectName("textEdit_2")

        # Add the first tab (Camera view) to the tab widget
        self.tabWidget.addTab(self.tab, "")

        # Create the second tab (Beam view) and set its object name
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")

        # Add the second tab (Beam view) to the tab widget
        self.tabWidget.addTab(self.tab_2, "")

        # Set the central widget of the main window
        MainWindow.setCentralWidget(self.centralwidget)

        # Create a menu bar for the main window and set its geometry and object name
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1343, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        # Create a status bar for the main window and set its object name
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

       # Create widgets for displaying centroid and estimated beam width
        # Labels show d4 sigma, while LCD widgets display the computed widths

        # Create a label to display the centroid, set its font, text, geometry
        self.label_centroid = QtWidgets.QLabel(self.tab_2)
        self.label_centroid.setFont(QtGui.QFont("Any", 12))
        self.label_centroid.setText("Centroid (x,y) = 0, 0")
        self.label_centroid.setGeometry(QtCore.QRect(550, 540, 250, 30))

        # Create a label and LCD widget for displaying the d4 sigma in x-direction
        self.label_dx = QtWidgets.QLabel(self.tab_2)
        self.label_dx.setGeometry(QtCore.QRect(20, 230, 101, 41))
        self.lcdNumber_dx = QtWidgets.QLCDNumber(self.tab_2)
        self.lcdNumber_dx.setGeometry(QtCore.QRect(20, 260, 81, 41))
        self.lcdNumber_dx.display(0)

        # Create a label and LCD widget for displaying the d4 sigma in y-direction
        self.label_dy = QtWidgets.QLabel(self.tab_2)
        self.label_dy.setGeometry(QtCore.QRect(20, 300, 101, 41))
        self.lcdNumber_dy = QtWidgets.QLCDNumber(self.tab_2)
        self.lcdNumber_dy.setGeometry(QtCore.QRect(20, 330, 81, 41))
        self.lcdNumber_dy.display(0)

        # Create a label and combo box for selecting the camera resolution
        self.label_resolution = QtWidgets.QLabel(self.tab)
        self.label_resolution.setGeometry(QtCore.QRect(820, 210, 101, 41))
        self.label_resolution.setText("Resolution:")
        self.comboBox_resolution = QtWidgets.QComboBox(self.tab)
        self.comboBox_resolution.setGeometry(QtCore.QRect(820, 240, 121, 31))
        self.comboBox_resolution.addItems(
            [
                "640x480",
                "1280x720",
                "1920x1080",
                "2560x1440",
                "4056x3040",
            ]
        )

        # Create widgets for adjustable aperture
        # Labels show aperture x, y, radius, and line edits allow for adjusting the digital aperture

        # Create a label and line edit for adjusting the aperture in x-direction
        self.label_apx = QtWidgets.QLabel(self.tab_2)
        self.label_apx.setGeometry(QtCore.QRect(20, 380, 101, 41))
        self.lineEdit_apx = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_apx.setGeometry(QtCore.QRect(20, 410, 80, 40))
        self.lineEdit_apx.setText(str(int(self.W / 2)))

        # Create a label and line edit for adjusting the aperture in y-direction
        self.label_apy = QtWidgets.QLabel(self.tab_2)
        self.label_apy.setGeometry(QtCore.QRect(20, 440, 101, 41))
        self.lineEdit_apy = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_apy.setGeometry(QtCore.QRect(20, 470, 80, 40))
        # Set the y-coordinate for the adjustable aperture
        self.lineEdit_apy.setText(str(int(self.H / 2)))

        # Create a label and line edit for adjusting the aperture radius
        self.label_apr = QtWidgets.QLabel(self.tab_2)
        self.label_apr.setGeometry(QtCore.QRect(20, 500, 111, 40))
        self.lineEdit_apr = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_apr.setGeometry(QtCore.QRect(20, 530, 80, 40))
        self.lineEdit_apr.setText(str(int(self.H / 2 - 100)))

        # Create live charts for beam profile in x and y directions
        self.live_chart_x = self.create_live_chart_x(self.tab_2)
        self.live_chart_x.setGeometry(165, 600, 535, 300)
        self.live_chart_y = self.create_live_chart_y(self.tab_2)
        self.live_chart_y.setGeometry(900, 200, 535, 300)

        # Create a line edit for entering the save file prefix
        self.lineEdit_savePrefix = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_savePrefix.setGeometry(QtCore.QRect(955, 45, 100, 23))
        self.lineEdit_savePrefix.setPlaceholderText("Enter prefix")

        # Create a plain text edit for displaying small text
        self.plainTextEdit_smallText = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_smallText.setGeometry(QtCore.QRect(1000, 700, 535, 300))
        self.plainTextEdit_smallText.setObjectName("plainTextEdit_smallText")

        # Create labels and line edits for setting shutter speed and frame rate
        self.label_shutter = QtWidgets.QLabel(self.tab)
        self.label_shutter.setGeometry(QtCore.QRect(820, 60, 101, 41))
        self.label_shutter.setText("Shutter Speed")
        self.lineEdit_shutter = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_shutter.setGeometry(QtCore.QRect(820, 90, 80, 40))
        self.lineEdit_shutter.setText(str(int(1000)))

        self.label_framerate = QtWidgets.QLabel(self.tab)
        self.label_framerate.setGeometry(QtCore.QRect(820, 120, 101, 41))
        self.label_framerate.setText("Framerate:")
        self.lineEdit_frame = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_frame.setGeometry(QtCore.QRect(820, 150, 80, 40))
        self.lineEdit_frame.setText(str(int(1)))

        # Create labels, combo box, and line edits for setting AWB mode and gains
        self.label_awb = QtWidgets.QLabel(self.tab)
        self.label_awb.setGeometry(QtCore.QRect(940, 60, 101, 41))
        self.label_awb.setText("AWB Mode:")
        self.comboBox_awb = QtWidgets.QComboBox(self.tab)
        self.comboBox_awb.setGeometry(QtCore.QRect(940, 90, 121, 31))
        self.comboBox_awb.addItems(
            [
                "off",
                "auto",
                "sunlight",
                "cloudy",
                "shade",
                "tungsten",
                "fluorescent",
                "incandescent",
                "flash",
                "horizon",
            ]
        )

        # Create a label and line edits for setting AWB gains
        self.label_awb_gains = QtWidgets.QLabel(self.tab)
        self.label_awb_gains.setGeometry(QtCore.QRect(1100, 60, 101, 41))
        self.label_awb_gains.setText("AWB Gains:")
        self.lineEdit_awb_gains_r = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_awb_gains_r.setGeometry(QtCore.QRect(1100, 90, 61, 31))
        self.lineEdit_awb_gains_r.setText("3.1")
        self.lineEdit_awb_gains_b = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_awb_gains_b.setGeometry(QtCore.QRect(1180, 90, 61, 31))
        self.lineEdit_awb_gains_b.setText("3.1")

        # Create a label and line edit for setting brightness
        self.label_brightness = QtWidgets.QLabel(self.tab)
        self.label_brightness.setGeometry(QtCore.QRect(940, 120, 101, 41))
        self.label_brightness.setText("Brightness:")
        self.lineEdit_brightness = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_brightness.setGeometry(QtCore.QRect(940, 150, 61, 31))
        self.lineEdit_brightness.setText("50")

        # Create a label and combo box for setting meter mode
        self.label_meter_mode = QtWidgets.QLabel(self.tab)
        self.label_meter_mode.setGeometry(QtCore.QRect(1100, 120, 101, 41))
        self.label_meter_mode.setText("Meter Mode:")
        self.comboBox_meter_mode = QtWidgets.QComboBox(self.tab)
        self.comboBox_meter_mode.setGeometry(QtCore.QRect(1100, 150, 121, 31))
        self.comboBox_meter_mode.addItems(["average", "spot", "backlit", "matrix"])

        # Create a label and combo box for setting exposure mode
        self.label_exposure_mode = QtWidgets.QLabel(self.tab)
        self.label_exposure_mode.setGeometry(QtCore.QRect(940, 180, 101, 41))
        self.label_exposure_mode.setText("Exposure Mode:")
        self.comboBox_exposure_mode = QtWidgets.QComboBox(self.tab)
        self.comboBox_exposure_mode.setGeometry(QtCore.QRect(940, 210, 121, 31))
        self.comboBox_exposure_mode.addItems(
            [
                "off",
                "auto",
                "night",
                "nightpreview",
                "backlight",
                "spotlight",
                "sports",
                "snow",
                "beach",
                "verylong",
                "fixedfps",
                "antishake",
                "fireworks",
            ]
        )

        # Create a label and line edit for setting exposure compensation
        self.label_exposure_comp = QtWidgets.QLabel(self.tab)
        self.label_exposure_comp.setGeometry(QtCore.QRect(1100, 180, 101, 41))
        self.label_exposure_comp.setText("Exposure Comp:")
        self.lineEdit_exposure_comp = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_exposure_comp.setGeometry(QtCore.QRect(1100, 210, 61, 31))
        self.lineEdit_exposure_comp.setText("0")


        # Create a label and line edit for setting ISO
        self.label_iso = QtWidgets.QLabel(self.tab)
        self.label_iso.setGeometry(QtCore.QRect(820, 180, 101, 41))
        self.label_iso.setText("ISO:")
        self.lineEdit_iso = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_iso.setGeometry(QtCore.QRect(820, 210, 80, 40))
        self.lineEdit_iso.setText(str(int(1)))

        # Create a label and line edit for setting saturation
        self.label_saturation = QtWidgets.QLabel(self.tab)
        self.label_saturation.setGeometry(QtCore.QRect(820, 240, 101, 41))
        self.label_saturation.setText("Saturation:")
        self.lineEdit_saturation = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_saturation.setGeometry(QtCore.QRect(820, 270, 80, 40))
        self.lineEdit_saturation.setText(str(int(0)))

        # Create a push button for applying settings
        self.pushButton_apply = QtWidgets.QPushButton(MainWindow)
        self.pushButton_apply.setGeometry(QtCore.QRect(889, 45, 55, 23))
        self.pushButton_apply.setObjectName("pushButton_apply")

        # Create a push button for saving data
        self.pushButton_S = QtWidgets.QPushButton(MainWindow)
        self.pushButton_S.setGeometry(QtCore.QRect(765, 45, 55, 23))

        # Create a push button for logging data continuously
        self.pushButton_L = QtWidgets.QPushButton(MainWindow)
        self.pushButton_L.setGeometry(QtCore.QRect(827, 45, 55, 23))

        # Create image frames for raw image, beam image, and colorbar
        self.image_frame = QtWidgets.QLabel(self.tab)
        self.beam_frame = QtWidgets.QLabel(self.tab_2)
        self.cb_frame = QtWidgets.QLabel(self.tab_2)

        # Load colorbar image (cb.png) and set it to the colorbar frame
        colorbar = cv2.imread("cb.png")
        colorbar = cv2.cvtColor(colorbar, cv2.COLOR_RGB2BGR)
        self.cb_frame.move(790, 49)
        imGUI = QtGui.QImage(
            colorbar.data,
            colorbar.shape[1],
            colorbar.shape[0],
            colorbar.shape[1] * 3,
            QtGui.QImage.Format_RGB888,
        )
        self.cb_frame.setPixmap(QtGui.QPixmap.fromImage(imGUI))

        # Connect push buttons to corresponding functions
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.pushButton.clicked.connect(self.run)
        self.pushButton_S.clicked.connect(self.save)
        self.pushButton_L.clicked.connect(self.log)
        self.pushButton_apply.clicked.connect(self.apply)

        # Establish connections between objects and their corresponding slots
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    # Create a live chart for beam profile along x-axis at y-centroid
    def create_live_chart_x(self, parent):
        #Create an empty live chart in the given parent widget.
        fig = Figure(figsize=(7, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.set_title("Beam profile along x-axis at y-centroid")
        canvas = FigureCanvas(fig)
        canvas.setParent(parent)
        return canvas

    # Create a live chart for beam profile along y-axis at x-centroid
    def create_live_chart_y(self, parent):
        #Create an empty live chart in the given parent widget.
        fig = Figure(figsize=(7, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.set_title("Beam profile along y-axis at x-centroid")
        canvas = FigureCanvas(fig)
        canvas.setParent(parent)
        return canvas

    # Set text for GUI elements
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Beam GUI"))
        self.label.setText(_translate("MainWindow", "Info:"))
        self.pushButton.setText(_translate("MainWindow", "Run"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Camera")
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Beam")
        )
        self.label_dx.setText(_translate("MainWindow", "D4σx (μm)"))
        self.label_dy.setText(_translate("MainWindow", "D4σy (μm)"))
        self.label_apx.setText(_translate("MainWindow", "Aperture x"))
        self.label_apy.setText(_translate("MainWindow", "Aperture y"))
        self.label_apr.setText(_translate("MainWindow", "Ap. Radius"))
        self.pushButton_S.setText(_translate("MainWindow", "Save"))
        self.pushButton_L.setText(_translate("MainWindow", "Log"))
        self.pushButton_apply.setText(_translate("MainWindow", "Apply"))

    # run image acquisition and processing thread
    # Global variable for running state
    RUNNING = False

    # Start image acquisition and processing thread
    def run(self):
        if not self.RUNNING:
            self.threadA = captureThread(self, self.W, self.H)
            self.threadA.start()
            self.RUNNING = True
        else:
            self.lineEdit.setText("System already running")

    # Apply new settings to the running system
    def apply(self):
        if self.RUNNING:
            self.threadA.stop()
            time.sleep(2)
            self.threadA.stop_camera()
            self.threadA.start()
            self.RUNNING = False
        else:
            self.lineEdit.setText("System already running")

    # Start/stop logging of data
    def log(self):
        if self.RUNNING:
            if not self.threadA.LOGGING:
                self.threadA.SAVE_NOW = True
                self.threadA.LOGGING = True
                self.pushButton_L.setText("Stop")
            else:
                self.threadA.SAVE_NOW = False
                self.threadA.LOGGING = False
                self.pushButton_L.setText("Log")
                self.lineEdit.setText("Data logging stopped")
        else:
            self.lineEdit.setText("Run the system before logging data")

    # Save images and statistics
    def save(self):
        if self.RUNNING:
            self.threadA.SAVE_NOW = True
        else:
            self.lineEdit.setText("Run the system before saving data")


# thread which handles live image acquisition and beam image processing
# runs separately from main GUI thread to prevent hang ups
class captureThread(QThread):
    # variables which can be accessed across functions and threads
    image_live = np.empty(1)  # live camera image
    camera = None  # camera variable for PiCamera
    rawCapture = None  # rawCapture variable for PiCamera
    MainWindow = None  # MainWindow passed to thread so thread can modify UI elements
    SAVE_NOW = False  # flag to save all data once
    LOGGING = False  # flag to continuously log data
    # used to set camera and beam frame sizes and locations to draw images on
    FRAMES_INIT = False
    # used to reset aperture values if input is left blank
    count_x, count_y, count_r = 0, 0, 0
    # mask values for digital aperture. Changes based on text input
    mask_x, mask_y, mask_r = 1296, 972, 880
    W = 0  # camera/image width to be set
    H = 0  # camera/image height to be set
    # multiply a pixel width by 1.55 micron to get physical width #SENSOR DEPENDENT
    pixel_um = 1.55

    # initialize camera and set main window for interaction between thread and MainWindow
    def __init__(self, MainWindow, W, H):
        QThread.__init__(self)
        # set the camera resolution

        self.W, self.H = W, H
        self.MainWindow = MainWindow
        self.running = True
        self.init_camera()

    # capture live images and convert to beam profile
    def stop(self):  
        self.running = False

    # Continuously run live image acquisition and beam analysis while the system is running
    def run(self):
        while self.running:
            self.live_image()
            self.beam()
            self.update_live_chart()

    # Update the live charts for x and y profiles with the latest data
    def update_live_chart(self):
        # Create an empty mask of the same size as the live image
        mask = np.zeros([self.H, self.W])

        # Convert the live image to grayscale
        image = cv2.cvtColor(self.image_live, cv2.COLOR_BGR2GRAY)

        # Apply the mask to the live image (copy)
        image_m = np.copy(self.image_live)
        image_m[mask == 0] = 0

        # Calculate image moments and centroid coordinates
        MOM = cv2.moments(image)
        centroid_x = int(MOM["m10"] / MOM["m00"])
        centroid_y = int(MOM["m01"] / MOM["m00"])

        # Extract x and y profiles centered at the centroid
        x_prof = image[round(centroid_y), :]
        y_prof = image[:, round(centroid_x)]

        # Fit Gaussian to the x and y profiles
        popt_x = fit_gaussian(x_prof)
        popt_y = fit_gaussian(y_prof)
        fitted_x = gaussian(np.arange(len(x_prof)), *popt_x)
        fitted_y = gaussian(np.arange(len(y_prof)), *popt_y)

        # Update the live charts with new data
        self.update_chart(self.MainWindow.live_chart_x, x_prof, fitted_x)
        self.update_chart(self.MainWindow.live_chart_y, y_prof, fitted_y)

    # Update a given chart with new data and redraw the canvas
    def update_chart(self, chart, data, fitted_data):
        # Clear previous plot
        ax = chart.figure.get_axes()[0]
        ax.clear()

        # Update the chart with new data and fitted Gaussian
        ax.plot(range(len(data)), data, label="Data")
        ax.plot(
            range(len(fitted_data)),
            fitted_data,
            label="Fitted Gaussian",
            linestyle="--",
        )

        # Set axis limits and labels
        ax.set_xlim(0, len(data) - 1)
        ax.set_ylim(0, 255)
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Intensity")

        # Add a legend
        ax.legend()

        # Redraw the canvas
        chart.draw()


    # initialize camera settings
    def init_camera(self):
        # Get the selected resolution from the comboBox and set camera width and height
        resolution = self.MainWindow.comboBox_resolution.currentText().split("x")
        self.W, self.H = int(resolution[0]), int(resolution[1])

        # Initialize the PiCamera and set the resolution
        camera = PiCamera()
        camera.resolution = (self.W, self.H)
        rawCapture = PiRGBArray(camera, size=(self.W, self.H))

        # Allow camera to warm up
        time.sleep(0.1)

        # Set camera settings based on MainWindow inputs
        camera.awb_mode = self.MainWindow.comboBox_awb.currentText()
        r_gain = float(self.MainWindow.lineEdit_awb_gains_r.text())
        b_gain = float(self.MainWindow.lineEdit_awb_gains_b.text())
        camera.awb_gains = (r_gain, b_gain)
        camera.brightness = int(self.MainWindow.lineEdit_brightness.text())
        camera.meter_mode = self.MainWindow.comboBox_meter_mode.currentText()
        camera.exposure_mode = self.MainWindow.comboBox_exposure_mode.currentText()
        camera.exposure_compensation = int(
            self.MainWindow.lineEdit_exposure_comp.text()
        )
        camera.shutter_speed = int(self.MainWindow.lineEdit_shutter.text())
        camera.vflip = True
        camera.hflip = False
        camera.iso = int(self.MainWindow.lineEdit_iso.text())
        camera.saturation = int(self.MainWindow.lineEdit_saturation.text())

        # Optional settings for zoom and framerate
        ZOOM_BOOL = False
        if ZOOM_BOOL:
            crop_factor = 0.4
            roi_start_x = (1 - crop_factor) / 2
            roi_start_y = (1 - crop_factor) / 2
            camera.zoom = (roi_start_x, roi_start_y, crop_factor, crop_factor)

        # Print camera settings if desired
        CAMERA_SETTINGS = True
        if CAMERA_SETTINGS:
            print("AWB is " + str(camera.awb_mode))
            print("AWB gain is " + str(camera.awb_gains))
            print("Brightness is " + str(camera.brightness))
            print("Aperture is " + str(camera.exposure_compensation))
            print("Shutter speed is " + str(camera.shutter_speed))
            print("Camera exposure speed is " + str(camera.exposure_speed))
            print("Iso is " + str(camera.iso))
            print("Camera digital gain is " + str(camera.digital_gain))
            print("Camera analog gain is " + str(camera.analog_gain))
            print("Camera v/h flip is " + str(camera.vflip) + ", " + str(camera.hflip))
            print("Camera contrast is " + str(camera.contrast))
            print("Camera color saturation is " + str(camera.saturation))
            print("Camera meter mode is " + str(camera.meter_mode))
            # print("framerate "+str(camera.framerate))
            if ZOOM_BOOL:
                print("Camera crop factor is " + str(crop_factor))
            else:
                print("Camera crop is disabled")
        # Assign camera and rawCapture to the instance
        self.camera = camera
        self.rawCapture = rawCapture

        # Update the GUI with a status message
        self.MainWindow.lineEdit.setText(
            "Camera initialized! Image processing system running"
        )

        # Update the GUI with the actual camera settings
        self.MainWindow.lineEdit_shutter.setText(str(self.camera.shutter_speed))
        self.MainWindow.lineEdit_frame.setText(str(camera.framerate))
        self.MainWindow.lineEdit_brightness.setText(str(camera.brightness))
        self.MainWindow.lineEdit_exposure_comp.setText(
            str(camera.exposure_compensation)
        )
        self.MainWindow.lineEdit_iso.setText(str(camera.iso))
        self.MainWindow.lineEdit_saturation.setText(str(camera.saturation))

    # Stop the camera and update the GUI with a status message
    def stop_camera(self):
        if self.camera:
            self.camera.close()
            self.MainWindow.lineEdit.setText("Camera stopped & settings applied")

    # Capture an image from the camera and store it to self.image_live
    def img_capture(self):
        self.camera.capture(self.rawCapture, format="bgr")
        self.image_live = self.rawCapture.array
        self.rawCapture.truncate(0)

    # Take camera capture and display live on "Camera" tab
    def live_image(self):
        # Time printouts can be used for runtime optimization which directly translates to framerate of images
        # A = datetime.datetime.now()

        # Capture an image
        self.img_capture()

        # Determine the scale factor based on the camera resolution
        if self.W == 640 and self.H == 480:
            scale = 1
        elif self.W == 1280 and self.H == 720:
            scale = 2
        elif self.W == 1920 and self.H == 1080:
            scale = 3
        elif self.W == 2560 and self.H == 1440:
            scale = 4
        elif self.W == 4056 and self.H == 3040:
            scale = 6
        else:
            scale = 1

        # Resize the image to fit the GUI screen
        imR = cv2.resize(self.image_live, (int(self.W / scale), int(self.H / scale)))

        # Set the image frame to the proper position and size on the window, if not already done
        if not self.FRAMES_INIT:
            self.MainWindow.image_frame.move(125, 60)
            self.MainWindow.image_frame.resize(int(self.W / scale), int(self.H / scale))

        # Convert the image from BGR to RGB format
        imBGR2RGB = cv2.cvtColor(imR, cv2.COLOR_BGR2RGB)

        # Create a QImage to be displayed in the GUI
        imGUI = QtGui.QImage(
            imBGR2RGB.data,
            imBGR2RGB.shape[1],
            imBGR2RGB.shape[0],
            imBGR2RGB.shape[1] * 3,
            QtGui.QImage.Format_RGB888,
        )

        # Set the QPixmap for the image frame in the GUI
        self.MainWindow.image_frame.setPixmap(QtGui.QPixmap.fromImage(imGUI))

        # B = datetime.datetime.now()
        # print("Live image runtime: "+str(B-A))

# Convert camera image to beam profile (rainbow map) and display on GUI
# Compute metrics of the beam (centroid, D4σ)
    def beam(self):
        # Time printouts can be used for runtime optimization which directly translates to framerate of images
        # A = datetime.datetime.now()

        # Convert the live image to grayscale for intensity profiling
        image = cv2.cvtColor(self.image_live, cv2.COLOR_BGR2GRAY)

        # Compute the centroid and D4σ in pixel values if the image is not empty
        MOM = cv2.moments(image)
        if MOM["m00"] != 0:
            centroid_x = MOM["m10"] / MOM["m00"]
            centroid_y = MOM["m01"] / MOM["m00"]

            # Calculate the D4σ in physical units (using pixel size in microns)
            d4x = (
                self.pixel_um
                * 4
                * math.sqrt(abs(MOM["m20"] / MOM["m00"] - centroid_x**2))
            )
            d4y = (
                self.pixel_um
                * 4
                * math.sqrt(abs(MOM["m02"] / MOM["m00"] - centroid_y**2))
            )
        else:
            centroid_x = self.mask_x
            centroid_y = self.mask_y
            d4x = 0
            d4y = 0

        # Update the GUI with centroid and D4σ values
        self.MainWindow.label_centroid.setText(
            "Centroid x,y: " + str(round(centroid_x)) + ", " + str(round(centroid_y))
        )
        self.MainWindow.lcdNumber_dx.display(round(d4x))
        self.MainWindow.lcdNumber_dy.display(round(d4y))

        # Invert the grayscale image and apply the rainbow colormap
        dark_pixel_threshold = 0
        num_dark_pixels = np.sum(image <= dark_pixel_threshold)
        image_n = 255 - image
        beam = cv2.applyColorMap(image_n, cv2.COLORMAP_RAINBOW)

        # Save all data if the SAVE_NOW flag is set by the save button, then reset the flag
        if self.SAVE_NOW:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_prefix = self.MainWindow.lineEdit_savePrefix.text()
            filename6 = save_prefix + "_entered_info_" + timestamp + ".txt"
            savepath = os.path.join(os.getcwd(), "saves" + timestamp)

            # If the savepath directory doesn't exist, create it
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            # Define filenames for different types of data to be saved
            filename1 = save_prefix + "camera_" + timestamp + ".png"
            filename2 = save_prefix + "beam_" + timestamp + ".png"
            filename3 = save_prefix + "stats_" + timestamp + ".csv"
            filename4 = save_prefix + "x_profile_" + timestamp + ".png"
            filename5 = save_prefix + "y_profile_" + timestamp + ".png"

            # Save the live image and beam profile as PNG images
            cv2.imwrite(os.path.join(savepath, filename1), self.image_live)
            cv2.imwrite(os.path.join(savepath, filename2), beam)

            # Create and write statistics to a CSV file
            statsfile = open(os.path.join(savepath, filename3), "w")
            statsfile.write("Image width (px), height (px)\n")
            statsfile.write(str(self.W) + "," + str(self.H) + "\n")
            statsfile.write("Centroid x (px), y (px)\n")
            statsfile.write(str(centroid_x) + "," + str(centroid_y) + "\n")
            statsfile.write("D4σ x, y\n")
            statsfile.write(str(d4x) + "," + str(d4y) + "\n")
            max_pixel = np.amax(image)
            min_pixel = np.amin(image)
            statsfile.write("Number of dark pixels (threshold: {})\n".format(dark_pixel_threshold))
            statsfile.write(str(num_dark_pixels) + "\n")
            total_pixel_counts = np.sum(image)
            average_pixel_count = np.mean(image)
            statsfile.write("Max pixel value\n")
            statsfile.write(str(max_pixel) + "\n")
            statsfile.write("Min pixel value\n")
            statsfile.write(str(min_pixel) + "\n")
            statsfile.write("Total pixel counts\n")
            statsfile.write(str(total_pixel_counts) + "\n")
            statsfile.write("Average pixel count\n")
            statsfile.write(str(average_pixel_count) + "\n")
            statsfile.close()

            # Save additional information entered by the user in a text file
            with open(os.path.join(savepath, filename6), "w") as small_text_file:
                small_text_file.write(self.MainWindow.plainTextEdit_smallText.toPlainText())

            # Generate and save x-axis and y-axis beam profiles as PNG images
            x_prof = image[round(centroid_y), :]
            plt.plot(range(len(x_prof)), x_prof)
            plt.title("Beam profile along x-axis at y-centroid")
            plt.xlim(0, len(x_prof) - 1)
            plt.ylim(0, 255)
            plt.xlabel("Pixel")
            plt.ylabel("Intensity")
            plt.savefig(os.path.join(savepath, filename4))
            plt.close("all")

            y_prof = image[:, round(centroid_x)]
            plt.plot(range(len(y_prof)), y_prof)
            plt.title("Beam profile along y-axis at x-centroid")
            plt.xlim(0, len(y_prof) - 1)
            plt.ylim(0, 255)
            plt.xlabel("Pixel")
            plt.ylabel("Intensity")
            plt.savefig(os.path.join(savepath, filename5))
            plt.close("all")

            # Save pixel intensity data in a text file
            output_file = "pixel_data.txt"
            # Open the output file for writing pixel intensity data
            with open(os.path.join(savepath, output_file), "w") as f:
                # Write the header for the output file
                f.write("{:<10}{:<25}{:<25}\n".format("Pixel #", "X-axis Intensity Value", "Y-axis Intensity Value"))

                # Iterate through the pixel intensities and write them to the output file
                num_pixels = max(len(x_prof), len(y_prof))
                for i in range(num_pixels):
                    x_intensity = x_prof[i] if i < len(x_prof) else ""
                    y_intensity = y_prof[i] if i < len(y_prof) else ""
                    f.write("{:<10}{:<25}{:<25}\n".format(i, x_intensity, y_intensity))

            # Update the GUI's info bar depending on the logging status
            if not self.LOGGING:
                self.MainWindow.lineEdit.setText("Data saved to: " + savepath)
                self.SAVE_NOW = False
            else:
                self.MainWindow.lineEdit.setText("Data logging to: " + savepath)

        
        # Round the centroid and D4σ values to integers
        d4x, d4y, centroid_x, centroid_y = round(d4x), round(d4y), round(centroid_x), round(centroid_y)

        # Draw centroid lines on the beam profile image
        cv2.line(beam, (centroid_x, 0), (centroid_x, self.H), (0, 0, 0), thickness=5)
        cv2.line(beam, (0, centroid_y), (self.W, centroid_y), (0, 0, 0), thickness=5)

        # Determine the scale factor for downsampling the image to fit on the GUI screen
        if self.W == 640 and self.H == 480:
            scale = 1
        elif self.W == 1280 and self.H == 720:
            scale = 2
        elif self.W == 1920 and self.H == 1080:
            scale = 3
        elif self.W == 2560 and self.H == 1440:
            scale = 4
        elif self.W == 4056 and self.H == 3040:
            scale = 6
        else:
            scale = 1

        # Resize the beam profile image according to the scale factor
        beam_R = cv2.resize(beam, (int(self.W / scale), int(self.H / scale)))
        beam_R = cv2.cvtColor(beam_R, cv2.COLOR_BGR2RGB)

        # Draw the aperture mask circle on the resized beam profile image
        beam_R = cv2.circle(beam_R, (round(self.mask_x / scale), round(self.mask_y / scale)), int(self.mask_r / scale), (0, 0, 0), 2)

        # Set the image to the proper position on the window if not already done
        if not self.FRAMES_INIT:
            self.MainWindow.beam_frame.move(125, 60)
            self.MainWindow.beam_frame.resize(int(self.W / scale), int(self.H / scale))
            self.FRAMES_INIT = True

        # Convert the resized beam profile image to a QImage for display on the GUI
        imGUI = QtGui.QImage(beam_R.data, beam_R.shape[1], beam_R.shape[0], beam_R.shape[1] * 3, QtGui.QImage.Format_RGB888)
        self.MainWindow.beam_frame.setPixmap(QtGui.QPixmap.fromImage(imGUI))
        # B = datetime.datetime.now()
        # print("Beam runtime: "+str(B-A))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
