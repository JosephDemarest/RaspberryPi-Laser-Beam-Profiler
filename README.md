# Laser Beam Profiler

This repository contains a Python-based laser beam profiler application that utilizes the **OpenCV** library for image processing and **PyQt5** for GUI implementation. The software captures images from a connected camera, applies a rainbow colormap to visualize the beam profile, computes beam metrics, and saves the results.

## 🚀 Features

- Live camera feed display
- Rainbow colormap visualization of the laser beam profile
- Centroid and D4σ computation (beam width metrics)
- Adjustable circular aperture
- Saving camera images, beam images, beam statistics, and profiles
- Data logging capability

## 📦 Requirements
Software
- Python 3.x
- OpenCV
- PyQt5
- NumPy
- Matplotlib
Hardware
- Raspberry Pi 4
- Raspberry PI HQ Camera

## ⚙️ Installation

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`

## 📚 Usage

1. Run the application with `python BeamProfiler.py`
2. Use the GUI to adjust the aperture mask settings and capture images.
3. Save images, statistics, and beam profiles using the "Save" button or enable logging.

## 📂 Application Structure

The application consists of the following components:

1. Camera configuration and image capture
2. Image processing, including resizing and BGR to RGB conversion for GUI display
3. Aperture mask configuration based on user input
4. Computation of beam metrics, including centroid and D4σ
5. Saving and logging data, including images, statistics, and profiles
6. GUI implementation using PyQt5

## 🤝 Contributing

Please feel free to create issues or submit pull requests for any improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License.
