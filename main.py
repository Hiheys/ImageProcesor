import sys
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, QFileDialog,
                             QVBoxLayout, QWidget, QComboBox, QHBoxLayout, QSpinBox, QCheckBox,
                             QSlider)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QIcon
from PyQt5.QtCore import Qt
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import cv2
import numpy as np

def load_image(image_path):
    """Load an image from the specified path."""
    return Image.open(image_path)

def save_image(image, path):
    """Save an image to the specified path."""
    image.save(path)
    print(f"Image saved at: {path}")

def resize_image(image, size):
    """Resize the image to the specified size."""
    return image.resize(size, Image.LANCZOS)

def apply_filter(image, filter_type):
    """Apply a filter to the image."""
    if filter_type == 'BLUR':
        return image.filter(ImageFilter.BLUR)
    elif filter_type == 'CONTOUR':
        return image.filter(ImageFilter.CONTOUR)
    elif filter_type == 'DETAIL':
        return image.filter(ImageFilter.DETAIL)
    elif filter_type == 'EDGE_ENHANCE':
        return image.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_type == 'GAUSSIAN_BLUR':
        return image.filter(ImageFilter.GaussianBlur(5))
    elif filter_type == 'SHARPEN':
        return image.filter(ImageFilter.SHARPEN)
    elif filter_type == 'SMOOTH':
        return image.filter(ImageFilter.SMOOTH)
    elif filter_type == 'EMBOSS':
        return image.filter(ImageFilter.EMBOSS)
    else:
        return image

def convert_to_grayscale(image):
    """Convert the image to grayscale."""
    return ImageOps.grayscale(image)

def adjust_brightness(image, factor):
    """Adjust the brightness of the image."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    """Adjust the contrast of the image."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_saturation(image, factor):
    """Adjust the color saturation of the image."""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def adjust_sharpness(image, factor):
    """Adjust the sharpness of the image."""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def detect_edges(image_path, output_path):
    """Detect edges in the image and save the result."""
    image = cv2.imread(image_path, 0)
    edges = cv2.Canny(image, 100, 200)
    cv2.imwrite(output_path, edges)
    print(f"Edges saved at: {output_path}")

def apply_sobel_filter(image_path, output_path):
    """Apply Sobel filter to the image and save the result."""
    image = cv2.imread(image_path, 0)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    cv2.imwrite(output_path, sobel)
    print(f"Sobel filter result saved at: {output_path}")

def rotate_image(image, angle):
    """Rotate the image by the specified angle."""
    return image.rotate(angle, expand=True)

def flip_image(image, direction):
    """Flip the image horizontally or vertically."""
    if direction == 'Horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == 'Vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return image

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ImageProcessor')
        self.setGeometry(100, 100, 1000, 800)
        
 
        self.setWindowIcon(QIcon('app-src\icon.png'))
        
        self.setStyleSheet("background-color: #2e2e2e; color: white;")
        
        self.imageLabel = QLabel('No image loaded')
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setStyleSheet("QLabel {background-color: #1e1e1e; padding: 10px;}")
        self.imageLabel.setAcceptDrops(True)
        self.imageLabel.mousePressEvent = self.loadImageFromClick
        self.imageLabel.dragEnterEvent = self.dragEnterEvent
        self.imageLabel.dropEvent = self.dropEvent
        
        self.loadButton = self.createButton('Load Image', self.loadImage)
        self.saveButton = self.createButton('Save Image', self.saveImage)
        
        self.filterComboBox = QComboBox()
        self.filterComboBox.addItems(['None', 'BLUR', 'CONTOUR', 'DETAIL', 'EDGE_ENHANCE', 'GAUSSIAN_BLUR', 'SHARPEN', 'SMOOTH', 'EMBOSS'])
        self.filterComboBox.setStyleSheet("QComboBox {background-color: #444444; color: white;}")
        self.filterComboBox.currentTextChanged.connect(self.updateImage)

        self.resizeWidthSpinBox = QSpinBox()
        self.resizeWidthSpinBox.setRange(1, 5000)
        self.resizeWidthSpinBox.setValue(200)
        self.resizeWidthSpinBox.setStyleSheet("QSpinBox {background-color: #444444; color: white;}")
        self.resizeWidthSpinBox.valueChanged.connect(self.updateImage)
        
        self.resizeHeightSpinBox = QSpinBox()
        self.resizeHeightSpinBox.setRange(1, 5000)
        self.resizeHeightSpinBox.setValue(200)
        self.resizeHeightSpinBox.setStyleSheet("QSpinBox {background-color: #444444; color: white;}")
        self.resizeHeightSpinBox.valueChanged.connect(self.updateImage)
        
        self.resizeButton = self.createButton('Resize Image', self.updateImage)
        
        self.grayscaleCheckBox = QCheckBox('Convert to Grayscale')
        self.grayscaleCheckBox.setStyleSheet("QCheckBox {color: white;}")
        self.grayscaleCheckBox.stateChanged.connect(self.updateImage)
        
        self.brightnessSlider = self.createSlider(self.updateImage)
        self.contrastSlider = self.createSlider(self.updateImage)
        self.saturationSlider = self.createSlider(self.updateImage)
        self.sharpnessSlider = self.createSlider(self.updateImage)
        
        self.detectEdgesButton = self.createButton('Detect Edges', self.detectEdges)
        self.applySobelButton = self.createButton('Apply Sobel Filter', self.applySobelFilter)
        
        self.rotateButton = self.createButton('Rotate 90°', self.rotateImage)
        self.flipComboBox = QComboBox()
        self.flipComboBox.addItems(['None', 'Horizontal', 'Vertical'])
        self.flipComboBox.setStyleSheet("QComboBox {background-color: #444444; color: white;}")
        self.flipComboBox.currentTextChanged.connect(self.updateImage)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.loadButton)
        self.layout.addWidget(self.imageLabel)
        self.layout.addWidget(self.filterComboBox)
        
        resizeLayout = QHBoxLayout()
        resizeLayout.addWidget(QLabel('Width:'))
        resizeLayout.addWidget(self.resizeWidthSpinBox)
        resizeLayout.addWidget(QLabel('Height:'))
        resizeLayout.addWidget(self.resizeHeightSpinBox)
        resizeLayout.addWidget(self.resizeButton)
        
        self.layout.addLayout(resizeLayout)
        self.layout.addWidget(self.grayscaleCheckBox)
        
        brightnessLayout = QHBoxLayout()
        brightnessLayout.addWidget(QLabel('Brightness:'))
        brightnessLayout.addWidget(self.brightnessSlider)
        
        contrastLayout = QHBoxLayout()
        contrastLayout.addWidget(QLabel('Contrast:'))
        contrastLayout.addWidget(self.contrastSlider)
        
        saturationLayout = QHBoxLayout()
        saturationLayout.addWidget(QLabel('Saturation:'))
        saturationLayout.addWidget(self.saturationSlider)
        
        sharpnessLayout = QHBoxLayout()
        sharpnessLayout.addWidget(QLabel('Sharpness:'))
        sharpnessLayout.addWidget(self.sharpnessSlider)
        
        self.layout.addLayout(brightnessLayout)
        self.layout.addLayout(contrastLayout)
        self.layout.addLayout(saturationLayout)
        self.layout.addLayout(sharpnessLayout)
        
        rotateLayout = QHBoxLayout()
        rotateLayout.addWidget(QLabel('Rotate:'))
        rotateLayout.addWidget(self.rotateButton)
        
        flipLayout = QHBoxLayout()
        flipLayout.addWidget(QLabel('Flip:'))
        flipLayout.addWidget(self.flipComboBox)

        self.layout.addLayout(rotateLayout)
        self.layout.addLayout(flipLayout)
        
        self.layout.addWidget(self.detectEdgesButton)
        self.layout.addWidget(self.applySobelButton)
        self.layout.addWidget(self.saveButton)
        
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.resetImage()  # Reset all parameters on startup

    def createButton(self, text, func):
        button = QPushButton(text)
        button.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                padding: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:pressed {
                background-color: #333333;
            }
        """)
        button.clicked.connect(func)
        return button

    def createSlider(self, func, min_value=0, max_value=100, default_value=50):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_value, max_value)
        slider.setValue(default_value)
        slider.setStyleSheet("""
            QSlider {
                background-color: #2e2e2e;
            }
            QSlider::groove:horizontal {
                height: 10px;
                background: #444444;
                margin: 0;
            }
            QSlider::handle:horizontal {
                background: #888888;
                width: 20px;
                margin: -2px 0;
            }
        """)
        slider.valueChanged.connect(func)
        return slider

    def loadImageFromClick(self, event):
        self.loadImage()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.loadImage(file_path)

    def loadImage(self, file_path=None):
        if not file_path:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif)', options=options)
        if file_path:
            self.imagePath = file_path
            self.originalImage = load_image(file_path)
            self.image = self.originalImage.copy()
            self.displayImage()
            self.resetSliders()  # Reset sliders to default values

    def saveImage(self):
        if self.image:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            outputPath, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif)', options=options)
            if outputPath:
                save_image(self.image, outputPath)

    def updateImage(self):
        if self.originalImage:
            self.image = self.originalImage.copy()

            # Apply filter
            filter_type = self.filterComboBox.currentText()
            if filter_type != 'None':
                self.image = apply_filter(self.image, filter_type)

            # Resize image
            width = self.resizeWidthSpinBox.value()
            height = self.resizeHeightSpinBox.value()
            self.image = resize_image(self.image, (width, height))

            # Convert to grayscale
            if self.grayscaleCheckBox.isChecked():
                self.image = convert_to_grayscale(self.image)

            # Adjust brightness, contrast, saturation, sharpness
            brightness = self.brightnessSlider.value() / 50
            self.image = adjust_brightness(self.image, brightness)

            contrast = self.contrastSlider.value() / 50
            self.image = adjust_contrast(self.image, contrast)

            saturation = self.saturationSlider.value() / 50
            self.image = adjust_saturation(self.image, saturation)

            sharpness = self.sharpnessSlider.value() / 50
            self.image = adjust_sharpness(self.image, sharpness)

            # Flip image
            flip_direction = self.flipComboBox.currentText()
            if flip_direction != 'None':
                self.image = flip_image(self.image, flip_direction)

            self.displayImage()

    def detectEdges(self):
        if self.imagePath:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            outputPath, _ = QFileDialog.getSaveFileName(self, 'Save Edges Image', '', 'Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif)', options=options)
            if outputPath:
                detect_edges(self.imagePath, outputPath)

    def applySobelFilter(self):
        if self.imagePath:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            outputPath, _ = QFileDialog.getSaveFileName(self, 'Save Sobel Filter Image', '', 'Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif)', options=options)
            if outputPath:
                apply_sobel_filter(self.imagePath, outputPath)

    def rotateImage(self):
        if self.image:
            self.image = rotate_image(self.image, 90)
            self.displayImage()

    def displayImage(self):
        if self.image:
            qt_image = self.pil2pixmap(self.image)
            self.imageLabel.setPixmap(qt_image)
        else:
            self.imageLabel.setText("No image loaded")

    def pil2pixmap(self, im):
        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "BGRA")
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(qim)
        return pixmap

    def resetImage(self):
        self.originalImage = None
        self.image = None
        self.imageLabel.setText("No image loaded")
        self.filterComboBox.setCurrentText('None')
        self.resizeWidthSpinBox.setValue(200)
        self.resizeHeightSpinBox.setValue(200)
        self.grayscaleCheckBox.setChecked(False)
        self.resetSliders()
        self.flipComboBox.setCurrentText('None')

    def resetSliders(self):
        self.brightnessSlider.setValue(50)
        self.contrastSlider.setValue(50)
        self.saturationSlider.setValue(50)
        self.sharpnessSlider.setValue(50)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())
