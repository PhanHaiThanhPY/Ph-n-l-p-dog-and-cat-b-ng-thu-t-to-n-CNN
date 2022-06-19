import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from GiaoDien import Ui_MainWindow
from PyQt5.QtGui import QPixmap
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class_ = ['cat','dog']
model = keras.models.load_model('model_keras.h5')
class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)
        #Khai báo nút ấn
        self.uic.pushButton.clicked.connect(self.linkto)
    def linkto(self):
        #tim đường dẫn
        link = QFileDialog.getOpenFileName(filter='*.jpg *.png')
        #Mở hình lên
        self.uic.LoadImage.setPixmap(QPixmap(link[0]))
        #Hiện đường truyền lên
        self.uic.lineEdit.setText(link[0])
        #khai báo nút Scan ảnh
        self.uic.pushButton_2.clicked.connect(self.Scanpic)
        global linking
        linking = link[0]

    def Scanpic(self):
      img_arr = cv2.imread(linking)
      img_arr = cv2.resize(img_arr, (150, 150))  # resize ảnh về kích thước 150

      img_arr = img_arr.astype('float32') / 255.0
      img_arr = img_arr.reshape(1, img_arr.shape[0], img_arr.shape[1], 3)

      y_pred = model.predict(img_arr)
      print(y_pred)
      self.uic.label.setText(class_[np.argmax(y_pred)])
      acc_ = round(np.amax(y_pred) * 100, 2)
      self.uic.label_2.setText(str(acc_) + ' %')

    def show(self):
        self.main_win.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())