from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QCursor

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np
import os
import time
import shutil
import nibabel as nib
from PIL import Image
import json
import EF
import contour_extraction

StyleSheet = '''
    QPushButton {
        background-color: rgb(165, 20, 23);
        color: #FFFFFF;
        padding: 2px;
        font: bold 18px;
        border-style: solid;
        border-width: 2px;
        border-radius: 2px;
        border-color: rgb(165, 20, 23);
    }
    QPushButton:hover {
        background-color: rgb(240, 240, 240);
        color: rgb(165, 20, 23);
        border-style: solid;
        border-width: 2px;
        border-radius: 3px;
        border-color: rgb(165, 20, 23);
    }
    QSlider:handle {
        background-color: rgb(165, 20, 23);
        border-color: rgb(165, 20, 23);
    }
    
    QSlider::handle:hover {
        background-color: #8c1113;
        border-color: #8c1113;
    }
'''
# Read and sort directory names
input_names = np.array(os.listdir("Input/"))
input_names.sort()
name_idx = 0

# Load parameters
parameter_dict = json.loads(open('parameters.json', 'r').read())
parameter_dict["lv_loc"] = []


# Convert dicom to gif for display
def dicom_to_gif(input_path, output_folder):
    ds_sa_ed_img = nib.load(input_path).get_fdata()
    for i in range(np.shape(ds_sa_ed_img)[-1]):
        img = ds_sa_ed_img[:, :, i].astype('uint16')
        if np.max(img) != 0:
            img = ((img / np.max(img)) * 255).astype('uint16')
        im = Image.fromarray(img)
        im.save(output_folder + str(i + 1) + ".gif")


def delete_all_files(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


all_image_pts = []


def display_images(window, size, input_folder, row, height_multiplier=2):
    # Read and sort directory names
    gif_names = np.array(os.listdir(input_folder))
    gif_names.sort()
    pixmap = QPixmap(input_folder + gif_names[0])

    # Define screen width and height
    screen_width = size.width()
    screen_height = size.height()

    # Resize image
    width = pixmap.width()
    height = pixmap.height()
    if (width + 3) * len(gif_names) + 50 > screen_width:
        width = (screen_width - 50) // len(gif_names) - 5
    if height * height_multiplier + 250 > screen_height:
        height = (screen_height - 250) // height_multiplier - 2
    pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
    window.setGeometry(50, 50, len(gif_names) * pixmap.width() + 50, (pixmap.height() + 80) * height_multiplier)
    # window.setFixedWidth(len(gif_names) * pixmap.width() + 50)
    # window.setFixedHeight(pixmap.height() * height_multiplier + 200)

    # Display images
    for i in range(len(gif_names)):
        label = QLabel(window)
        pixmap = QPixmap(input_folder + gif_names[i])
        pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        label.move(i * (pixmap.width() + 3) + 25 - 3 * ((len(gif_names) - 1) // 2), (row - 1) * (pixmap.height() + 5))
        all_image_pts.append(label)
    return len(gif_names) * pixmap.width() + 50, (pixmap.height() * 2 + 5), QPixmap(
        input_folder + gif_names[len(gif_names) // 2])


# Add buttons
all_button_pts = []


def clear_all(window):
    global all_button_pts, all_image_pts, all_slider_pts, all_label_pts, row_slider_count, col_slider_count
    for b in all_button_pts:
        b.deleteLater()
        all_button_pts = []
    for img in all_image_pts:
        img.clear()
        all_image_pts = []
    for slider in all_slider_pts:
        slider.deleteLater()
        all_slider_pts = []
    for label in all_label_pts:
        label.clear()
        all_label_pts = []
    row_slider_count = 1
    col_slider_count = 1
    window.hide()


class ClickTracker(QWidget):
    def __init__(self, img, geometry):
        super(ClickTracker, self).__init__()
        self.chosen_points = []
        self.img = img
        self.geometry = geometry
        self.initUI()

    def mouseReleaseEvent(self, cursor_event):
        parameter_dict["lv_loc"] = [cursor_event.pos().x(), cursor_event.pos().y()]
        print("The new LV centroid: ({}, {}) is successfully stored".format(cursor_event.pos().x(),
                                                                            cursor_event.pos().y()))
        self.chosen_points.append(cursor_event.pos())
        self.update()

    def paintEvent(self, paint_event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.img)
        pen = QtGui.QPen()
        pen.setWidth(3)
        pen.setColor(QColor(255, 0, 0, 255))
        painter.setPen(pen)
        try:
            painter.drawPoint(self.chosen_points[-1])
            self.setWindowTitle('Saved')
        except:
            pass

    def initUI(self):
        self.setGeometry(self.geometry[0], self.geometry[1], self.geometry[2], self.geometry[3])
        self.setWindowTitle('Centroid')
        self.setWindowIcon(QtGui.QIcon('logo.png'))


def run_next(window):
    if name_idx < len(input_names):
        QMessageBox.about(window, "Notice", "Results are saved in the Output folder. "
                          + "Click Ok to run the next file: " + input_names[name_idx])
        window.close()
        run_unet_segmentation(input_names[name_idx])
        run_unet_gui(app, window, input_names[name_idx])
    else:
        QMessageBox.about(window, "Notice", "Results are saved in the Output folder. "
                          + "Click Ok to finish.")
        window.close()


def add_button(app, img_width, img_height, window, row, text, name, mid_img, width_multi=1.0):
    global all_button_pts
    # Creating push button widget
    button = QPushButton(text, window)
    button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
    button.setFont(QFont('sans-serif', 18))
    # Setting geometry
    button.setGeometry(img_width // 2 - 100,
                       img_height + int(width_multi * 50 + (150 * (width_multi - 1))) + (row - 1) * 50,
                       int(200 * width_multi), 40)

    # Clicked event
    button.clicked.connect(lambda: click_method(text))
    all_button_pts.append(button)
    screen = app.primaryScreen()
    size = screen.size()
    click_window = ClickTracker(mid_img, [size.width() // 2, size.height() // 2, mid_img.width(), mid_img.height()])

    def click_method(button_name):
        global name_idx, parameter_dict, SA_LV_mask_ED, SA_img_ED, slice_locs_trimed
        if button_name == "Finish Segmentation":
            name_idx += 1
            clear_all(window)
            run_next(window)
        elif button_name == "EF Postprocessing":
            delete_all_files('sa_ed_seg_ef_gif/')
            delete_all_files('sa_ed_ef_gif/')
            parameter_dict["name"] = input_names[name_idx]
            with open('parameters.json', 'w', encoding='utf-8') as f:
                json.dump(parameter_dict, f, ensure_ascii=False, indent=4)
            SA_LV_mask_ED, SA_img_ED, slice_locs_trimed = EF.run_ef_segmentation()
            contour_extraction.extract_contour(SA_LV_mask_ED, SA_img_ED, slice_locs_trimed, "Output/example2/")

            clear_all(window)
            run_ef_gui(app, window, name)
            window.show()
        elif button_name == "Run EF Postprocessing":
            delete_all_files('sa_ed_seg_ef_gif/')
            delete_all_files('sa_ed_ef_gif/')
            parameter_dict["name"] = input_names[name_idx]
            with open('parameters.json', 'w', encoding='utf-8') as f:
                json.dump(parameter_dict, f, ensure_ascii=False, indent=4)
            SA_LV_mask_ED, SA_img_ED, slice_locs_trimed = EF.run_ef_segmentation()

            clear_all(window)
            run_ef_gui(app, window, name)
            window.show()
        elif button_name == "LV Manual Localization":
            click_window.show()
        elif button_name == "Finish EF Postprocessing":
            name_idx += 1
            contour_extraction.extract_contour(SA_LV_mask_ED, SA_img_ED, slice_locs_trimed, "Output/example2/")
            clear_all(click_window)
            clear_all(window)
            run_next(window)


def run_unet_segmentation(name):
    # Clear the ukbb_cardiac directory
    delete_all_files('ukbb_cardiac/demo_image/1')

    # Copy the input file to ukbb_cardiac dir
    shutil.copy("Input/" + name, 'ukbb_cardiac/demo_image/1')
    os.rename('ukbb_cardiac/demo_image/1/' + name, 'ukbb_cardiac/demo_image/1/sa.nii.gz')

    # Deploy the segmentation network
    CUDA_VISIBLE_DEVICES = 0
    print('Deploying the segmentation network ...')
    os.system('python3 ukbb_cardiac/common/deploy_network.py --seq_name sa --data_dir ukbb_cardiac/demo_image '
              '--model_path ukbb_cardiac/trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))

    # For display images
    delete_all_files('sa_ed_gif/')
    dicom_to_gif('ukbb_cardiac/demo_image/1/sa_ED.nii.gz', 'sa_ed_gif/')  # Original dicom image to gif images
    delete_all_files('sa_ed_seg_gif/')
    dicom_to_gif('ukbb_cardiac/demo_image/1/seg_sa_ED.nii.gz',
                 'sa_ed_seg_gif/')  # Dicom segmentation image to gif images

    # Save output files to Output folder
    try:
        os.mkdir('Output/' + name.split(".")[0])
    except FileExistsError:
        pass
    for output_file in os.listdir("ukbb_cardiac/demo_image/1"):
        shutil.copy('ukbb_cardiac/demo_image/1/' + output_file, 'Output/' + name.split(".")[0])


def run_unet_gui(app, window, name):
    screen = app.primaryScreen()
    size = screen.size()

    window.setWindowIcon(QtGui.QIcon('logo.png'))
    window.setWindowTitle("U-Net Segmentation -" + name.split(".")[0])

    # Display images
    _, _, mid_img = display_images(window, size, 'sa_ed_gif/', 1)
    img_width, img_height, _ = display_images(window, size, 'sa_ed_seg_gif/', 2)

    # Add buttons
    add_button(app, img_width, img_height, window, 1, "Finish Segmentation", name, mid_img)
    add_button(app, img_width, img_height, window, 2, "EF Postprocessing", name, mid_img)

    window.show()
    app.exec()


row_slider_count = 1
col_slider_count = 1

all_slider_pts = []
all_label_pts = []


def add_sliders(para_name, val_list, slider_rows, slider_width, img_height):
    global row_slider_count, col_slider_count

    init_val = val_list[0]
    min_val = val_list[1]
    max_val = val_list[2]
    step = val_list[3]

    slider = QtWidgets.QSlider(window)
    slider.setOrientation(QtCore.Qt.Horizontal)
    slider.setFocusPolicy(Qt.StrongFocus)
    slider.setSingleStep(step)
    slider.setRange(min_val, max_val)
    slider.setValue(init_val)
    slider.setGeometry((slider_width + 32) * (row_slider_count - 1) + 60, 80 * col_slider_count + img_height,
                       slider_width, 20)
    slider.valueChanged.connect(lambda: update_parameter(para_name))

    label = QLabel(para_name + ": " + str(init_val), window)
    label.setGeometry((slider_width + 32) * (row_slider_count - 1) + 65, 80 * col_slider_count + 20 + img_height,
                      slider_width - 20, 25)
    label.setFont(QFont('sans-serif', 12))
    if row_slider_count == slider_rows:
        row_slider_count = 0
        col_slider_count += 1
    row_slider_count += 1
    all_slider_pts.append(slider)
    all_label_pts.append(label)

    def update_parameter(para_name):
        global parameter_dict
        slider_val = slider.value()
        if slider_val % 2 == 0 and step == 2:
            slider_val += 1
        label.setText(para_name + ": " + str(slider_val))
        parameter_dict[para_name][0] = slider_val


def run_ef_gui(app, window, name):
    window.setWindowTitle("EF Postprocessing -" + name.split(".")[0])

    screen = app.primaryScreen()
    size = screen.size()

    # Display images
    _, _, mid_img = display_images(window, size, "sa_ed_ef_gif/", 1, 4)
    img_width, img_height, _ = display_images(window, size, 'sa_ed_seg_ef_gif/', 2, 4)

    # Add sliders
    slide_rows = len(parameter_dict) // 3
    slider_width = (img_width - 200) // slide_rows
    for para_name, val_list in parameter_dict.items():
        if len(val_list) == 4:
            add_sliders(para_name, val_list, slide_rows, slider_width, img_height)

    add_button(app, img_width, img_height + col_slider_count * 70, window, 1, "Run EF Postprocessing", name, mid_img,
               1.3)
    add_button(app, img_width, img_height + col_slider_count * 70, window, 2, "LV Manual Localization", name, mid_img,
               1.3)
    add_button(app, img_width, img_height + col_slider_count * 70, window, 3, "Finish EF Postprocessing", name, mid_img,
               1.3)


if __name__ == '__main__':
    # run_unet_segmentation(input_names[name_idx])

    # Add window and properties
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    window = QWidget()

    run_unet_gui(app, window, input_names[name_idx])
