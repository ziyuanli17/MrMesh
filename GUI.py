from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QCursor

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np
import string
import os
import time
import shutil
import nibabel as nib
from PIL import Image
import json
import EF
import contour_extraction
import generate_mesh

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
slice_locs = parameter_dict["slice_locs"]

mesh_parameter_dict = json.loads(open('mesh_parameters.json', 'r').read())


# Convert dicom to gif for display
def dicom_to_gif(input_path, output_folder):
    ds_sa_ed_img = nib.load(input_path).get_fdata()
    for i in range(np.shape(ds_sa_ed_img)[-1]):
        img = ds_sa_ed_img[:, :, i].astype('uint16')
        if np.max(img) != 0:
            img = ((img / np.max(img)) * 255).astype('uint16')
        im = Image.fromarray(img)
        im.save(output_folder + str(string.ascii_lowercase[i]) + ".gif")


def read_ed_frames():
    global slice_locs
    ds_sa_ed_img = nib.load('ukbb_cardiac/demo_image/1/sa_ED.nii.gz').get_fdata()
    ds_sa_ed_seg = nib.load('ukbb_cardiac/demo_image/1/seg_sa_ED.nii.gz').get_fdata()
    sa_seg_ed = []
    sa_img_ed = []
    locs_trimmed = []

    for i in range(np.shape(ds_sa_ed_img)[-1]):
        img = ds_sa_ed_img[:, :, i].astype('uint16')
        seg = ds_sa_ed_seg[:, :, i].astype('uint16')
        if np.max(img) != 0:
            img = ((img / np.max(img)) * 255).astype('uint16')
        if len(np.where(img != 0)) // len(np.where(img == 0)) > 0.05:
            sa_img_ed.append(img)
            sa_seg_ed.append(seg)
            locs_trimmed.append(slice_locs[i])
    return sa_img_ed, sa_seg_ed, locs_trimmed


def generate_dummies():
    sa_img_ed, sa_seg_ed, locs_trimmed = read_ed_frames()
    seg_list = []
    for i in range(len(sa_img_ed)):
        seg = np.zeros((np.shape(sa_img_ed[0])), np.uint8)
        seg_list.append(seg)
    return sa_img_ed, seg_list, locs_trimmed


def delete_all_files(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


all_image_pts = []


def display_images(window, size, input_folder, row, height_multiplier=2, identifier=None, ver_shift=0, num_skip=0, hon_win_shift=False):
    gif_names = np.array(os.listdir(input_folder))
    # Read and sort directory names
    if identifier is not None:
        gif_names = gif_names[np.flatnonzero(np.core.defchararray.find(gif_names, identifier) != -1)]

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
    shift = 0
    if hon_win_shift:
        shift = screen_width - (len(gif_names) * pixmap.width() + 70)

    window.setGeometry(10+shift, 40, len(gif_names) * pixmap.width() + 50, (pixmap.height() + 80) * height_multiplier)
    # window.setFixedWidth(len(gif_names) * pixmap.width() + 50)
    # window.setFixedHeight(pixmap.height() * height_multiplier + 200)

    # Display images
    for i in range(len(gif_names) - num_skip):
        label = QLabel(window)
        pixmap = QPixmap(input_folder + gif_names[i])
        pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        label.move(i * (pixmap.width() + 3) + 25 - 3 * ((len(gif_names) - 1) // 2),
                   (row - 1) * (pixmap.height() + 5) + ver_shift)
        all_image_pts.append(label)
    return len(gif_names) * pixmap.width() + 50, (pixmap.height() * 2 + 5), QPixmap(
        input_folder + gif_names[len(gif_names) // 2])


# Add buttons
all_button_pts = []


def clear_all(window):
    global all_button_pts, all_image_pts, all_slider_pts, all_label_pts, row_slider_count, col_slider_count
    for b in all_button_pts:
        try:
            b.deleteLater()
            all_button_pts = []
        except RuntimeError:
            all_button_pts = []
            pass
    for img in all_image_pts:
        try:
            img.clear()
            all_image_pts = []
        except RuntimeError:
            all_image_pts = []
            pass
    for slider in all_slider_pts:
        try:
            slider.deleteLater()
            all_slider_pts = []
        except RuntimeError:
            all_slider_pts = []
            pass
    for label in all_label_pts:
        try:
            label.clear()
            all_label_pts = []
        except RuntimeError:
            all_label_pts = []
            pass
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
        msg = QMessageBox.information(window, "Notice", "Results are saved in the Output folder. "
                                      + "Click Ok to run the next file: " + input_names[name_idx],
                                      QMessageBox.Ok | QMessageBox.Cancel)
        if msg == QMessageBox.Ok:
            window.close()
            run_unet_segmentation(input_names[name_idx])
            run_unet_gui(app, window, input_names[name_idx])
        else:
            window.close()

    else:
        QMessageBox.about(window, "Notice", "Results are saved in the Output folder. "
                          + "Click Ok to finish.")
        window.close()
        exit(0)

show_process = False
ef_process_window = None


def add_button(app, img_width, img_height, window, row, text, name, mid_img, width_multi=1.0):
    global all_button_pts, ef_process_window
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

    if mid_img != 0:
        click_window = ClickTracker(mid_img, [size.width() // 2, size.height() // 2, mid_img.width(), mid_img.height()])
    mesh_window = QWidget()
    ef_process_window = QWidget()

    def click_method(button_name):
        global name_idx, parameter_dict, mesh_parameter_dict, SA_LV_mask_ED, SA_img_ED, slice_locs_trimed, show_process
        cur_name = input_names[name_idx].split(".")[0] + "/"
        if button_name == "Finish Segmentation":
            name_idx += 1
            clear_all(window)
            # contour_extraction.extract_contour(SA_LV_mask_ED, SA_img_ED, slice_locs_trimed, "Output/"+cur_name)
            run_next(window)
        elif button_name == "EF Postprocessing":
            delete_all_files('sa_ed_seg_ef_gif/')
            delete_all_files('sa_ed_ef_gif/')
            parameter_dict["name"] = input_names[name_idx]
            with open('parameters.json', 'w', encoding='utf-8') as f:
                json.dump(parameter_dict, f, ensure_ascii=False, indent=4)
            try:
                SA_LV_mask_ED, SA_img_ED, slice_locs_trimed = EF.run_ef_segmentation()
            except:
                SA_img_ED, SA_LV_mask_ED, slice_locs_trimed = generate_dummies()
                EF.save_gif(SA_LV_mask_ED, "sa_ed_seg_ef_gif/")
                EF.save_gif(SA_img_ED, "sa_ed_ef_gif/")

            # contour_extraction.extract_contour(SA_LV_mask_ED, SA_img_ED, slice_locs_trimed, "Output/" + cur_name)
            clear_all(window)
            run_ef_gui(app, window, name)
            window.show()
        elif button_name == "Run EF Postprocessing":
            delete_all_files('sa_ed_seg_ef_gif/')
            delete_all_files('sa_ed_ef_gif/')
            parameter_dict["name"] = input_names[name_idx]
            with open('parameters.json', 'w', encoding='utf-8') as f:
                json.dump(parameter_dict, f, ensure_ascii=False, indent=4)
            try:
                SA_LV_mask_ED, SA_img_ED, slice_locs_trimed = EF.run_ef_segmentation()
            except:
                SA_img_ED, SA_LV_mask_ED, slice_locs_trimed = generate_dummies()
                EF.save_gif(SA_LV_mask_ED, "sa_ed_seg_ef_gif/")
                EF.save_gif(SA_img_ED, "sa_ed_ef_gif/")

            clear_all(window)
            run_ef_gui(app, window, name)
            window.show()
            if show_process:
                run_ef_process_gui(app, ef_process_window, name)
                ef_process_window.show()

        elif button_name == "LV Manual Localization":
            click_window.show()
        elif button_name == "Finish EF Postprocessing":
            name_idx += 1
            contour_extraction.extract_contour(SA_LV_mask_ED, SA_img_ED, slice_locs_trimed, "Output/" + cur_name)
            clear_all(click_window)
            clear_all(window)
            run_next(window)

        elif button_name == "Mesh Generation":
            # Add window and properties
            run_mesh_gui(app, mesh_window, name)
            mesh_window.show()
        elif button_name == "Generate All Meshes":
            name_idx += 1
            with open('mesh_parameters.json', 'w', encoding='utf-8') as f:
                json.dump(mesh_parameter_dict, f, ensure_ascii=False, indent=4)
            try:
                contour_extraction.extract_contour(SA_LV_mask_ED, SA_img_ED, slice_locs_trimed, "Output/" + cur_name)
            except:
                sa_img_ed, sa_seg_ed, locs_trimmed = read_ed_frames()
                contour_extraction.extract_contour(sa_seg_ed, sa_img_ed, locs_trimmed, "Output/" + cur_name)
            generate_mesh.generate_meshes("Output/" + cur_name + "/")

            clear_all(mesh_window)
            clear_all(ef_process_window)
            clear_all(window)
            run_next(window)
            mesh_window.close()
            ef_process_window.close()
            show_process = False
        elif button_name == "Show EF Process":
            run_ef_process_gui(app, ef_process_window, name)
            ef_process_window.show()
            show_process = True


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
    add_button(app, img_width, img_height, window, 1, "EF Postprocessing", name, mid_img)
    add_button(app, img_width, img_height, window, 2, "Mesh Generation", name, mid_img)

    window.show()
    app.exec()


row_slider_count = 1
col_slider_count = 1

all_slider_pts = []
all_label_pts = []


def add_sliders(window, para_name, val_list, slider_rows, slider_width, img_height):
    global row_slider_count, col_slider_count

    init_val = val_list[0]
    min_val = val_list[1]
    max_val = val_list[2]
    step = val_list[3]
    multiplier = 1
    if isinstance(step, float):
        multiplier = int("1" + len(str(step).split(".")[1]) * "0")

    slider = QtWidgets.QSlider(window)
    slider.setOrientation(QtCore.Qt.Horizontal)
    slider.setFocusPolicy(Qt.StrongFocus)
    slider.setSingleStep(int(step * multiplier))
    slider.setRange(int(min_val * multiplier), int(max_val * multiplier))
    slider.setValue(int(init_val * multiplier))
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
        global mesh_parameter_dict
        if multiplier == 1:
            slider_val = slider.value()
        else:
            slider_val = slider.value() / multiplier
        if slider_val % 2 == 0 and step == 2:
            slider_val += 1
        label.setText(para_name + ": " + str(slider_val))
        try:
            parameter_dict[para_name][0] = slider_val
        except KeyError:
            mesh_parameter_dict[para_name][0] = slider_val


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
            add_sliders(window, para_name, val_list, slide_rows, slider_width, img_height)

    add_button(app, img_width, img_height + col_slider_count * 70, window, 1, "Run EF Postprocessing", name, mid_img,
               1.3)
    add_button(app, img_width, img_height + col_slider_count * 70, window, 2, "LV Manual Localization", name, mid_img,
               1.3)
    add_button(app, img_width, img_height + col_slider_count * 70, window, 3, "Show EF Process", name, mid_img,
               1.3)
    add_button(app, img_width, img_height + col_slider_count * 70, window, 4, "Mesh Generation", name, mid_img,
               1.3)


def run_ef_process_gui(app, window, name):
    gif_names = np.array(os.listdir("ef_process/"))
    max_num_img = 0
    min_num_img = np.inf
    min_id = ""
    for id in ["A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_", "I_", "J_", "K_"]:
        num_img = len(gif_names[np.flatnonzero(np.core.defchararray.find(gif_names, id) != -1)])
        if num_img > max_num_img:
            max_num_img = num_img
        if num_img < min_num_img:
            min_num_img = num_img
            min_id = id

    window.setWindowTitle("EF Process -" + name.split(".")[0])
    window.setWindowIcon(QtGui.QIcon('logo.png'))
    screen = app.primaryScreen()
    size = screen.size()

    # Display images
    i = 0
    num_skip = max_num_img - min_num_img
    for id in ["A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_", "I_", "J_", "K_"]:
        if i % 2 == 0:
            ver_shift = 5
        else:
            ver_shift = 1
        i += 1
        if id == min_id:
            num_skip = 0
        display_images(window, size, "ef_process/", i, 10, id, ver_shift, num_skip, True)


def run_mesh_gui(app, window, name):
    global row_slider_count, col_slider_count
    row_slider_count = 1
    col_slider_count = 1

    window.setWindowTitle("Mesh Generation -" + name.split(".")[0])
    window.setWindowIcon(QtGui.QIcon('logo.png'))
    screen = app.primaryScreen()
    size = screen.size()

    # Add sliders
    slide_rows = len(mesh_parameter_dict) // 3
    slider_width = int((size.width() / 2 - 200) // slide_rows)
    for para_name, val_list in mesh_parameter_dict.items():
        if len(val_list) == 4:
            add_sliders(window, para_name, val_list, slide_rows, slider_width, 0)

    add_button(app, int(size.width() / 2 - 100), 0 + col_slider_count * 70, window, 3, "Generate All Meshes", name, 0,
               1.3)


if __name__ == '__main__':
    # Start creating output folders
    try:
        os.mkdir('Output')
        os.mkdir('sa_ed_ef_gif')
        os.mkdir('sa_ed_gif')
        os.mkdir('sa_ed_seg_ef_gif')
        os.mkdir('sa_ed_seg_gif')
        os.mkdir('ef_process')
    except FileExistsError:
        pass
    os.mkdir('ef_processw')
    run_unet_segmentation(input_names[name_idx])

    # Add window and properties
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    window = QWidget()

    run_unet_gui(app, window, input_names[name_idx])
