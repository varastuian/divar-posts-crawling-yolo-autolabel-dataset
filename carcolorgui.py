import sys
import os
import time
import urllib.request
import yaml
import cv2
import torch
from urllib.parse import urlparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QProgressBar, QLineEdit, QPushButton, QLabel, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ultralytics import YOLO

class Worker(QThread):
    progress_updated = pyqtSignal(int, int, int, int)  # slot_idx, current, total, percent
    status_message = pyqtSignal(int, str)  # slot_idx, message
    finished = pyqtSignal(int, int, str)  # slot_idx, count, url
    error = pyqtSignal(int, str)  # slot_idx, message

    def __init__(self, url, output_dir, slot_idx):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        self.slot_idx = slot_idx
        self.running = True
        self.current_count = 0
        self.total_ads = 0
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Worker {slot_idx} using device: {self.device}")

    def run(self):
        try:
            # Load YOLOv11 model on GPU
            try:
                self.deep_model = YOLO("yolo11n.pt").to(self.device)
                self.status_message.emit(self.slot_idx, f"YOLOv11 model loaded on {'GPU' if 'cuda' in str(self.device) else 'CPU'}")
            except Exception as e:
                self.error.emit(self.slot_idx, f"Failed to load YOLOv11 model: {str(e)}")
                return

            # Define car models and classes
            self.interest_cars = [
                "rira", "206", "207", "405", "pars", "samand", "dena", "runna", "tara", "haima", "arisun",
                "sahand", "tondar-90", "pride", "tiba", "quick", "saina", "shahin", "zamyad", "atlas",
                "paykan", "rio", "arizo", "kmct9", "mazdakara", "changan", "brilliance", "mazda3",
                "mvm110", "mvm110s", "mvm315", "mvm530", "mvm550", "mvmx22", "mvmx33", "mvmx55",
                "j5", "j4", "s5", "s3",
                "lifan620", "lifanx50", "lifanx60", "lifan520", "lifan820", "ifanx70",
                "accent", "avante", "elantra", "verna", "azera-grandeur", "sonata", "tucson", "santafe", "i20", "i30", "i40",
                "cerato", "optima", "sportage", "sorento", "carnival",
                "tiggo-5", "tiggo-7",
                "fx", "arizo-8", "tiggo-7-pro", "tiggo-8-pro",
                "duster", "fluence", "koleos", "sandero", "megan", "symbol", "talisman",
                "aurion", "camry", "corolla", "prius", "yaris", "prado", "landcruiser",
                "fidelity", "dignity", "lamari"
            ]
            self.car_classes = {model: idx for idx, model in enumerate(self.interest_cars)}

            # Define color classes
            self.color_classes = {
                "تیتانیوم": 0, "سرمه‌ای": 1, "قرمز": 2, "آبی": 3, "سبز": 4, "زرد": 5, "سفید": 6, "مشکی": 7,
                "آلبالویی": 8, "اطلسی": 9, "بادمجانی": 10, "برنز": 11, "بژ": 12, "بنفش": 13, "پوست پیازی": 14,
                "تیتانیئم": 15, "خاکستری": 16, "خاکی": 17, "دلفینی": 18, "ذغالی": 19, "زرد زرشکی": 20,
                "زیتونی ": 21, "سربی": 22, "سفید صدفی": 23, "طلایی": 24, "طوسی": 25, "عدسی": 26,
                "عنابی": 27, "قهوه ای": 28, "کربن بلک": 29, "کرم": 30, "گیلاسی": 31, "مسی": 32, "موکا": 33,
                "نارنجی": 34, "نقرآبی": 35, "نقره ای": 36, "نوک مدادی": 37, "یشمی": 38
            }
            self.default_color_class = -1

            # Setup driver
            driver = self.setup_driver()
            self.status_message.emit(self.slot_idx, f"Loading URL: {self.url}")
            try:
                driver.get(self.url)
            except:
                driver.execute_script("window.stop()")
            time.sleep(3)
            
            # Create main output directory and subdirectories
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "model_annotations"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "color_annotations"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "detected_dataset"), exist_ok=True)
            
            # Create dataset YAML
            dataset = {
                "path": os.path.abspath(self.output_dir),
                "train": ".",  # Current directory for original images/labels
                "val": ".",    # Current directory for original images/labels
                "names": {i: name for i, name in enumerate(self.interest_cars)}
            }
            yaml_path = os.path.join(self.output_dir, "data.yaml")
            with open(yaml_path, "w", encoding='utf-8') as yaml_file:
                yaml.dump(dataset, yaml_file, default_flow_style=False, allow_unicode=True)

            if 'divar' in driver.current_url:
                # Collect ad links
                self.status_message.emit(self.slot_idx, "Collecting ad links...")
                unique_links = self.collect_links(driver)
                if not unique_links:
                    self.error.emit(self.slot_idx, "No ad links collected from the page")
                    return

                self.total_ads = len(unique_links)
                self.status_message.emit(self.slot_idx, f"Found {self.total_ads} ads to process")
                
                # Process ads
                self.process_ads(driver, unique_links)
                
            else:
                self.error.emit(self.slot_idx, "Only Divar.ir URLs are supported in this version")
                return

            # Final update
            self.progress_updated.emit(self.slot_idx, self.current_count, self.total_ads, 100)
            self.status_message.emit(self.slot_idx, "Processing completed")
            self.finished.emit(self.slot_idx, self.current_count, self.url)
            
        except Exception as e:
            self.error.emit(self.slot_idx, f"Error in processing: {str(e)}")
        finally:
            if 'driver' in locals():
                driver.quit()

    def setup_driver(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(10)  # Increased timeout
        return driver

    def collect_links(self, driver):
        unique_links = set()
        wait = WebDriverWait(driver, 10)
        max_attempts = 100
        attempt = 0
        last_ad_count = 0
        
        while len(unique_links) < 500 and attempt < max_attempts and self.running:  
            time.sleep(3)
            try:
                ads = driver.find_elements(By.CLASS_NAME, "kt-post-card__action")
                self.status_message.emit(self.slot_idx, f"Found {len(ads)} ads (attempt {attempt + 1})")
                
                for ad in ads:
                    try:
                        ad_link = ad.get_attribute("href")
                        if ad_link:
                            if not ad_link.startswith("https"):
                                ad_link = "https://divar.ir" + ad_link
                            unique_links.add(ad_link)
                    except:
                        continue
                
                if len(unique_links) == last_ad_count and len(unique_links) > 0 and attempt > 3:
                    break
                
                last_ad_count = len(unique_links)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                
                try:
                    load_more_btn = wait.until(EC.element_to_be_clickable(
                        (By.CLASS_NAME, "post-list__load-more-btn-be092")
                    ))
                    load_more_btn.click()
                    time.sleep(3)
                except:
                    if attempt > 3:
                        break
                
                attempt += 1
            except Exception as e:
                self.status_message.emit(self.slot_idx, f"Error collecting links: {str(e)}")
                continue
        print('-'*50)
        print("Collected links:", len(unique_links))
        return unique_links

    def process_ads(self, driver, unique_links):
        wait = WebDriverWait(driver, 10)
        
        for idx, ad_link in enumerate(unique_links):
            if not self.running:
                break
            print(idx)
            try:
                try:
                    driver.get(ad_link)
                except:
                    driver.execute_script("window.stop()")
                time.sleep(1)
                
                # Get image URL
                try:
                    img_element = wait.until(EC.presence_of_element_located(
                        (By.CLASS_NAME, "kt-image-block__image")
                    ))
                    image_url = img_element.get_attribute("src")
                    
                    if not image_url or "mapimage" in image_url:
                        self.status_message.emit(self.slot_idx, f"Map image ignored for {ad_link}")
                        continue
                except:
                    self.status_message.emit(self.slot_idx, f"Failed to load image for {ad_link}")
                    continue
                
                # Get car details
                try:
                    titlePersian = driver.find_element(By.CLASS_NAME, "kt-page-title__title").text.strip().split(' ')
                    modelItemsPersian = driver.find_element(By.CLASS_NAME, "kt-unexpandable-row__action").text.strip().split(' ')
                    href_element = driver.find_element(By.CLASS_NAME, "kt-unexpandable-row__action")
                    modelItemsEnglish = urlparse(href_element.get_attribute("href")).path.split("/")
                    
                    # Check vehicle condition
                    try:
                        status_div = driver.find_element(By.XPATH, "//p[@class='kt-base-row__title kt-unexpandable-row__title' and text()='وضعیت بدنه']")
                        value_text = status_div.find_element(By.XPATH, "./../../div[@class='kt-base-row__end kt-unexpandable-row__value-box']/p").text.strip()
                        if value_text == "تصادفی":
                            self.status_message.emit(self.slot_idx, f"Skipping accident car: {ad_link}")
                            continue
                    except:
                        pass
                    
                    # Scrape car color
                    try:
                        color_str = driver.find_element(By.XPATH, "(//tr[@class='kt-group-row__data-row'])[1]/td[3]").text.strip()
                        self.status_message.emit(self.slot_idx, f"Detected color: {color_str}")
                    except Exception as e:
                        self.status_message.emit(self.slot_idx, f"Could not extract color info: {str(e)}")
                        color_str = None
                    color_label = self.color_classes.get(color_str, self.default_color_class)
                    
                    # Detect car model                            
                    detected_model = None
                    if len(modelItemsEnglish) > 2 and modelItemsEnglish[-2] != 'other':
                        for model in self.car_classes.keys():
                            if (model in modelItemsEnglish[4] or 
                                (len(modelItemsEnglish) > 5 and model in modelItemsEnglish[5]) or 
                                (len(modelItemsEnglish) > 5 and model in (modelItemsEnglish[4].replace('-ir', '') + modelItemsEnglish[5].replace('-ir', '')))):
                                detected_model = model
                                break
                        
                    if not detected_model or not any(item in modelItemsPersian for item in titlePersian):
                        self.status_message.emit(self.slot_idx, f"No valid model detected or title mismatch for {ad_link}")
                        continue

                    timestamp = int(time.time())
                    image_filename = os.path.join(self.output_dir, f"{detected_model}_{timestamp}.jpg")
                    
                    # Download image
                    try:
                        urllib.request.urlretrieve(image_url, image_filename)
                        if not os.path.exists(image_filename) or os.path.getsize(image_filename) == 0:
                            raise Exception("Empty or invalid image file")
                    except Exception as e:
                        self.status_message.emit(self.slot_idx, f"Failed to download image: {str(e)}")
                        continue
                    
                    # Process with YOLOv11
                    try:
                        results = self.deep_model(image_filename)
                        if len(results) == 0:
                            self.status_message.emit(self.slot_idx, f"No detections found in {image_filename}")
                            os.remove(image_filename)
                            continue

                        rawImage = cv2.imread(image_filename)
                        if rawImage is None:
                            raise Exception("Failed to read image file")
                            
                        for result in results:
                            detected_image = result.plot()  # Original image with YOLO bounding box
                            largest_box = max(result.boxes, key=lambda box: box.xywh[0][2] * box.xywh[0][3])
                            cls_id = int(largest_box.cls[0].item())
                            conf = largest_box.conf[0].item()
                            aspect_ratio = largest_box.xywh[0][2] / largest_box.xywh[0][3]
                            
                            if cls_id not in [2, 7] or conf <= 0.3 or not (0.5 <= aspect_ratio <= 1.20):
                                self.status_message.emit(self.slot_idx, f"Deleting {image_filename} due to unmet condition: cls_id={cls_id}, conf={conf}, aspect_ratio={aspect_ratio}")
                                print(self.slot_idx, f"Deleting {image_url} due to unmet condition: cls_id={cls_id}, conf={conf}, aspect_ratio={aspect_ratio}")

                                os.remove(image_filename)
                                continue

                            # Adjust bounding box for hood (top third of the car)
                            x1, y1, x2, y2 = largest_box.xyxy[0].tolist()
                            hood_y1 = y2 - (y2 - y1) / 1.3  # Take top third for hood
                            hood_box = [x1, hood_y1, x2, y2]

                            result.boxes = [largest_box]  # Keep original for model detection
                            detected_image = result.plot()  # Re-plot with only the largest box

                            # Draw hood bounding box on the normal image (in red)
                            cv2.rectangle(detected_image, 
                                          (int(hood_box[0]), int(hood_box[1])), 
                                          (int(hood_box[2]), int(hood_box[3])), 
                                          (0, 0, 255),  # Red color in BGR
                                          3)  # Thickness

                            # Save detected image with bounding boxes
                            output_image_path = os.path.join(self.output_dir, "detected_dataset", f"{detected_model}_{timestamp}.jpg")
                            cv2.imwrite(output_image_path, detected_image)

                            # Save model annotation (full car)
                            model_annotation_file = os.path.join(self.output_dir, "model_annotations", f"{detected_model}_{timestamp}.txt")
                            x_center, y_center, width, height = largest_box.xywhn[0].tolist()
                            car_class_id = self.car_classes[detected_model]
                            with open(model_annotation_file, "w") as f:
                                f.write(f"{car_class_id} {x_center} {y_center} {width} {height}\n")

                            # Save color annotation (hood only)
                            color_annotation_file = os.path.join(self.output_dir, "color_annotations", f"{detected_model}_{timestamp}.txt")
                            hood_x_center = (hood_box[0] + hood_box[2]) / (2 * rawImage.shape[1])
                            hood_y_center = (hood_box[1] + hood_box[3]) / (2 * rawImage.shape[0])
                            hood_width = (hood_box[2] - hood_box[0]) / rawImage.shape[1]
                            hood_height = (hood_box[3] - hood_box[1]) / rawImage.shape[0]
                            color_class_id = color_label
                            with open(color_annotation_file, "w") as f:
                                f.write(f"{color_class_id} {hood_x_center} {hood_y_center} {hood_width} {hood_height}\n")

                            # Resize and annotate
                            height_raw, width_raw, _ = rawImage.shape
                            scale_x, scale_y = 0.1, 0.1
                            new_width = int(width_raw * scale_x)
                            new_height = int(height_raw * scale_y)
                            resized_image = cv2.resize(rawImage, (new_width, new_height))
                            output_image_path_resized = os.path.join(self.output_dir, f"resized_{detected_model}_{timestamp}.jpg")
                            cv2.imwrite(output_image_path_resized, resized_image)

                            # Draw rectangles on resized image
                            x1_resized = int(x1 * scale_x)
                            y1_resized = int(y1 * scale_y)
                            x2_resized = int(x2 * scale_x)
                            y2_resized = int(y2 * scale_y)
                            cv2.rectangle(resized_image, (x1_resized, y1_resized), (x2_resized, y2_resized), (0, 255, 0), 3)

                            hood_x1_resized = int(hood_box[0] * scale_x)
                            hood_y1_resized = int(hood_box[1] * scale_y)
                            hood_x2_resized = int(hood_box[2] * scale_x)
                            hood_y2_resized = int(hood_box[3] * scale_y)
                            cv2.rectangle(resized_image, (hood_x1_resized, hood_y1_resized), (hood_x2_resized, hood_y2_resized), (255, 0, 0), 3)

                            # Save annotated resized image
                            cv2.imwrite(os.path.join(self.output_dir, "detected_dataset", f"resized_{detected_model}_{timestamp}.jpg"), resized_image)

                            # Save resized annotations
                            model_annotation_file_resized = os.path.join(self.output_dir, "model_annotations", f"resized_{detected_model}_{timestamp}.txt")
                            color_annotation_file_resized = os.path.join(self.output_dir, "color_annotations", f"resized_{detected_model}_{timestamp}.txt")

                            x_center_resized = (x1_resized + x2_resized) / (2 * new_width)
                            y_center_resized = (y1_resized + y2_resized) / (2 * new_height)
                            box_width_resized = (x2_resized - x1_resized) / new_width
                            box_height_resized = (y2_resized - y1_resized) / new_height
                            with open(model_annotation_file_resized, "w") as f:
                                f.write(f"{car_class_id} {x_center_resized} {y_center_resized} {box_width_resized} {box_height_resized}\n")

                            hood_x_center_resized = (hood_x1_resized + hood_x2_resized) / (2 * new_width)
                            hood_y_center_resized = (hood_y1_resized + hood_y2_resized) / (2 * new_height)
                            hood_box_width_resized = (hood_x2_resized - hood_x1_resized) / new_width
                            hood_box_height_resized = (hood_y2_resized - hood_y1_resized) / new_height
                            with open(color_annotation_file_resized, "w") as f:
                                f.write(f"{color_class_id} {hood_x_center_resized} {hood_y_center_resized} {hood_box_width_resized} {hood_box_height_resized}\n")

                            self.current_count += 1
                            progress = min(100, int((self.current_count / self.total_ads) * 100))
                            self.progress_updated.emit(self.slot_idx, self.current_count, self.total_ads, progress)
                            self.status_message.emit(self.slot_idx, f"Processed {self.current_count}/{self.total_ads}")
                            break  # Process only the first valid result
                    except Exception as e:
                        self.status_message.emit(self.slot_idx, f"Error processing image: {str(e)}")
                        if os.path.exists(image_filename):
                            os.remove(image_filename)
                        continue
                except Exception as e:
                    self.status_message.emit(self.slot_idx, f"Error processing ad details: {str(e)}")
                    continue
            
            except Exception as e:
                self.status_message.emit(self.slot_idx, f"Error processing ad: {str(e)}")
                continue

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Image Scraper (YOLOv11)")
        self.setGeometry(100, 100, 1000, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.workers = []
        self.create_ui()
        
    def create_ui(self):
        for i in range(5):  # 5 download slots
            frame = QWidget()
            frame_layout = QHBoxLayout()
            frame.setLayout(frame_layout)
            
            url_entry = QLineEdit()
            url_entry.setPlaceholderText("Enter URL (https://divar.ir)")
            url_entry.setMinimumWidth(350)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setTextVisible(True)
            
            progress_label = QLabel("0/0 (0%)")
            progress_label.setAlignment(Qt.AlignCenter)
            progress_label.setMinimumWidth(100)
            
            status_label = QLabel("Ready")
            status_label.setAlignment(Qt.AlignCenter)
            status_label.setMinimumWidth(150)
            
            download_btn = QPushButton("Start")
            download_btn.setFixedWidth(80)
            
            frame_layout.addWidget(url_entry)
            frame_layout.addWidget(progress_bar)
            frame_layout.addWidget(progress_label)
            frame_layout.addWidget(status_label)
            frame_layout.addWidget(download_btn)
            
            self.layout.addWidget(frame)
            
            download_btn.clicked.connect(lambda _, idx=i, e=url_entry, p=progress_bar, l=progress_label, s=status_label: 
                                        self.start_download(idx, e, p, l, s))
    
    def start_download(self, slot_idx, url_entry, progress_bar, progress_label, status_label):
        url = url_entry.text().strip()
        if url:
            output_dir = f"dataset/dataset_{int(time.time())}"
            os.makedirs(output_dir, exist_ok=True)
            
            worker = Worker(url, output_dir, slot_idx)
            self.workers.append(worker)
            
            progress_bar.setValue(0)
            progress_label.setText("0/0 (0%)")
            status_label.setText("Starting...")
            url_entry.setEnabled(False)
            
            worker.progress_updated.connect(
                lambda s_idx, current, total, percent: self.update_progress(s_idx, current, total, percent, progress_bar, progress_label))
            worker.status_message.connect(
                lambda s_idx, msg: self.update_status(s_idx, msg, status_label))
            worker.finished.connect(
                lambda s_idx, count, url: self.on_finished(s_idx, count, url, url_entry, progress_bar, progress_label, status_label))
            worker.error.connect(
                lambda s_idx, msg: self.on_error(s_idx, msg, url_entry, progress_bar, progress_label, status_label))
            
            worker.start()
    
    def update_progress(self, slot_idx, current, total, percent, progress_bar, progress_label):
        if slot_idx == self.workers.index(self.sender()):
            progress_bar.setValue(percent)
            progress_label.setText(f"{current}/{total} ({percent}%)")
    
    def update_status(self, slot_idx, message, status_label):
        if slot_idx == self.workers.index(self.sender()):
            status_label.setText(message)
    
    def on_finished(self, slot_idx, count, url, url_entry, progress_bar, progress_label, status_label):
        if slot_idx == self.workers.index(self.sender()):
            url_entry.setEnabled(True)
            status_label.setText("Completed")
            QMessageBox.information(self, "Completed", f"Processed {count} images from {url}")
    
    def on_error(self, slot_idx, message, url_entry, progress_bar, progress_label, status_label):
        if slot_idx == self.workers.index(self.sender()):
            url_entry.setEnabled(True)
            progress_bar.setValue(0)
            progress_label.setText("0/0 (0%)")
            status_label.setText("Error")
            QMessageBox.critical(self, "Error", message)
    
    def closeEvent(self, event):
        for worker in self.workers:
            worker.stop()
        event.accept()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force CUDA GPU
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    palette = app.palette()
    palette.setColor(palette.Window, Qt.white)
    palette.setColor(palette.WindowText, Qt.black)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())