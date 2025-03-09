from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import os
import urllib.request
from urllib.parse import urlparse
from ultralytics import YOLO
from PIL import Image
import cv2
import yaml
from collections import OrderedDict



DeepModel = YOLO("yolo11n.pt")
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)
TARGET_SIZE = (640, 640)

ikco_models = ["206", "207", "405","pars", "samand", "soren", "dena", "runna", "tara", "haima", "arisun"]
saipa_models = ["pride", "tiba", "quick", "saina", "shahin", "zamyad"]

interestCars = ikco_models + saipa_models

car_classes = {model: idx for idx, model in enumerate(interestCars)}

dataset = {
    "path": "datasets",
    "train": "train",
    "val": "val",
    "names": dict(car_classes)  
}

yaml_path = os.path.join(output_dir, "customdata.yaml")
with open(yaml_path, "w") as yaml_file:
    yaml.dump(dataset, yaml_file, default_flow_style=False, allow_unicode=True, sort_keys=False) 

options = Options()
# options.add_argument("--headless")  
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--ignore-certificate-errors")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# url = "https://divar.ir/s/tehran-province/car"
url = "https://divar.ir/s/tehran/car?brand_model_manufacturer_origin=domestic"
driver.get(url)
time.sleep(5) 

ad_links = []
unique_links = set()

for _ in range(55):
    ads = driver.find_elements(By.CLASS_NAME, "kt-post-card__action")

    for ad in ads:
        try:
            ad_link = ad.get_attribute("href")
            ad_title = ad.find_element(By.CLASS_NAME, "kt-post-card__title").text.strip()
            if not ad_link.startswith("https"):
                ad_link = "https://divar.ir" + ad_link
            if ad_link in unique_links:
                print(f"Duplicate link found: {ad_link}")
            else:
                unique_links.add(ad_link)
                ad_links.append(ad_link)
        except:
            continue
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
print(len(ad_links))


for idx, ad_link in enumerate(ad_links):
    try:
        driver.get(ad_link)
        time.sleep(3)

        image_url = driver.find_element(By.CLASS_NAME, "kt-image-block__image").get_attribute("src")
        if "mapimage" in image_url:
            print("map image ignored")
            continue 
        titlePersian = driver.find_element(By.CLASS_NAME, "kt-page-title__title").text.strip().split(' ') 
        modelItemsPersian = driver.find_element(By.CLASS_NAME, "kt-unexpandable-row__action").text.strip().split(' ') 
        modelItemsEnglish = urlparse(driver.find_element(By.CLASS_NAME, "kt-unexpandable-row__action").get_attribute("href") ).path.split("/")  
        status_div = driver.find_element(By.XPATH, "//p[@class='kt-base-row__title kt-unexpandable-row__title' and text()='وضعیت بدنه']")
        value_text = status_div.find_element(By.XPATH, "./../../div[@class='kt-base-row__end kt-unexpandable-row__value-box']/p").text.strip()

        if value_text == "تصادفی":
            print("Ignoring this ad (تصادفی found).")
            continue

        if modelItemsEnglish[len(modelItemsEnglish)-2] != 'other':
            for model in car_classes.keys():
                if model in modelItemsEnglish[4] or model in modelItemsEnglish[5]:
                    detected_model = model
                    
                    if any(item in modelItemsPersian for item in titlePersian):
                                print(f"Model: {modelItemsPersian} | modelitem1: {titlePersian}")
                                image_filename = os.path.join(output_dir, f"{detected_model}_{idx}.jpg")
                                urllib.request.urlretrieve(image_url, image_filename)
                                image = cv2.imread(image_filename)
                                if image is None:
                                    print(f"Error: Could not read {image_filename}")
                                else:
                                    resized_image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                                    cv2.imwrite(image_filename, resized_image)  

                                    # Step 3: Run YOLO Model on Resized Image
                                    results = DeepModel(image_filename)

                                results = DeepModel(image_filename)

                                for result in results:
                                    detected_image = result.plot()  
                                    height, width, _ = detected_image.shape
                                    filtered_boxes = [
                                            box for box in result.boxes 
                                            if int(box.cls[0].item()) == 2 
                                            and box.conf[0].item() > 0.4
                                            and (box.xywh[0][2] * box.xywh[0][3]) > 0.21 * (width * height) 
                                        ]
                                    if filtered_boxes:
                                        best_box = max(filtered_boxes, key=lambda b: b.conf[0].item())
                                        result.boxes = [best_box] 
                                        detected_image = result.plot()  

                                        output_image_path = os.path.join(output_dir, f"detected_{image_filename}")
                                        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                                        cv2.imwrite(output_image_path, detected_image)


                                        with open(f"{image_filename.split('.')[0]}.txt", "w") as f:
                                            x_center, y_center, width, height = best_box.xywhn[0]
                                            # class_id = int(best_box.cls[0].item())
                                            class_id = car_classes[detected_model]
                                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                                        break
                                    else:
                                        print(f"Deleting {image_filename} because no predictions were made.")
                                        os.remove(image_filename)
    except Exception as e:
        print(f"Error processing {e}")

driver.quit()
