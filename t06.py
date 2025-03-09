from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import os
import urllib.request
from urllib.parse import urlparse
from ultralytics import YOLO
import cv2
import yaml



DeepModel = YOLO("yolo11n.pt")
output_dir = f"dataset/dataset_{int(time.time())}"
os.makedirs(output_dir, exist_ok=True)

ikco_models = ["rira","206", "207", "405","pars", "samand", "dena", "runna", "tara", "haima", "arisun"]
saipa_models = ["sahand","tondar-90","pride", "tiba", "quick", "saina", "shahin", "zamyad","atlas"]

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
options.add_argument("--headless")  
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
driver.set_page_load_timeout(1) 
try:
    driver.get("https://divar.ir/s/tehran/car?brand_model_manufacturer_origin=domestic")
except:
    driver.execute_script("window.stop()") 
time.sleep(2) 
wait = WebDriverWait(driver, 2)


ad_links = []
unique_links = set()
while len(unique_links)<3000:
    ads = driver.find_elements(By.CLASS_NAME, "kt-post-card__action")
    time.sleep(2)
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
                # ad_links.append(ad_link)
        except:
            continue
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # time.sleep(1)
    try:
        load_more_btn = driver.find_element(By.CLASS_NAME, "post-list__load-more-btn-be092")

        load_more_btn = wait.until(EC.element_to_be_clickable(
            (By.CLASS_NAME, "post-list__load-more-btn-be092")
        ))
        load_more_btn.click()
        # time.sleep(2) 
    except Exception as e:
        print(f"No more 'Load More' button found. {len(unique_links)}.")

print(len(unique_links))
for idx, ad_link in enumerate(unique_links):
    try:
        # driver.get(ad_link)
        try:
            driver.get(ad_link)

        except:
            driver.execute_script("window.stop()")
        # time.sleep(2)
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
        detected_model = None
        if modelItemsEnglish[len(modelItemsEnglish)-2] != 'other':
            for model in car_classes.keys():
                if model in modelItemsEnglish[4] or model in modelItemsEnglish[5]:
                    detected_model = model
                    break
            if detected_model is not None and any(item in modelItemsPersian for item in titlePersian):
                        print("car data matched")
                        timestamp = int(time.time())
                        print(f"LLLLIIINNNKK... {idx} {timestamp} {ad_link} {modelItemsEnglish}" , )

                        image_filename = os.path.join(output_dir, f"{detected_model}_{timestamp}.jpg")
                        urllib.request.urlretrieve(image_url, image_filename)

                        results = DeepModel(image_filename)

                        for result in results:
                            detected_image = result.plot()  
                            height, width, _ = detected_image.shape
                            filtered_boxes = [
                                    box for box in result.boxes 
                                    if int(box.cls[0].item()) == 2 
                                    and box.conf[0].item() > 0.4
                                    and (box.xywh[0][2] * box.xywh[0][3]) > 0.21 * (width * height) 
                                    and 0.6 <= (box.xywh[0][2] / box.xywh[0][3]) <= 1.15
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
                                    class_id = car_classes[detected_model]
                                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                                break
                            else:
                                print(f"!!!Deleting {image_filename} !!! ")
                                os.remove(image_filename)
    except Exception as e:
        print(f"Error processing {e}")


driver.quit()
