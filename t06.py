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


# time.sleep(4 * 60 * 60)
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
    "names": interestCars  
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
# service = Service(ChromeDriverManager().install())

service = Service( r"chromedriver-win64\chromedriver-win64\chromedriver.exe")
driver = webdriver.Chrome(service=service, options=options)
driver.set_page_load_timeout(1) 


try:
    # driver.get("https://divar.ir/s/tehran/car?brand_model_manufacturer_origin=domestic")
    #zamyad , 405
    driver.get("https://divar.ir/s/tehran/car/peugeot/405")
except:
    driver.execute_script("window.stop()") 
time.sleep(2) 
wait = WebDriverWait(driver, 2)

last_ad_count = 0 
ad_links = []
unique_links = set()
while len(unique_links)<1000:
    time.sleep(2)
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
                # ad_links.append(ad_link)
        except:
            continue
        
    if len(unique_links) != 0 and len(unique_links) == last_ad_count:
        print(f"No new content found after scrolling. Stopping at {len(unique_links)} ads.")
        break

    last_ad_count = len(unique_links)
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
            # print("Ignoring this ad (تصادفی found).")
            continue
        detected_model = None
        if modelItemsEnglish[len(modelItemsEnglish)-2] != 'other':
            for model in car_classes.keys():
                if model in modelItemsEnglish[4] or model in modelItemsEnglish[5]:
                    detected_model = model
                    break
            if detected_model is not None and any(item in modelItemsPersian for item in titlePersian):
                        # print("car data matched")
                        timestamp = int(time.time())
                        print(f"!!!  {idx}/{len(unique_links)} {timestamp} {ad_link} {modelItemsEnglish} !!!" , )

                        image_filename = os.path.join(output_dir, f"{detected_model}_{timestamp}.jpg")
                        urllib.request.urlretrieve(image_url, image_filename)

                        results = DeepModel(image_filename)
                        rawImage = cv2.imread(image_filename)
                        for result in results:
                            detected_image = result.plot()  
                            height, width, _ = detected_image.shape
                            largest_box = max(result.boxes, key=lambda box: box.xywh[0][2] * box.xywh[0][3])
                            cls_id = int(largest_box.cls[0].item())
                            conf = largest_box.conf[0].item()
                            # area = largest_box.xywh[0][2] * largest_box.xywh[0][3]
                            aspect_ratio = largest_box.xywh[0][2] / largest_box.xywh[0][3]
                            if (cls_id in [2, 7] and
                                conf > 0.4 and
                                0.6 <= aspect_ratio <= 1.19):
                                result.boxes = [largest_box] 
                                detected_image = result.plot()  

                                output_image_path = os.path.join(output_dir, f"detected_dataset/{detected_model}_{timestamp}.jpg")
                                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                                cv2.imwrite(output_image_path, detected_image)


                                with open(f"{image_filename.split('.')[0]}.txt", "w") as f:
                                    x_center, y_center, width, height = largest_box.xywhn[0]
                                    class_id = car_classes[detected_model]
                                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                                    
                                    
                                # Resize the detected image to a smaller size 
                                height,width,_= rawImage.shape
                                scale_x = 0.1
                                scale_y = 0.1
                                new_width = int(width * scale_x)
                                new_height = int(height * scale_y)
                                resized_image = cv2.resize(rawImage, (new_width, new_height))
                                output_image_path_resized = os.path.join(output_dir, f"resized_{detected_model}_{timestamp}.jpg")
                                cv2.imwrite(output_image_path_resized, resized_image)
                                # Get original box coordinates
                                x1, y1, x2, y2 = largest_box.xyxy[0]

                                # Scale bounding box to resized image
                                x1_resized = int(x1 * scale_x)
                                y1_resized = int(y1 * scale_y)
                                x2_resized = int(x2 * scale_x)
                                y2_resized = int(y2 * scale_y)

                                # Draw the resized bounding box on the resized image
                                cv2.rectangle(resized_image, (x1_resized, y1_resized), (x2_resized, y2_resized), (0, 255, 0), 3)

                                # Save resized image
                                cv2.imwrite(os.path.join(output_dir, f"detected_dataset/resized_{detected_model}_{timestamp}.jpg"), resized_image)

                                # Save annotations for resized image
                                with open(f"{output_image_path_resized.split('.')[0]}.txt", "w") as f:
                                    # Normalize coordinates for YOLO format
                                    x_center_resized = (x1_resized + x2_resized) / (2 * new_width)
                                    y_center_resized = (y1_resized + y2_resized) / (2 * new_height)
                                    box_width_resized = (x2_resized - x1_resized) / new_width
                                    box_height_resized = (y2_resized - y1_resized) / new_height

                                    f.write(f"{class_id} {x_center_resized} {y_center_resized} {box_width_resized} {box_height_resized}\n")
                                    
                                break
                            else:
                                print(f"!!!Deleting {image_filename} !!! ")
                                os.remove(image_filename)
    except Exception as e:
        # print(f"Error processing {e}")
        continue


driver.quit()
