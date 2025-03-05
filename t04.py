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


model = YOLO("yolo11n.pt")


ikco_models = ["206", "207", "405","rd", "pars", "2008", "508", "301", "samand", "soren", "dena", "runna", "tara", "haima", "arisun"]
saipa_models = ["saipa-shahin,pride", "tiba", "quick", "saina", "shahin", "zamyad", "roham", "padra", "151"]
interestCars = ikco_models + saipa_models

car_classes = {model: idx for idx, model in enumerate(interestCars)}
dataset = {
    "path": "/datasets",
    "train": "train.txt",
    "val": "val.txt",
    "names": car_classes  
}



options = Options()
# options.add_argument("--headless")  
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--ignore-certificate-errors")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

url = "https://divar.ir/s/tehran-province/car"
driver.get(url)
# time.sleep(5) 

# for _ in range(3):
#     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#     time.sleep(3)

ads = driver.find_elements(By.CLASS_NAME, "kt-post-card__action")

ad_links = []
for ad in ads:
    try:
        # ad_link = ad.find_element(By.TAG_NAME, "a").get_attribute("href")
        ad_link = ad.get_attribute("href")
        ad_title = ad.find_element(By.CLASS_NAME, "kt-post-card__title").text.strip()
        if not ad_link.startswith("https"):
            ad_link = "https://divar.ir" + ad_link
        ad_links.append(ad_link)
    except:
        continue
print(len(ad_links))
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

for idx, ad_link in enumerate(ad_links):
    try:
        driver.get(ad_link)
        time.sleep(3)

        image_url = driver.find_element(By.CLASS_NAME, "kt-image-block__image").get_attribute("src")
        if "mapimage" in image_url:
            print("map image ignored")
            continue 
        title = driver.find_element(By.CLASS_NAME, "kt-page-title__title").text.strip().split(' ') 
        modelitem1 = driver.find_element(By.CLASS_NAME, "kt-unexpandable-row__action").text.strip().split(' ') 
        modelItems = urlparse(driver.find_element(By.CLASS_NAME, "kt-unexpandable-row__action").get_attribute("href") ).path.split("/")  
        
        # if (modelItems[4]!='other'):
        #     modell = " "
        #     if any(modelItems[4] in model  for model in interestCars):
        #         modell=f"{modelItems[4]}"

        #     elif any(modelItems[5] in model for model in interestCars):
        #         modell=f"{modelItems[5]}"
        detected_model = None
        if modelItems[4] != 'other':
            for model in car_classes.keys():
                if model in modelItems[4] or model in modelItems[5]:
                    detected_model = model
                    break
        if detected_model and any(item in modelitem1 for item in title):
            # if modell != " " :
            #     if any(item in modelitem1 for item in title):
                    print(f"Model: {modelitem1} | modelitem1: {modelitem1}")
                    image_filename = os.path.join(output_dir, f"{modell}_{idx}.jpg")
                    urllib.request.urlretrieve(image_url, image_filename)
                    
                    results = model(image_filename)


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

        yaml_path = os.path.join(output_dir, "customdata.yaml")
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(dataset, yaml_file, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        print(f"Error processing {e}")

driver.quit()
