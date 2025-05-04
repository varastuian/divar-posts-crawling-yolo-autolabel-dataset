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

import traceback

# time.sleep(4 * 60 * 60)
DeepModel = YOLO("yolo11n.pt")
output_dir = f"dataset/dataset_koleos"
os.makedirs(output_dir, exist_ok=True)

output_car_dir = f"{output_dir}/car"
os.makedirs(output_car_dir, exist_ok=True)

output_color_dir = f"{output_dir}/color"
os.makedirs(output_color_dir, exist_ok=True)
car_classes = {
    "rira": 0, "206": 1, "207": 2, "405": 3, "pars": 4, "samand": 5, "dena": 6, "runna": 7, "tara": 8, "haima": 9, "arisun": 10,
    "sahand": 11, "tondar-90": 12, "pride": 13, "tiba": 14, "quick": 15, "saina": 16, "shahin": 17, "zamyad": 18, "atlas": 19,
    "paykan": 20, "rio": 21, "arizo": 22, "kmct9": 23, "mazdakara": 24, "changan": 25, "brilliance": 26, "mazda3": 27,
    "mvm110": 28, "mvm110s": 29, "mvm315": 30, "mvm530": 31, "mvm550": 32, "mvmx22": 33, "mvmx33": 34, "mvmx55": 35,
    "j5": 36, "j4": 37, "s5": 38, "s3": 39,
    "lifan620": 40, "lifanx50": 41, "lifanx60": 42, "lifan520": 43, "lifan820": 44, "ifanx70": 45,
    "accent": 46, "avante": 47, "elantra": 48, "verna": 49, "azera-grandeur": 50, "sonata": 51, "tucson": 52, "santafe": 53, "i20": 54, "i30": 55, "i40": 56,
    "cerato": 57, "optima": 58, "sportage": 59, "sorento": 60, "carnival": 61,
    "tiggo-5": 62, "tiggo-7": 63, "fx": 64, "arizo-8": 65, "tiggo-7-pro": 66, "tiggo-8-pro": 67,
    "duster": 68, "fluence": 69, "koleos": 70, "sandero": 71, "megan": 72, "symbol": 73, "talisman": 74,
    "aurion": 75, "camry": 76, "corolla": 77, "prius": 78, "yaris": 79, "prado": 80, "landcruiser": 81,
    "fidelity": 82, "dignity": 83, "lamari": 84,"koleos":85
}


color_classes = {
    "تیتانیوم": 0, "سرمه‌ای": 1, "قرمز": 2, "آبی": 3, "سبز": 4, "زرد": 5, "سفید": 6, "مشکی": 7,
    "آلبالویی": 8, "اطلسی": 9, "بادمجانی": 10, "برنز": 11, "بژ": 12, "بنفش": 13, "پوست پیازی": 14,
    "تیتانیئم": 15, "خاکستری": 16, "خاکی": 17, "دلفینی": 18, "ذغالی": 19, "زرد زرشکی": 20,
    "زیتونی ": 21, "سربی": 22, "سفید صدفی": 23, "طلایی": 24, "طوسی": 25, "عدسی": 26,
    "عنابی": 27, "قهوه ای": 28, "کربن بلک": 29, "کرم": 30, "گیلاسی": 31, "مسی": 32, "موکا": 33,
    "نارنجی": 34, "نقرآبی": 35, "نقره ای": 36, "نوک مدادی": 37, "یشمی": 38
}


color_normalizer = {
    "آبی": ("آبی", "Blue"),
    "نقرآبی": ("آبی", "Blue"),
    "اطلسی": ("آبی", "Blue"),
    "سرمه‌ای": ("آبی", "Blue"),

    "قرمز": ("قرمز", "Red"),
    "آلبالویی": ("قرمز", "Red"),
    "گیلاسی": ("قرمز", "Red"),
    "عنابی": ("قرمز", "Red"),

    "مشکی": ("مشکی", "Black"),
    "کربن بلک": ("مشکی", "Black"),

    "سفید": ("سفید", "White"),
    "سفید صدفی": ("سفید", "White"),

    "نقره ای": ("خاکستری", "Gray"),
    "تیتانیوم": ("خاکستری", "Gray"),
    "تیتانیئم": ("خاکستری", "Gray"),
    "خاکستری": ("خاکستری", "Gray"),
    "دلفینی": ("خاکستری", "Gray"),
    "ذغالی": ("خاکستری", "Gray"),
    "نوک مدادی": ("خاکستری", "Gray"),
    "سربی": ("خاکستری", "Gray"),

    "قهوه ای": ("قهوه ای", "Brown"),
    "مسی": ("قهوه ای", "Brown"),
    "موکا": ("قهوه ای", "Brown"),
    "برنز": ("قهوه ای", "Brown"),

    "زرد": ("زرد", "Yellow"),
    "طلایی": ("زرد", "Yellow"),
    "زرد زرشکی": ("زرد", "Yellow"),

    "سبز": ("سبز", "Green"),
    "زیتونی": ("سبز", "Green"),
    "یشمی": ("سبز", "Green"),
    "خاکی": ("سبز", "Green"),

    "نارنجی": ("نارنجی", "Orange"),

    "بنفش": ("بنفش", "Purple"),
    "بادمجانی": ("بنفش", "Purple"),

    "بژ": ("بژ", "Beige"),
    "کرم": ("بژ", "Beige"),
    "پوست پیازی": ("بژ", "Beige"),
    "عدسی": ("بژ", "Beige"),

    "طوسی": ("خاکستری", "Gray"),
}


import re

def normalize_color(raw_color: str) -> tuple[str, str]:
    clean = re.sub(r'[\u200c\u200f\u200e]', '', raw_color.strip())
    return color_normalizer.get(clean, (clean, "Unknown"))


color_classes = {
    "سفید": 0,
    "مشکی": 1,
    "خاکستری": 2,
    "نقره ای": 3,
    "آبی": 4,
    "قرمز": 5,
    "سبز": 6,
    "زرد": 7,
    "نارنجی": 8,
    "طلایی": 9,
    "قهوه ای": 10,
    "بنفش": 11,
    "طوسی": 12,
    "سرمه‌ای": 13,
}


options = Options()
# options.add_argument("--headless")  
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
driver.set_page_load_timeout(5) 


try:
    driver.get("url")
except:
    driver.execute_script("window.stop()") 
time.sleep(2) 
wait = WebDriverWait(driver, 2)

last_ad_count = 0 
ad_links = []
unique_links = set()
while len(unique_links)<912:
    time.sleep(2)
    ads = driver.find_elements(By.CLASS_NAME, "unsafe-kt-post-card__action")
    for ad in ads:
        try:
            ad_link = ad.get_attribute("href")
            ad_title = ad.find_element(By.CLASS_NAME, "unsafe-kt-post-card__title").text.strip()
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
        print(f"No button found. {len(unique_links)}.")

# unique_links.add("https://divar.ir/v/%D9%87%DB%8C%D9%88%D9%86%D8%AF%D8%A7%DB%8C-%D8%A7%D9%84%D9%86%D8%AA%D8%B1%D8%A7-1800cc-%D9%85%D8%AF%D9%84-%DB%B2%DB%B0%DB%B1%DB%B5-%D9%86%D9%82%D8%AF-%D9%88-%D8%A7%D9%82%D8%B3%D8%A7%D8%B7/wZ126B9h")
    

print(len(unique_links))
for idx, ad_link in enumerate(unique_links):

    try:
        try:
            driver.get(ad_link)

        except:
            driver.execute_script("window.stop()")
        time.sleep(5)
        image_url = driver.find_element(By.CLASS_NAME, "kt-image-block__image").get_attribute("src")
        
        if "mapimage" in image_url:
            print("map image ignored")
            continue 
        token = ad_link.rstrip('/').split('/')[-1]
        titlePersian = driver.find_element(By.CLASS_NAME, "kt-page-title__title").text.strip().split(' ') 
        modelItemsPersian = driver.find_element(By.CLASS_NAME, "kt-unexpandable-row__action").text.strip().split(' ') 
        modelItemsEnglish = urlparse(driver.find_element(By.CLASS_NAME, "kt-unexpandable-row__action").get_attribute("href") ).path.split("/")  
        color_str = driver.find_element(By.XPATH, "(//tr[@class='kt-group-row__data-row'])[1]/td[3]").text.strip()
        color_label = normalize_color(color_str)

        value_text = driver.find_element(By.XPATH, "//div[p[text()='بدنه']]/following-sibling::div//div[contains(@class, 'kt-score-row__score')]").text
        if "تصادفی" in value_text:
            continue
        detected_model = None
        if len(modelItemsEnglish) > 2 and modelItemsEnglish[-2] != 'other':
            for model in car_classes.keys():
                if model in modelItemsEnglish[4] or (len(modelItemsEnglish) > 5 and model in modelItemsEnglish[5]) or \
                (len(modelItemsEnglish) > 5 and model in (modelItemsEnglish[4].replace('-ir', '') + modelItemsEnglish[5].replace('-ir', '')) ):
                    detected_model = model
                    break
            if detected_model is not None and any(item in modelItemsPersian for item in titlePersian):
                        timestamp = int(time.time())
                        print(f"!!!  {idx}/{len(unique_links)} {token} {ad_link} {detected_model}" , )

                        images = driver.find_elements(By.XPATH,f"//img[contains(@class, 'kt-image-block__image')]")
                                                  
                        for i, img in enumerate(images):
                            img_url = img.get_attribute("src")
                            if img_url is None:
                                continue
                            print("img_url",img_url)
                            nameing = f"{token}_{detected_model}_{i}_{timestamp}"
                            image_filename = os.path.join(output_dir,nameing+".jpg")
                            car_filename = os.path.join(output_car_dir, nameing+".txt")
                            color_filename = os.path.join(output_color_dir+'', nameing+".txt")

                            urllib.request.urlretrieve(img_url, image_filename)
                            print(img_url)
                            print(image_filename)

                            results = DeepModel(image_filename)
                            for result in results:
                                detected_image = result.plot()  
                                height, width, _ = detected_image.shape
                                print("sag",len(result.boxes))

                                if result.boxes and len(result.boxes) > 1:
                                    print("sag")

                                    largest_box = max(result.boxes, key=lambda box: box.xywh[0][2] * box.xywh[0][3])
    
                                    cls_id = int(largest_box.cls[0].item())
                                    if cls_id in [2, 7] : 
                                

                                        with open(car_filename, "w") as f:
                                            x_center, y_center, width, height = largest_box.xywhn[0]
                                            class_id = car_classes[detected_model]
                                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")



                                        with open(color_filename, "w") as f:
                                            x_center, y_center, width, height = largest_box.xywhn[0]
                                            class_id = color_classes[color_label[0]]
                                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                                        print("done")
                                        
                                    else:
                                        print(f"Deleting1 {img_url} {image_filename} {car_filename } {color_filename}  ")

                                        os.remove(image_filename)

                                else:
                                    print(f"Deleting2 {img_url} {image_filename} {car_filename } {color_filename}  ")

                                    os.remove(image_filename)

                                print('-'*50)

    except Exception as e:
        print(f"sag Error processing {e}")
        traceback.print_exc()
        print(f"Deleting3 {img_url} {image_filename} {car_filename } {color_filename}  ")

        if os.path.exists(image_filename):
            os.remove(image_filename)
        if os.path.exists(car_filename):
            os.remove(car_filename)
        if os.path.exists(color_filename):
            os.remove(color_filename)
        continue


driver.quit()
