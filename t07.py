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

output_car_dir = f"dataset/dataset_{int(time.time())}/car"
os.makedirs(output_car_dir, exist_ok=True)

output_color_dir = f"dataset/dataset_{int(time.time())}/color"
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
    "fidelity": 82, "dignity": 83, "lamari": 84
}


color_classes = {
    "تیتانیوم": 0, "سرمه‌ای": 1, "قرمز": 2, "آبی": 3, "سبز": 4, "زرد": 5, "سفید": 6, "مشکی": 7,
    "آلبالویی": 8, "اطلسی": 9, "بادمجانی": 10, "برنز": 11, "بژ": 12, "بنفش": 13, "پوست پیازی": 14,
    "تیتانیئم": 15, "خاکستری": 16, "خاکی": 17, "دلفینی": 18, "ذغالی": 19, "زرد زرشکی": 20,
    "زیتونی ": 21, "سربی": 22, "سفید صدفی": 23, "طلایی": 24, "طوسی": 25, "عدسی": 26,
    "عنابی": 27, "قهوه ای": 28, "کربن بلک": 29, "کرم": 30, "گیلاسی": 31, "مسی": 32, "موکا": 33,
    "نارنجی": 34, "نقرآبی": 35, "نقره ای": 36, "نوک مدادی": 37, "یشمی": 38
}

# color_normalizer = {
#     # 🔵 Blue variants
#     "آبی": "آبی",
#     "نقرآبی": "آبی",
#     "اطلسی": "آبی",
#     "سرمه‌ای": "آبی",

#     # 🔴 Red variants
#     "قرمز": "قرمز",
#     "آلبالویی": "قرمز",
#     "گیلاسی": "قرمز",
#     "عنابی": "قرمز",

#     # ⚫ Black variants
#     "مشکی": "مشکی",
#     "کربن بلک": "مشکی",

#     # ⚪ White variants
#     "سفید": "سفید",
#     "سفید صدفی": "سفید",

#     # ⚙️ Gray / Silver variants
#     "نقره ای": "خاکستری",
#     "تیتانیوم": "خاکستری",
#     "تیتانیئم": "خاکستری",
#     "خاکستری": "خاکستری",
#     "دلفینی": "خاکستری",
#     "ذغالی": "خاکستری",
#     "نوک مدادی": "خاکستری",
#     "سربی": "خاکستری",

#     # 🟤 Brown variants
#     "قهوه ای": "قهوه ای",
#     "مسی": "قهوه ای",
#     "موکا": "قهوه ای",

#     # 🟡 Yellow/Gold variants
#     "زرد": "زرد",
#     "طلایی": "زرد",
#     "زرد زرشکی": "زرد",

#     # 🟢 Green variants
#     "سبز": "سبز",
#     "زیتونی": "سبز",
#     "یشمی": "سبز",
#     "خاکی": "سبز",

#     # 🟠 Orange variants
#     "نارنجی": "نارنجی",

#     # 🟣 Purple variants
#     "بنفش": "بنفش",
#     "بادمجانی": "بنفش",

#     # 🟡 Beige-like variants
#     "بژ": "بژ",
#     "کرم": "بژ",
#     "پوست پیازی": "بژ",
#     "عدسی": "بژ",

#     # 🟤 Bronze-like
#     "برنز": "قهوه ای",

#     # 🌀 Other variants mapped loosely
#     "طوسی": "خاکستری",
# }

color_normalizer = {
    # Persian : (Base Persian Color, English Translation)
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

# dataset = {
#     "path": "datasets",
#     "train": "train",
#     "val": "val",
#     "names": interestCars  
# }

# yaml_path = os.path.join(output_dir, "customdata.yaml")
# with open(yaml_path, "w") as yaml_file:
#     yaml.dump(dataset, yaml_file, default_flow_style=False, allow_unicode=True, sort_keys=False) 

options = Options()
# options.add_argument("--headless")  
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
# driver.set_page_load_timeout(2) 


try:
    driver.get("https://divar.ir/s/iran/car/hyundai/elantra?cities=33%2C708%2C4%2C1737%2C1723%2C1747%2C1727%2C1750%2C1724%2C1725%2C1744%2C849%2C1745%2C1746%2C30%2C848%2C1749%2C1748%2C31%2C2200%2C2201%2C2202%2C2203%2C2204%2C2205%2C2206%2C2207%2C2208%2C2209%2C2210%2C2211%2C2212%2C2213%2C2214%2C2215%2C2216%2C2217%2C2218%2C2219%2C2220%2C2221%2C2222%2C2223%2C2224%2C2225%2C2226%2C2227%2C2228%2C2229%2C2230%2C2231%2C2232%2C2233%2C2234%2C2235%2C2236%2C2237%2C2238%2C2239%2C5%2C853%2C759%2C760%2C852%2C761%2C762%2C763%2C764%2C10%2C859%2C765%2C766%2C767%2C857%2C768%2C858%2C856%2C769%2C792%2C770%2C17%2C771%2C772%2C1741%2C1743%2C773%2C1742%2C2%2C1722%2C1721%2C1739%2C1740%2C850%2C1751%2C1738%2C1720%2C1753%2C1752%2C774%2C1754%2C1%2C1709%2C1715%2C1714%2C29%2C1764%2C1707%2C1768%2C1760%2C1767%2C1766%2C781%2C1718%2C782%2C783%2C1765%2C1769%2C1763%2C1713%2C1717%2C1759%2C1712%2C1710%2C1716%2C1772%2C1770%2C1758%2C1761%2C1708%2C1719%2C1706%2C1771%2C784%2C1711%2C1762")
except:
    driver.execute_script("window.stop()") 
time.sleep(2) 
wait = WebDriverWait(driver, 2)

last_ad_count = 0 
ad_links = []
unique_links = set()
while len(unique_links)<612:
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
        

print(len(unique_links))
for idx, ad_link in enumerate(unique_links):

    try:
        # driver.get(ad_link)
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
        # color_label = self.color_classes.get(color_str, self.default_color_class)
        color_label = normalize_color(color_str)

        value_text = driver.find_element(By.XPATH, "//div[p[text()='بدنه']]/following-sibling::div//div[contains(@class, 'kt-score-row__score')]").text
        if "تصادفی" in value_text:
            # print("Ignoring this ad (تصادفی found).")
            continue
        # Detect car model                            
        detected_model = None
        if len(modelItemsEnglish) > 2 and modelItemsEnglish[-2] != 'other':
            for model in car_classes.keys():
                if model in modelItemsEnglish[4] or (len(modelItemsEnglish) > 5 and model in modelItemsEnglish[5]) or \
                (len(modelItemsEnglish) > 5 and model in (modelItemsEnglish[4].replace('-ir', '') + modelItemsEnglish[5].replace('-ir', '')) ):
                    detected_model = model
                    break
            if detected_model is not None and any(item in modelItemsPersian for item in titlePersian):
                        # print("car data matched")
                        timestamp = int(time.time())
                        print(f"!!!  {idx}/{len(unique_links)} {token} {ad_link} {detected_model}" , )

                        images = driver.find_elements(By.XPATH,f"//img[contains(@class, 'kt-image-block__image')]")
                                                    #   and contains(@alt, {titlePersian[1]})]")
                        for i, img in enumerate(images):
                            img_url = img.get_attribute("src")
                            nameing = f"{token}_{detected_model}_{i}_{timestamp}"
                            image_filename = os.path.join(output_dir,nameing+".jpg")

                            urllib.request.urlretrieve(img_url, image_filename)
                            # time.sleep(2)

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
                                    0.3 <= aspect_ratio <= 1.49):
                                    result.boxes = [largest_box] 
                                    # detected_image = result.plot()  

                                    # output_image_path = os.path.join(output_dir, f"detected_dataset/{token}_{detected_model}_{i}_{timestamp}.jpg")
                                    # os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                                    # cv2.imwrite(output_image_path, detected_image)

                                    car_filename = os.path.join(output_car_dir, nameing+".txt")

                                    with open(car_filename, "w") as f:
                                        x_center, y_center, width, height = largest_box.xywhn[0]
                                        class_id = car_classes[detected_model]
                                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


                                    color_filename = os.path.join(output_color_dir+'', nameing+".txt")

                                    with open(color_filename, "w") as f:
                                        x_center, y_center, width, height = largest_box.xywhn[0]
                                        class_id = color_classes[color_label[0]]
                                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                                        
                                        
                                    # # Resize the detected image to a smaller size 
                                    # height,width,_= rawImage.shape
                                    # scale_x = 0.19
                                    # scale_y = 0.19
                                    # new_width = int(width * scale_x)
                                    # new_height = int(height * scale_y)
                                    # resized_image = cv2.resize(rawImage, (new_width, new_height))
                                    # output_image_path_resized = os.path.join(output_dir, f"{token}_{detected_model}_{timestamp}_resized_.jpg")
                                    # cv2.imwrite(output_image_path_resized, resized_image)
                                    # # Get original box coordinates
                                    # x1, y1, x2, y2 = largest_box.xyxy[0]

                                    # # Scale bounding box to resized image
                                    # x1_resized = int(x1 * scale_x)
                                    # y1_resized = int(y1 * scale_y)
                                    # x2_resized = int(x2 * scale_x)
                                    # y2_resized = int(y2 * scale_y)

                                    # # Draw the resized bounding box on the resized image
                                    # cv2.rectangle(resized_image, (x1_resized, y1_resized), (x2_resized, y2_resized), (0, 255, 0), 3)

                                    # # Save resized image
                                    # cv2.imwrite(os.path.join(output_dir, f"detected_dataset/{token}_{detected_model}_{timestamp}_resized_.jpg"), resized_image)

                                    # Save annotations for resized image
                                    # with open(f"{output_image_path_resized.split('.')[0]}.txt", "w") as f:
                                    #     # Normalize coordinates for YOLO format
                                    #     x_center_resized = (x1_resized + x2_resized) / (2 * new_width)
                                    #     y_center_resized = (y1_resized + y2_resized) / (2 * new_height)
                                    #     box_width_resized = (x2_resized - x1_resized) / new_width
                                    #     box_height_resized = (y2_resized - y1_resized) / new_height

                                    #     f.write(f"{class_id} {x_center_resized} {y_center_resized} {box_width_resized} {box_height_resized}\n")
                                        
                                    # break
                                else:
                                    print(f"Deleting {img_url}  ")
                                    # cv2.imwrite(os.path.join(output_dir, f"detected_dataset/resized_{detected_model}_{timestamp}.jpg"), resized_image)

                                    os.remove(image_filename)
                                    os.remove(car_filename)
                                    os.remove(color_filename)
    except Exception as e:
        print(f"Error processing {e}")
        continue


driver.quit()
