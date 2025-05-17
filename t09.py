import os
import re
import time
import traceback
import urllib.request
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from ultralytics import YOLO

DeepModel = YOLO("yolo11n.pt")

car_classes = {
    "rira": 0, "206": 1, "207": 2, "405": 3, "pars": 4, "samand": 5, "dena": 6, "runna": 7, "tara": 8, "haima": 9, "arisun": 10,
    "sahand": 11, "tondar-90": 12, "pride": 13, "tiba": 14, "quick": 15, "saina": 16, "shahin": 17, "zamyad": 18, "atlas": 19,
    "paykan": 20, "rio": 21, "arizo": 22, "kmct9": 23, "mazdakara": 24, "changan": 25, "brilliance": 26, "mazda3": 27,
    "mvm110": 28, "mvm110s": 29, "mvm315": 30, "mvm530": 31, "mvm550": 32, "mvmx22": 33, "mvmx33": 34, "mvmx55": 35,
    "j5": 36, "j4": 37, "s5": 38, "s3": 39,
    "lifan620": 40, "lifanx50": 41, "lifanx60": 42, "lifan520": 43, "lifan820": 44, "lifanx70": 45,
    "accent": 46, "avante": 47, "elantra": 48, "verna": 49, "azera-grandeur": 50, "sonata": 51, "tucson": 52, "santafe": 53, "i20": 54, "i30": 55, "i40": 56,
    "cerato": 57, "optima": 58, "sportage": 59, "sorento": 60, "carnival": 61,
    "tiggo-5": 62, "tiggo-7": 63, "fx": 64, "arizo-8": 65, "tiggo-7-pro": 66, "tiggo-8-pro": 67,
    "duster": 68, "fluence": 69, "koleos": 70, "sandero": 71, "megan": 72, "symbol": 73, "talisman": 74,
    "aurion": 75, "camry": 76, "corolla": 77, "prius": 78, "yaris": 79, "prado": 80, "landcruiser": 81,
    "fidelity": 82, "dignity": 83, "lamari": 84
}


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
    "بژ":14
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
    "کربنبلک": ("مشکی", "Black"),

    "سفید": ("سفید", "White"),
    "سفیدصدفی": ("سفید", "White"),

    "نقرهای": ("خاکستری", "Gray"),
    "تیتانیوم": ("خاکستری", "Gray"),
    "تیتانیئم": ("خاکستری", "Gray"),
    "خاکستری": ("خاکستری", "Gray"),
    "دلفینی": ("خاکستری", "Gray"),
    "ذغالی": ("خاکستری", "Gray"),
    "نوک مدادی": ("خاکستری", "Gray"),
    "سربی": ("خاکستری", "Gray"),
    "طوسی": ("خاکستری", "Gray"),

    "قهوهای": ("قهوه ای", "Brown"),
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
    "پوستپیازی": ("بژ", "Beige"),
    "عدسی": ("بژ", "Beige"),

    
}

def setup_driver():
    chrome_options = Options()
    # options.add_argument("--headless")  
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(10)
    return driver

def normalize_color(raw_color: str):
    clean = re.sub(r'[\u200c\u200f\u200e]', '', raw_color.strip())
    return color_normalizer.get(clean, (clean, "Unknown"))

def save_label(filename, class_id, box):
    x_center, y_center, width, height = box.xywhn[0]
    with open(filename, "w") as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def get_ad_links(driver, expected_count=1912):
    wait = WebDriverWait(driver, 5)
    links = set()
    last_count = 0

    while len(links) < expected_count:
        time.sleep(2)
        try:
            ads = driver.find_elements(By.CLASS_NAME, "unsafe-kt-post-card__action")
            for ad in ads:
                href = ad.get_attribute("href")
                if href and not href.startswith("http"):
                    href = "https://divar.ir" + href
                links.add(href)
            if len(links) == last_count:
                break
            last_count = len(links)

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            load_btn = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "post-list__load-more-btn-be092")))
            load_btn.click()
        except Exception:
            continue
    return list(links)

def process_ad(driver, link, out_dir):
    try:
        driver.get(link)
        time.sleep(3)

        token = link.rstrip('/').split('/')[-1]
        image_elements = driver.find_elements(By.XPATH, "//img[contains(@class, 'kt-image-block__image')]")

        color_str = driver.find_element(By.XPATH, "(//tr[@class='kt-group-row__data-row'])[1]/td[3]").text.strip()
        color_label = normalize_color(color_str)
        body_status = driver.find_element(By.XPATH, "//div[p[text()='بدنه']]/following-sibling::div//div[contains(@class, 'kt-score-row__score')]").text
        if "تصادفی" in body_status:
            return

        model_items = urlparse(driver.find_element(By.CLASS_NAME, "kt-unexpandable-row__action").get_attribute("href")).path.split("/")
        detected_model = next(
            (model for model in car_classes if 
            (len(model_items) > 4 and model in model_items[4] )
            or 
            (len(model_items) > 5 and (
                model in model_items[5] or 
                model in (model_items[4].replace('-ir', '') + model_items[5].replace('-ir', '')))
            )), 
            None
        )

        if not detected_model:
            return

        for i, img_el in enumerate(image_elements):
            img_url = img_el.get_attribute("src")
            if not img_url or "mapimage" in img_url:
                continue

            timestamp = int(time.time())
            name = f"{token}_{detected_model}_{i}_{timestamp}"
            img_path = os.path.join(out_dir, name + ".jpg")
            car_label_path = os.path.join(out_dir, name + ".txt")
            color_label_path = os.path.join(out_dir, name + "c.txt")

            try:
                urllib.request.urlretrieve(img_url, img_path)
                result = DeepModel(img_path)[0]

                if result.boxes and len(result.boxes) > 1:
                    largest = max(result.boxes, key=lambda b: b.xywh[0][2] * b.xywh[0][3])
                    cls_id = int(largest.cls[0].item())

                    if cls_id in [2, 7]:
                        save_label(car_label_path, car_classes[detected_model], largest)
                        save_label(color_label_path, color_classes.get(color_label[0], 0), largest)
                        print(f"Saved {img_path}")
                    else:
                        os.remove(img_path)
                else:
                    os.remove(img_path)
            except Exception as e:
                print(f"Image Error: {e}")
                traceback.print_exc()
                for f in [img_path, car_label_path, color_label_path]:
                    if os.path.exists(f):
                        os.remove(f)

    except Exception as e:
        print(f"Ad Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    url = "https://divar.ir/s/iran/car/mvm/x55?brand_model=MVM%20X55%20Pro%2CMVM%20X55%20Pro%20IE%2CMVM%20X55%20Pro%20IE%20Sport%2CMVM%20X55%20Pro%20Excellent-sport%2CMVM%20X55%20Pro%20Excellent%2CMVM%20X55%20Excellent%2CMVM%20X55%20Excellent-sport"

    model_name = urlparse(url).path.split('/')[-1]

    out_dir = f"dataset/dataset_{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    driver = setup_driver()
    try:
        driver.get(url)
        links = get_ad_links(driver)
        print(f"Collected {len(links)} ad links")

        for i, ad_link in enumerate(links):
            process_ad(driver, ad_link, out_dir)
            print(f"{i}/{len(links)} ")

    finally:
        driver.quit()
