import os
import re
import time
import urllib.request
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from ultralytics import YOLO

MODEL = YOLO("yolo11n.pt")

BASE_URL = "https://divar.ir"
CAR_CLASSES =  {
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

COLOR_CLASSES = {
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

COLOR_NORMALIZER =  {
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

def ensure_dirs(base):
    for sub in ("images", "labels/car", "labels/color"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

def setup_driver():
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
    return driver

def normalize_color(raw):
    clean = re.sub(r"[\u200c\u200f\u200e]", "", raw).strip()
    return COLOR_NORMALIZER.get(clean, (clean, "Unknown"))

def get_ad_links(driver, limit=None):
    links = set()
    wait = WebDriverWait(driver, 2)
    while True:
        # cards = driver.find_elements(By.CSS_SELECTOR, ".unsafe-kt-post-card__action")
        cards = driver.find_elements(By.CLASS_NAME, "unsafe-kt-post-card__action")

        for c in cards:
            href = c.get_attribute("href")
            if href:
                links.add(urljoin(BASE_URL, href))
                if limit and len(links) >= limit:
                    return list(links)
        # attempt scroll
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
        time.sleep(1)
        try:
            btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".post-list__load-more-btn-be092")))
            btn.click()
        except:
            break
    return list(links)

def process_ad(link, base_dir):
    driver = setup_driver()
    driver.get(link)
    wait = WebDriverWait(driver, 10)
    # metadata
    token = urlparse(link).path.split('/')[-1]
    # parse color
    color_raw = driver.find_element(By.XPATH, "(//tr[@class='kt-group-row__data-row'])[1]/td[3]").text
    color_label, _ = normalize_color(color_raw)
    # parse model
    model_href = driver.find_element(By.CLASS_NAME, "kt-unexpandable-row__action").get_attribute("href")
    detected_model = next((m for m in CAR_CLASSES if m in model_href), None)
    if not detected_model:
        driver.quit()
        return []
    # get images
    imgs = driver.find_elements(By.CSS_SELECTOR, "img.kt-image-block__image")
    paths = []
    for i, img in enumerate(imgs):
        src = img.get_attribute("src")
        if not src or 'mapimage' in src:
            continue
        name = f"{token}_{detected_model}_{i}"
        img_path = os.path.join(base_dir, 'images', f"{name}.jpg")
        try:
            urllib.request.urlretrieve(src, img_path)
            paths.append((img_path, detected_model, color_label))
        except:
            continue
    driver.quit()
    # batch inference
    results = MODEL.predict([p[0] for p in paths], imgsz=640, batch=8)
    saved = []
    for res, (img_path, model_name, color) in zip(results, paths):
        if res.boxes:
            # pick largest box
            box = max(res.boxes, key=lambda b: b.xywh[0][2] * b.xywh[0][3])
            cls_id = int(box.cls[0])
            if cls_id in (2,7):
                x,y,w,h = box.xywhn[0]
                label_dir = os.path.join(base_dir, 'labels')
                car_file = os.path.join(label_dir, 'car', os.path.splitext(os.path.basename(img_path))[0] + ".txt")
                color_file = os.path.join(label_dir, 'color', os.path.splitext(os.path.basename(img_path))[0] + ".txt")
                with open(car_file, 'w') as f:
                    f.write(f"{CAR_CLASSES[model_name]} {x} {y} {w} {h}\n")
                with open(color_file, 'w') as f:
                    f.write(f"{COLOR_CLASSES.get(color,0)} {x} {y} {w} {h}\n")
                saved.append(img_path)
            else:
                os.remove(img_path)
        else:
            os.remove(img_path)
    return saved

# Main runner with concurrency
if __name__ == '__main__':
    target_url = "https://divar.ir/s/iran/car/renault/koleos?brand_model=Renault%20Koleos%20second%20generation%2CRenault%20Koleos%20first%20generation"
    model_name = urlparse(target_url).path.split('/')[-1]
    out = f"dataset/dataset_{model_name}"
    ensure_dirs(out)
    driver = setup_driver()
    driver.get(target_url)
    ad_links = get_ad_links(driver, limit=1000)
    driver.quit()
    print(f"Found {len(ad_links)} ads")

    all_saved = []
    with ThreadPoolExecutor(max_workers=8) as exe:
        futures = {exe.submit(process_ad, link, out): link for link in ad_links}
        for fut in as_completed(futures):
            try:
                saved = fut.result()
                all_saved.extend(saved)
                print(f"Saved {len(saved)} from {futures[fut]}")
            except Exception as e:
                print(f"Error processing {futures[fut]}: {e}")
    print(f"Total saved images: {len(all_saved)}")
