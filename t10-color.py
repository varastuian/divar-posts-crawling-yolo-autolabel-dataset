import os
import re
import time
import traceback
import urllib.request
# from urllib.parse import urlparse
from urllib.parse import urlparse, parse_qs, unquote

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from ultralytics import YOLO

DeepModel = YOLO("yolo11n.pt")

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

        
        for i, img_el in enumerate(image_elements):
            img_url = img_el.get_attribute("src")
            if not img_url or "mapimage" in img_url:
                continue

            timestamp = int(time.time())
            name = f"{token}_{i}_{timestamp}"
            img_path = os.path.join(out_dir, name + ".jpg")
            color_label_path = os.path.join(out_dir, name + ".txt")

            try:
                urllib.request.urlretrieve(img_url, img_path)
                result = DeepModel(img_path)[0]

                if result.boxes and len(result.boxes) > 1:
                    largest = max(result.boxes, key=lambda b: b.xywh[0][2] * b.xywh[0][3])
                    save_label(color_label_path, color_classes.get(color_label[0], 0), largest)
                    print(f"Saved {img_path}")
                else:
                    os.remove(img_path)
            except Exception as e:
                print(f"Image Error: {e}")
                traceback.print_exc()
                for f in [img_path, color_label_path]:
                    if os.path.exists(f):
                        os.remove(f)

    except Exception as e:
        print(f"Ad Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    urls = [
            # 'https://divar.ir/s/iran/car?color=%D8%B3%D8%A8%D8%B2',
            # 'https://divar.ir/s/iran/car?color=%D9%82%D8%B1%D9%85%D8%B2',
            # 'https://divar.ir/s/iran/car?color=%D9%86%D8%A7%D8%B1%D9%86%D8%AC%DB%8C',
            'https://divar.ir/s/iran/car?color=%D8%B2%D8%B1%D8%AF',
            # 'https://divar.ir/s/iran/car?color=%D8%B3%D8%B1%D9%85%D9%87%E2%80%8C%D8%A7%DB%8C',
            # 'https://divar.ir/s/iran/car?color=%D8%A2%D8%A8%DB%8C',
            # 'https://divar.ir/s/iran/car?color=%D8%A8%D8%A7%D8%AF%D9%85%D8%AC%D8%A7%D9%86%DB%8C',
            # 'https://divar.ir/s/iran/car?color=%D9%82%D9%87%D9%88%D9%87%E2%80%8C%D8%A7%DB%8C',
            # 'https://divar.ir/s/iran/car?color=%D9%86%D9%88%DA%A9%E2%80%8C%D9%85%D8%AF%D8%A7%D8%AF%DB%8C',
            
             ]

    for url in urls:

        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        color_encoded = query_params.get('color', [''])[0]
        color = unquote(color_encoded)

        out_dir = f"dataset/dataset_{color}"

        driver = setup_driver()
        try:
            driver.get(url)
            links = get_ad_links(driver)
            print(f"Collected {len(links)} ad links")
            os.makedirs(out_dir, exist_ok=True)

            for i, ad_link in enumerate(links):
                process_ad(driver, ad_link, out_dir)
                print(f"{i}/{len(links)} ")
        except Exception as e:
            print(f"!!!!!!!!!!!!!!saggggggError loading URL {url}: {e}")
                
        finally:
            driver.quit()
