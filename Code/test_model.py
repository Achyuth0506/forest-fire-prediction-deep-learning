import cv2
import time
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from selenium import webdriver

# Load the model (assuming it's already loaded from previous code)
# model = ... (loaded from saved model)

clas = ["no fire", "fire"]

def test_image(path):
    # Test single image
    img = load_img(path, target_size=(224,224))
    img1 = cv2.imread(path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    re = cv2.imread(path)
    img = cv2.cvtColor(re, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    last = time.time()
    pred = model.predict(img)
    print(f"Prediction time: {time.time()-last} seconds")
    
    op = clas[np.argmax(pred)]
    e = np.argmax(pred)
    x = round(pred[0][e]*100, 2)
    conf = str(x) + '%'
    
    print(f"Confidence: {x}%")
    print(f"Prediction: {op}")
    
    op = op + '-' + conf
    
    if x > 80.0:
        cv2.putText(img1, op, (25,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,200,0), 3)
    
    plt.imshow(img1)
    plt.show()
    
    return e, x

# SENDING WHATSAPP ALERT USING SELENIUM
def send_whatsapp_alert():
    filename = "test_image.jpg"  # Replace with actual photo filename
    
    # Test the image
    e, confidence = test_image(filename)
    
    if e == 1 and confidence > 80.0:
        driver = webdriver.Chrome('/content/drive/My Drive/selenium/chromedriver_win32/chromedriver.exe')
        driver.get('http://web.whatsapp.com')
        
        msg = "FIRE DETECTED PLEASE HELP..."
        name = 'Fire'  # Contact name or group name
        
        input("Enter after screen load")
        
        user = driver.find_element_by_xpath('//span[@title = "{}"]'.format(name))
        user.click()
        
        msg_box = driver.find_element_by_xpath('//*[@id="main"]/footer/div[1]/div[2]/div/div[2]')
        msg_box.send_keys(msg)
        
        driver.find_element_by_xpath('//*[@id="main"]/footer/div[1]/div[3]/button/span').click()

# Example usage
if __name__ == "__main__":
    # Test different images
    test_paths = [
        "/content/drive/MyDrive/archive/test/1/100.jpg",  # Fire image
        "/content/drive/MyDrive/archive/test/0/450.jpg"   # No fire image
    ]
    
    for path in test_paths:
        print(f"\nTesting: {path}")
        result, confidence = test_image(path)
        print(f"Result: {'Fire' if result == 1 else 'No Fire'} with {confidence}% confidence")