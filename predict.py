import requests
import base64

API_URL = "https://XXXXXXXXXXXXX"

def predict_disease(image_path, crop):
    # Convert image to base64
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    data = {
        "crop": crop,
        "image": img_base64
    }

    response = requests.post(API_URL, json=data)

    # result
    if response.status_code == 200:
        result = response.json()


        prediction = result.get("prediction")
        crop_name = result.get("crop")

        print(f"Prediction: {prediction}, Crop: {crop_name}")
        # print(result)
    else:
        print("Error:", response.text)



image_path = r"G:\Crop Diseases Dataset\Banana Leaf Spot Diseases (BananaLSD) Dataset for Classification of Banana Leaf Diseases Using Machine Learning\BananaLSD\OriginalSet\pestalotiopsis\9.jpeg"

print("1. Banana")
print("2. Grapes")
print("3. Corn")

s = int(input("Enter your Crop no. : "))

if s == 1:
    crop = 'Banana'
elif s == 2:
    crop = 'Grapes'
elif s == 3:
    crop = "Corn"
else:
    print("Invalid choice")
    exit()

predict_disease(image_path, crop)