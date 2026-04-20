# Plant Disease Detection (AWS + PyTorch)

This project predicts plant leaf diseases for three crops (`Banana`, `Corn`, `Grapes`) using a PyTorch CNN model hosted on Amazon SageMaker.

It includes:
- A SageMaker inference script (`inference.py`)
- An AWS Lambda function that calls the SageMaker endpoint and sends SNS alerts (`Lambda function.py`)
- A local client script that sends images to the API (`predict.py`)
- A notebook with deployment/testing snippets (`model.ipynb`)

## Project Goal

Given a crop name and a leaf image, the system returns the predicted disease class.

If the predicted class is not `healthy`, Lambda sends an SNS notification.

## High-Level Architecture

1. `predict.py` converts an image to Base64 and sends JSON to the API endpoint.
2. API Gateway triggers `lambda_handler` in `Lambda function.py`.
3. Lambda calls the SageMaker endpoint (`invoke_endpoint`).
4. SageMaker loads the crop-specific model in `inference.py` and predicts a disease class.
5. Lambda returns prediction response and publishes SNS alert when disease is not healthy.

## Repository Structure

```text
P1/
|- inference.py          # SageMaker model loading, input parsing, prediction, output formatting
|- Lambda function.py    # API/Lambda logic + SNS notification
|- predict.py            # Local client script to send image + crop to deployed API
|- model.ipynb           # Notebook snippets for deploy/test endpoint
|- test1.jpg             # Sample image (currently in root)
|- requirements.txt      # Python dependencies (currently empty)
|- Data/                 # Dataset directory (currently empty)
`- models/               # Trained model files directory (currently empty)
```

## Model Details

`inference.py` defines `CNN_Model` with:
- 3 convolution layers
- Max pooling
- Fully connected layers with dropout
- 4 output classes per crop

SageMaker model loading behavior:
- `model_fn` expects these files inside the model artifact directory:
	- `Banana_disease.pth`
	- `Corn_disease.pth`
	- `Grapes_disease.pth`

Prediction labels:
- `Banana`: `cordana`, `healthy`, `pestalotiopsis`, `sigatoka`
- `Corn`: `Common_Rust`, `Gray_Leaf_Spot`, `Healthy`, `Northern_Leaf_Blight`
- `Grapes`: `Black_Root`, `Esca`, `Leaf_Blight`, `Healthy`

## Input/Output Contract

Request (JSON):

```json
{
	"crop": "Banana",
	"image": "<base64-encoded-image>"
}
```

Response from inference:

```json
{
	"prediction": "pestalotiopsis",
	"crop": "Banana"
}
```

## File-by-File Explanation

### `inference.py`
- Implements SageMaker-compatible functions:
	- `model_fn(model_dir)` loads all crop models.
	- `input_fn(request_body, request_content_type)` reads JSON + Base64 image.
	- `predict_fn(input_data, models)` applies transform and runs selected crop model.
	- `output_fn(prediction_data, content_type)` returns JSON response.

### `Lambda function.py`
- Reads API Gateway request body.
- Calls SageMaker endpoint using `boto3.client("sagemaker-runtime")`.
- Sends SNS alert if prediction is not healthy.
- Returns HTTP response with predicted disease and crop.

Hard-coded configuration in this file:
- `ENDPOINT_NAME`
- `SNS_TOPIC_ARN`

### `predict.py`
- Local script for manual testing.
- Reads image from local path, converts to Base64, and sends to API URL.
- User chooses crop from console input.

Hard-coded configuration in this file:
- `API_URL`
- `image_path`

### `model.ipynb`
- Contains deployment and quick endpoint testing snippets using SageMaker SDK.
- Prints deployed endpoint name.

## Setup Instructions

## 1. Create and activate virtual environment (local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

`requirements.txt` is currently empty, so install manually:

```powershell
pip install torch torchvision pillow requests boto3 sagemaker
```

After validating your environment, you can save them:

```powershell
pip freeze > requirements.txt
```

## 3. Prepare trained model artifacts

Ensure SageMaker model package contains expected files:
- `Banana_disease.pth`
- `Corn_disease.pth`
- `Grapes_disease.pth`

## 4. Deploy to SageMaker

Use logic from `model.ipynb` (or equivalent script):
- Create `PyTorchModel`
- Provide `model_data` S3 path
- Deploy endpoint

## 5. Configure Lambda

Update these values in `Lambda function.py`:
- `ENDPOINT_NAME`
- `SNS_TOPIC_ARN`

Deploy Lambda and connect it to API Gateway.

## 6. Test end-to-end

Update `predict.py`:
- `API_URL` to your API Gateway invoke URL
- `image_path` to a real local image path

Run:

```powershell
python predict.py
```



