import json
import boto3

runtime = boto3.client("sagemaker-runtime")
sns = boto3.client("sns")

ENDPOINT_NAME = "pytorch-inference-2026-04-12-08-15-35-794"
SNS_TOPIC_ARN = "arn:aws:sns:ap-south-1:260317865695:PlantDiseaseAlerts:f46c389a-l19f-45k2-95hd-0bc741tg761e"


def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        crop = body["crop"]

        # SageMaker
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=event["body"]
        )
        result = json.loads(response["Body"].read().decode())
        prediction = result["prediction"]


        if prediction.lower() != "healthy": #SNS
            message = f"Alert! {prediction} detected in your {crop} crop."
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=message,
                Subject="Plant Disease Detection Alert"
            )

        return {
            "statusCode": 200,
            "body": json.dumps({"Predicted Disease": prediction, "Crop": crop})
        }
    except Exception as e:
        return {"statusCode": 500, "body": str(e)}