import os
import logging
import openai
import whisper
import base64
import requests
import urllib.parse
from io import BytesIO
from PIL import Image
from heyoo import WhatsApp
from dotenv import load_dotenv
from flask import Flask, request, make_response
import google.cloud.texttospeech as tts
from openai import OpenAI
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from pydub import AudioSegment
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate





# Initialize Flask App
app = Flask(__name__)

# Load .env file
load_dotenv()
access_token = os.getenv("ACCESS_TOKEN")
phone_number_id = os.getenv("PHONE_NUMBER_ID")
if not access_token or not phone_number_id:
    logging.error("Missing ACCESS_TOKEN or PHONE_NUMBER_ID environment variable")
    exit(1)

messenger = WhatsApp(access_token, phone_number_id)
VERIFY_TOKEN = "12345"
user_greeted = {}
openai_api_key = ""
google_api_key = os.getenv("GOOGLE_API_KEY")
service_account_path = "./orca-sa.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
client = OpenAI()
# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@app.get("/webhook")
def verify_token():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        logging.info("Verified webhook")
        response = make_response(request.args.get("hub.challenge"), 200)
        response.mimetype = "text/plain"
        return response
    logging.error("Webhook Verification failed")
    return "Invalid verification token"

@app.post("/webhook")
def hook():
    # Handle Webhook Subscriptions
    data = request.get_json()
    
    changed_field = messenger.changed_field(data)
    if changed_field == "messages":
        new_message = messenger.is_message(data)
        if new_message:
            mobile = messenger.get_mobile(data)
            name = messenger.get_name(data)
            message_type = messenger.get_message_type(data)
            logging.info(f"New Message; sender:{mobile} name:{name} type:{message_type}")


            # Check if the user has been greeted already
            if not user_greeted.get(mobile, False):
                # Send greeting and mark user as greeted
                messenger.send_message(f"Hi {name}, I am your chatbot. Send me a message.", mobile)
                user_greeted[mobile] = True

            # Now handle different message types without repeating the greeting
            if message_type == "text":
                message = messenger.get_message(data)
                response = process_message(message)
                print(response)
                translated_text = translate_text(response, "ml")
                text_to_wav("ml-IN-Wavenet-D", translated_text)
                messenger.send_message(translated_text, mobile)  
                # messenger.send_audio(audio="temp.ogg", mobile)
            #     messenger.send_button(
            #     recipient_id="+919048806904",
            #     button={
            #         "header": "Header Testing",
            #         "body": "Body Testing",
            #         "footer": "Footer Testing",
            #         "action": {
            #             "button": "Button Testing",
            #             "sections": [
            #                 {
            #                     "title": "iBank",
            #                     "rows": [
            #                         {"id": "row 1", "title": "Send Money", "description": ""},
            #                         {
            #                             "id": "row 2",
            #                             "title": "Withdraw money",
            #                             "description": "",
            #                         },
            #                     ],
            #                 }
            #             ],
            #         },
            #     },
            # )


            elif message_type == "interactive":
                message_response = messenger.get_interactive_response(data)
                interactive_type = message_response.get("type")
                message_id = message_response[interactive_type]["id"]
                message_text = message_response[interactive_type]["title"]
                logging.info(f"Interactive Message; {message_id}: {message_text}")
                messenger.send_message(f"Interactive Message; {message_id}: {message_text}", mobile)

            elif message_type == "location":
                message_location = messenger.get_location(data)
                message_latitude = message_location["latitude"]
                message_longitude = message_location["longitude"]
                logging.info("Location: %s, %s", message_latitude, message_longitude)
                messenger.send_message(f"Location: {message_latitude}, {message_longitude}", mobile)
                fetch_device_details("iphone 15")

            elif message_type == "image":
                image = messenger.get_image(data)
                image_id, mime_type = image["id"], image["mime_type"]
                image_url = messenger.query_media_url(image_id)
                image_filename = messenger.download_media(image_url, mime_type)  # Assumes this function returns the local file path
                logging.info(f"Processing image from file: {image_filename}")
                response = process_image(image_filename)
                messenger.send_message(response, mobile)
                logging.info(f"{mobile} sent image {image_id}")

            elif message_type == "video":
                video = messenger.get_video(data)
                video_id, mime_type = video["id"], video["mime_type"]
                video_url = messenger.query_media_url(video_id)
                video_filename = messenger.download_media(video_url, mime_type)
                logging.info(f"{mobile} sent video {video_filename}")

            elif message_type == "audio":
                audio = messenger.get_audio(data)
                audio_id, mime_type = audio["id"], audio["mime_type"]
                audio_url = messenger.query_media_url(audio_id)
                audio_filename = messenger.download_media(audio_url, mime_type)

                # Converts the audio file to text
                transcription_text = convert_audio_to_text(audio_filename)

                # Process the transcribed text and send a response
                response = process_message(transcription_text)
                print(response)
                messenger.send_message(response, mobile)
                print(transcription_text)

                # Text to WAV using Google Cloud Text-to-Speech
                text_to_wav("en-US-Wavenet-D", response)
                wav_filename = "en-US-Wavenet-D.wav"
                mp3_filename = "en-US-Wavenet-D.mp3"
                convert_wav_to_mp3(wav_filename, mp3_filename)

                # Send the generated audio file using local path
                send_local_audio(mp3_filename, mobile)

            elif message_type == "document":
                file = messenger.get_document(data)
                file_id, mime_type = file["id"], file["mime_type"]
                file_url = messenger.query_media_url(file_id)
                file_filename = messenger.download_media(file_url, mime_type)
                logging.info(f"{mobile} sent file {file_filename}")
            else:
                logging.info(f"{mobile} sent {message_type} ")
                logging.info(data)
        else:
            delivery = messenger.get_delivery(data)
            if delivery:
                logging.info(f"Message : {delivery}")
            else:
                logging.info("No new message")
    return "OK", 200

def text_to_wav(voice_name: str, text: str):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    # Create the client using the explicit credentials
    client = texttospeech.TextToSpeechClient.from_service_account_file(service_account_path)

    try:
        response = client.synthesize_speech(
            input=text_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        filename = f"{voice_name}.wav"
        with open(filename, "wb") as out:
            out.write(response.audio_content)
            print(f'Generated speech saved to "{filename}"')

    except Exception as e:
        print(f"Error: {e}")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_filename):
    """
    Process the locally downloaded image and return a description using GPT-4's vision capabilities.
    """
    logging.info(f"Attempting to process local image file: {image_filename}")
    try:
        # Encode the image in base64
        base64_image = encode_image(image_filename)
        
        # Create the payload as per the documentation
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        payload = {
            "model": "gpt-4o",  # Ensure this matches the correct model ID from the documentation
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 300
        }

        # Send the request to OpenAI API
        api_response = requests.post(
            "https://api.openai.com/v1/chat/completions",  # Replace with the correct endpoint
            headers=headers,
            json=payload
        )

        if api_response.status_code != 200:
            logging.error(f"Failed to get a response from the API. Status code: {api_response.status_code}")
            logging.error(f"API Response: {api_response.text}")
            return "Failed to get a response from the API."

        response_text = api_response.json()["choices"][0]["message"]["content"]
        logging.info("Received response from GPT-4")
        return response_text.strip()
    
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return "An error occurred while processing the image."

def convert_audio_to_text(audio_filename):
    mp3_filename = "temp.ogg"
    with open(mp3_filename, "rb") as audio_file:
            transcript = client.audio.translations.create(
                model="whisper-1",
                file=audio_file     
            )
    return transcript.text

client1 = translate.Client()


def translate_text(text, target_language) -> str:
    result = client1.translate(text, target_language=target_language)
    print("Original Text: {}".format(result['input']))
    print("Translated Text: {}".format(result['translatedText']))
    print("Detected Source Language: {}".format(result['detectedSourceLanguage']))
    
    mal_string = result['translatedText']
    return mal_string


def fetch_device_details(device_name):
    if  not device_name:
        print("No device name selected. Please use the /selectdevice command first.")
    formatted_device_name = urllib.parse.quote(device_name.strip(), safe='+')
    
    print(formatted_device_name)
    # Send request to Google SERP API's Shopping API
    url = f"https://serpapi.com/search?engine=google_shopping&q={formatted_device_name}&api_key={google_api_key}&gl=in&img=1"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        details = data.get('shopping_results')
        if details:
            trusted_platforms = ["Amazon","Flipkart"] 
            trusted_results = [item for item in details if item.get('source') in trusted_platforms]
            if trusted_results:
                sorted_trusted_results = sorted(trusted_results, key=lambda x: x.get('price', float('inf')))
                result = sorted_trusted_results[0]
                platform = result.get('source')
                price = result.get('price')
                link = result.get('link')
                message = f"Platform: {platform}\nPrice: {price}\nURL: {link}\n\n"
                messenger.send_message(message, "+919048806904")
                print(message)
            # return "\n".join(device_info)
                
    except requests.exceptions.RequestException as e:
        print(f"Error fetching device details: {e}")

    return None


def convert_wav_to_mp3(wav_filename, mp3_filename):
    audio = AudioSegment.from_wav(wav_filename)
    audio.export(mp3_filename, format="mp3")

def send_local_audio(file_path, recipient_id):
    url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # Determine correct MIME type for the file and set the MIME type in the upload request.
    mime_type = 'audio/mpeg'  # Assuming you're using MP3 format; adjust as needed
    with open(file_path, 'rb') as file_data:
        files = {
            'file': (os.path.basename(file_path), file_data, mime_type)
        }
        response_media = requests.post(
            f"https://graph.facebook.com/v18.0/{phone_number_id}/media",
            headers=headers,
            files=files,
            data={"messaging_product": "whatsapp", "type": "audio"}
        )

    if response_media.status_code == 200:
        media_id = response_media.json()["id"]
        logging.info(f"Uploaded media ID: {media_id}")
        
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "audio",
            "audio": {"id": media_id}
        }

        response_message = requests.post(url, headers=headers, json=payload)
        if response_message.status_code == 200:
            logging.info(f"Audio message sent to {recipient_id} successfully.")
        else:
            logging.error(f"Failed to send audio message to {recipient_id}. Status Code: {response_message.status_code}, Response: {response_message.text}")
    else:
        logging.error(f"Failed to upload media. Status Code: {response_media.status_code}, Response: {response_media.text}")



def generate_prompt(description, product_keywords):
    prompt = "User Description: " + description + "i dont have any technical knowledge.explain it to me considering that"
    if product_keywords:
        prompt += " " + ", ".join(product_keywords)
    return prompt

def train_model(prompt, context):
    global device_selectedd
    chat_model = ChatOpenAI(
        temperature=0,  
        model="gpt-4o",
        openai_api_key=openai_api_key,
        max_tokens=350
    )
    output = chat_model([
        HumanMessage(content=context),  
        HumanMessage(content=prompt)  
    ])

    # Get the response from the model
    response = output.content
    global engtext
    engtext = response
    print(response)
    return response  

def process_message(message):
    description = message
    product_keywords = extract_product_keywords(description)
    prompt = generate_prompt(description, product_keywords)
    context_message = "Context: Orca is an AI assistant that provides recommendations about devices based on user requirements."
    response1 = train_model(prompt,context_message)
    return response1

def extract_product_keywords(description):
    # Read keywords from keywords.txt
    with open("keywords.txt", "r") as file:
        relevant_keywords = [line.strip() for line in file]

    # Check if any product keywords are present in the description
    found_keywords = [keyword for keyword in relevant_keywords if keyword in description]
    print(found_keywords)
    return found_keywords



if __name__ == "__main__":
    app.run(port=8000, debug=False)