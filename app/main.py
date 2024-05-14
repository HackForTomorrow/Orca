import os
import logging
import openai
import whisper
import base64
import requests
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
openai_api_key = os.getenv("OPENAI_API_KEY")
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
                messenger.send_message(response, mobile)  
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
                
                # Converting audio to text using OpenAI's Whisper cloud model
                transcription_text = convert_audio_to_text(audio_filename)
                
                # Process the transcribed text and send a response
                response = process_message(transcription_text)
                print(response)
                messenger.send_message(response, mobile)
                print(transcription_text)
                
                text_to_wav("en-US-Wavenet-D", transcription_text)
                wav_filename = "en-US-Wavenet-D.wav"
                mp3_filename = "en-US-Wavenet-D.mp3"
                convert_wav_to_mp3(wav_filename, mp3_filename)

                messenger.send_audio(
                    audio=mp3_filename,
                    recipient_id=mobile,  # Use the provided mobile ID dynamically
                ) # logging.info(f"{mobile} sent audio {audio_filename}")

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
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    
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

def convert_wav_to_mp3(wav_filename, mp3_filename):
    audio = AudioSegment.from_wav(wav_filename)
    audio.export(mp3_filename, format="mp3")


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
        HumanMessage(content=context),  # Provide context
        HumanMessage(content=prompt)  # Provide prompt
    ])

    # Get the response from the model
    response = output.content
    global engtext
    engtext = response
    print(response)
    return response  # Return the response

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