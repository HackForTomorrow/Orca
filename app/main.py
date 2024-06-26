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
from supabase import create_client, Client
from datetime import datetime ,timezone, timedelta
import re
import json
import serpapi


# Initialize Flask App
app = Flask(__name__)

# Load .env file
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client: Client = create_client(supabase_url, supabase_key)
access_token = os.getenv("ACCESS_TOKEN")
phone_number_id = os.getenv("PHONE_NUMBER_ID")
serp_api=os.getenv("SERP_API")
if not access_token or not phone_number_id:
    logging.error("Missing ACCESS_TOKEN or PHONE_NUMBER_ID environment variable")
    exit(1)

messenger = WhatsApp(access_token, phone_number_id)
VERIFY_TOKEN = "12345"
user_greeted = {}
processed_image = {}
openai_api_key = os.getenv("OPENAI_API_KEY")
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

gptresponse_dict = {}
image_device_names = {}
user_preferences = {}

@app.post("/webhook")
def hook():
   
    # Handle Webhook Subscriptions
    data = request.get_json()
    global mobile,processed_image
   
    
    changed_field = messenger.changed_field(data)
    if changed_field == "messages":
        new_message = messenger.is_message(data)
        if new_message:
            message_id= messenger.get_message_id(data)
            mark_whatsapp_message_as_read(message_id)
            mobile = messenger.get_mobile(data)
            name = messenger.get_name(data)
            message_type = messenger.get_message_type(data)
            logging.info(f"New Message; sender:{mobile} name:{name} type:{message_type}")
            record_session(mobile)
            user_result = supabase_client.table("users").select("*").eq("mobile", mobile).execute()
            user_registered = len(user_result.data) > 0
            # Check if the user has been greeted already
            if mobile not in user_greeted:
                    if not user_registered:
                        # User is not registered, so register and send the first greeting
                        supabase_client.table("users").insert({"mobile": mobile, "name": name}).execute()
                        messenger.send_message(f"Hello {name}, Thank you for choosing Orca! I am here to assist you. How can I help you today? ", mobile)
                        
                    else:
                        # User is registered, send normal greeting
                        messenger.send_message(f"Welcome back {name} 😁, how can I assist you today?", mobile)
                    # send_language_selection(mobile)
                    user_greeted[mobile] = True
                    processed_image[mobile] = False

            if mobile not in gptresponse_dict:
                gptresponse_dict[mobile] = True
            # Now handle different message types without repeating the greeting
            if message_type == "text":
                message = messenger.get_message(data)
                print('messageeee',message)
                detected_language = detect_lang(message)
                greetings = ["hi", "hello"]
                is_greeting = any(greeting in message.lower() for greeting in greetings)

                if not is_greeting:
            # Process non-greeting messages
                    if gptresponse_dict.get(mobile):
                        # Get a response from GPT
                        translated_message = translate_text(message,'en')
                        print('translateeeee',translated_message)
                        response = process_message(translated_message)
                        translated_text = translate_text(response,detected_language)
                        messenger.send_message(translated_text,mobile)
                        # messenger.send_message(response, mobile)
                        print(response)
                        if not processed_image.get(mobile):
                                send_reply_button(mobile)
                        # Translate the response to Malayalam (ml)
                       
                        # text_to_wav("ml-IN-Wavenet-D", translated_text)
                        
                        # Disable GPT response for subsequent messages
                        gptresponse_dict[mobile] = False

                        # Send the reply button once after the greeting phase
                        

                else:
                    # First greeting from the user
                    if not user_greeted.get(mobile):
                        response = f"Hi {name}, how can I assist you today?"
                        messenger.send_message(response, mobile)
                        user_greeted[mobile] = True



            elif message_type == "interactive":
                gptresponse_dict[mobile] = False
                message_response = messenger.get_interactive_response(data)
                interactive_type = message_response.get("type")
                message_id = message_response[interactive_type]["id"]
                message_text = message_response[interactive_type]["title"]
                logging.info(f"Interactive Message; {message_id}: {message_text}")
                device_names = load_device_names_from_json("deviceNames.json")
                rows = [{"id": f"row {index}", "title": device_name, "description": ""} for index, device_name in enumerate(device_names, start=1)]

                # messenger.send_message(f"Interactive Message; {message_id}: {message_text}", mobile)
                if message_id == "b1":
                    messenger.send_message("Please select the device name", mobile)
                                    # Use the rows in the send_button call
                    messenger.send_button(
                        recipient_id=mobile,  # make sure 'mobile' contains the correct recipient id
                        button={
                            "header": "",
                            "body": "Devices List",
                            "footer": "",
                            "action": {
                                "button": "Select one device",
                                "sections": [
                                    {
                                        "title": "Devices",
                                        "rows": rows,
                                    }
                                ],
                            },
                        },
                    )

                                # Assuming fetch_device_details requires a device name argument.
                                # The device name should come from previous conversation context or set by some other logic.
                                # device_name = "iphone 12"  # This is just an example. Replace with actual device name.
                                # fetch_device_details(device_name)
                elif message_id == "b2":
                    gptresponse_dict[mobile] = True

                elif message_id == "b3":
                    device_selected = image_device_names[mobile]
                    messenger.send_message(f"Device selected: {device_selected}", mobile)
                    fetch_device_details(device_selected)  


                elif message_id=="b4":
                    response = send_whatsapp_location_request()
                    
                    print(response)

                # elif message_id.startswith('lang_'):
                #     selected_language = message_id.split('_')[1] 
                #     logging.info(f"Language selected: {selected_language}")
                #     user_preferences[mobile] = {'language': selected_language}
                #     user_language = user_preferences.get(mobile, {}).get('language')
                #     messenger.send_message(f"You have selected {message_text}.", mobile)
                #     gptresponse_dict[mobile] = True

                else:
                    messenger.send_message(f"Device selected: {message_text}", mobile)
                    detected_language = detect_lang(message_text)
                    print("huuuuuuuu",detected_language)
                    res = get_device_response(message_text)
                    trans = translate_text(res,detected_language)
                    print(type(res))
                    messenger.send_message(trans, mobile)
                    fetch_device_details(message_text)
                    send_whatsapp_location_request()





# Notes:
# - Each 'row' dictionary now has a unique id of the format 'row_INDEX', where INDEX is the 1-based index.
# - The 'fetch_device_details' function will be called with the title (which is the device name) from the matching row.



# Notes:
# - Each 'row' dictionary now has a unique id of the format 'row_INDEX', where INDEX is the 1-based index.
# - The 'fetch_device_details' function will be called with the title (which is the device name) from the matching row.


            elif message_type == "location":
                message_location = messenger.get_location(data)
                message_latitude = message_location["latitude"]
                message_longitude = message_location["longitude"]
                logging.info("Location: %s, %s", message_latitude, message_longitude)
                # messenger.send_message(f"Location: {message_latitude}, {message_longitude}", mobile)
                # location_name = get_location_name(message_latitude, message_longitude)
                # messenger.send_message(f"Location: {location_name}", mobile)
                params = {
                    "engine": "google_maps",
                    "q": f"electronics store near me ",
                    "ll":"@{lat},{lon},15.1z".format(lat=message_latitude, lon=message_longitude),
                    "type": "search",
                    "api_key": "e1ccc210769c27c7d9428210ded95882c315b66fb08d03de78806aacebfd9454"
                    }

                search = serpapi.search(params)
                results = search.as_dict()
                local_results = results.get("local_results", [])

                # Limit the response to the nearest five shops if there are more than five
                nearest_shops = local_results[:5]
                final_message="Here are the closest electronics stores where you might find one."
                messenger.send_message(final_message, mobile)# Display the store info
                # Iterate over each result to extract the store name and address
                for store in nearest_shops:
                    store_name = store.get("title")
                    store_address = store.get("address")
                    store_phone=store.get("phone")
                    store_website=store.get("website")
                    
                    # Now do something with each store name and address
                    if store_name and store_address and store_phone:
                        # Combine store name and address
                        store_info = f"*Name: {store_name}*\n\n*Address:* {store_address}\n\n *Phone:* {store_phone}\n\n"

        # Include the Link heading only if a website is available
                        store_website = store.get("website")
                        if store_website:
                            store_info += f"*Link:* {store_website}\n"
                        
                        messenger.send_message(store_info, mobile)  # Send the store info as a message



            elif message_type == "image":
                image = messenger.get_image(data)
                image_id, mime_type = image["id"], image["mime_type"]
                image_url = messenger.query_media_url(image_id)
                image_filename = messenger.download_media(image_url, mime_type)
                logging.info(f"Processing image from file: {image_filename}")
                
                # Process the image and get an initial response
                initial_response = process_image(image_filename)

                # Save the initial response into the conversation history
                store_message(mobile, "assistant", initial_response)

                # Use the existing 'process_message' function to get the name of the device from the response.
                gpt_prompt_for_device_name = "From the following description, return only the device name as a single word ,not as a sentence for example only (for example Apple Iphone,Samsung s24,etc): " + initial_response
                device_name_response = process_message(gpt_prompt_for_device_name)
                
                # Send the initial response followed by the device name to WhatsApp
                messenger.send_message(initial_response, mobile)
                if device_name_response:
                    image_device_names[mobile] = device_name_response
                
                # Set processed_image to True for this mobile number
                processed_image[mobile] = True

                # Send the image specific reply buttons
                send_image_reply_button(mobile)
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
                if gptresponse_dict.get(mobile):
                # Process the transcribed text and send a response
                    response = process_message(transcription_text)
                    print(response)
                    messenger.send_message(response, mobile)
                    print(transcription_text)
                    if not processed_image.get(mobile):
                        send_reply_button(mobile)

                    # Text to WAV using Google Cloud Text-to-Speech
                    text_to_wav("en-US-Wavenet-D", response)
                    # media_id = messenger.upload_media(
                    # media='ml-IN-Wavenet-D.wav',
                    # )['id']
                    # messenger.send_audio(
                    #     audio=media_id,
                    #     recipient_id=mobile,
                    #     link=False
                    # )
                    # Send the generated audio file using local path
                    send_local_audio("en-US-Wavenet-D.wav", mobile)
                    gptresponse_dict[mobile] = False


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


# def send_language_selection(recipient_id):
#     messenger.send_reply_button(
#         recipient_id=recipient_id,
#         button={
#             "type": "button",
#             "body": {
#                 "text": "Please select a language"
#             },
#             "action": {
#                 "buttons": [
#                     {
#                         "type": "reply",
#                         "reply": {
#                             "id": "lang_en",
#                             "title": "English"
#                         }
#                     },
#                     {
#                         "type": "reply",
#                         "reply": {
#                             "id": "lang_ml",
#                             "title": "Malayalam"
#                         }
#                     },
#                     {
#                         "type": "reply",
#                         "reply": {
#                             "id": "lang_hi",
#                             "title": "Hindi"
#                         }
#                     }
#                     # Add more language options here...
#                 ]
#             }
#         },
#     )


## this function fetches the title from the created json file as button to whstapp user
def load_device_names_from_json(file_path):
    # Read the JSON file
    with open(file_path, "r") as f:
        # Deserialize JSON content into a Python object (in this case, a list)
        device_names = json.load(f)
    return device_names

def nearby_store(recipient_id):
    messenger.send_reply_button(
        recipient_id=recipient_id,
        button={
            "type": "button",
            "body": {
                "text": ""
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": "b4",
                            "title": "Find Near By Devices"
                        }
                    }
                ]
            }
        },
    )


def send_reply_button(recipient_id):
    messenger.send_reply_button(
        recipient_id=recipient_id,
        button={
            "type": "button",
            "body": {
                "text": "Please select one"
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": "b1",
                            "title": "Select Device"
                        }
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "b2",
                            "title": "Ask more"
                        }
                    }
                ]
            }
        },
    )

def send_image_reply_button(recipient_id):
    messenger.send_reply_button(
        recipient_id=recipient_id,
        button={
            "type": "button",
            "body": {
                "text": "Please select one"
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": "b3",
                            "title": "Buy Now"
                        }
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "b2",
                            "title": "Ask more"
                        }
                    }
                ]
            }
        },
    )


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


# this is the  function for sendning the device title as json file by chatgpt

def extract_device_names_and_save_to_json(response):
    # Split the response by newline to get lines
    lines = response.split('\n')

    # Initialize an empty list to hold device names
    device_names = []

    # Define a regex pattern to extract device names (assuming they follow '### ')
    pattern = r"^### (.*)"

    # Loop over lines and use regex to match device names
    for line in lines:
        match = re.match(pattern, line)
        
        # If a match is found, append the device name to the device_names list
        if match:
            # The actual device name is in the first group of the match
            device_names.append(match.group(1))

    # Save the device names to a JSON file
    with open("deviceNames.json", "w") as f:
        json.dump(device_names, f)

    print("Device names have been saved to deviceNames.json")


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
                        {"type": "text", "text": "If the image is of a device or electronic appliance and fetch the specifications and details Else reply Sorry, I can't help with that. Can you try another image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 1000
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


def detect_lang(message)-> str:
    lang = client1.detect_language(message)
    detected = lang['language']
    print("languageeeeee",lang['language'])
    return detected


def translate_text(text, target_language) -> str:
    # Call the API translate method assuming it returns a dictionary with the translation
    result = client1.translate(text, target_language=target_language)
    # Accessing the translated text from the result dictionary
    translated_text = result.get('translatedText', '')
    
    if not isinstance(translated_text, str):
        raise TypeError("Expected 'result['translatedText']' to be a string")
        
    # Formatting each occurrence of '###' by ensuring it starts with a new line if needed
    formatted_text = re.sub(r'###', r'\n-------------------------------------\n', translated_text)
    # Formatting bold text by wrapping with newlines if needed (matching **bold**)
    formatted_text = re.sub(r'(?m)\*([^*]+)\*', r'\n*\1*\n', formatted_text)
    
    print("## Original Text")
    print("> {}".format(result['input']))
    print("\n## Translated Text")
    print("> {}".format(formatted_text))
    print("\n## Detected Source Language")
    print("> Code: {}".format(result['detectedSourceLanguage']))
    
    mal_string = formatted_text
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
                image_url = result.get('thumbnail')
                messenger.send_image(
                        image=image_url,
                        recipient_id=mobile,
                 )
                message = f"Platform: {platform}\nPrice: {price}\nURL: {link}\n\n"
                messenger.send_message(message, mobile)
                print(message)
            # return "\n".join(device_info)
                
    except requests.exceptions.RequestException as e:
        print(f"Error fetching device details: {e}")

    return None

def send_local_audio(file_path, recipient_id):
    url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # Determine correct MIME type for the file and set the MIME type in the upload request.
    mime_type = 'audio/wav'  # Assuming you're using MP3 format; adjust as needed
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

 

def process_message(message):
    user_id = mobile  # Ensure that this is defined globally or passed as an argument
    store_message(user_id, "user", message)

    # Retrieve the full conversation history for this user from your storage
    conversation = get_conversation(user_id)
    formatted_conversation = [{"role": msg['role'], "content": msg['content']} for msg in conversation]   

    # Get the latest message from the user and form the payload for OpenAI completion
    payload = [{
        "role": "system",
        "content": "Context: Orca is an AI assistant that provides only recommendations about electronic devices based on user requirements.No other requirements are dealt with.All other prombts must be avoided.Before the title of the device names you suggest put ###.Only before the device name it must be provided.Along with the title detailed descriptions and specifications about electronic devices must be provided.Device specifications strings should be enclosed in * and * instead of ** and **.Device name should only be maximum upto 23 characters"
    },
    {
        "role": "user",
        "content": message
    }]
    
    # Get the response from the model
    response_text = get_response(formatted_conversation + payload)        
    store_message(user_id, "assistant", response_text)       
    extract_device_names_and_save_to_json(response_text)
    return response_text


def get_response(conversation):
    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation
    )
    return response.choices[0].message.content 

def get_device_response(conversation):
    messages = [{"role": "user", "content": conversation}]
    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content 

def should_start_new_session(mobile: str) -> bool:
    # Define the threshold for when to consider starting a new session
    session_threshold = timedelta(minutes=5)
    
    # Get the current time in UTC
    now_utc = datetime.now(timezone.utc)
    
    # Query the 'sessions' table for the most recent session by 'mobile'
    response = supabase_client.table("sessions")\
        .select("timestamp")\
        .eq("mobile", mobile)\
        .order("timestamp")\
        .limit(1)\
        .execute()
        
    if response.data:
        # Parse the last session's timestamp and convert it to UTC
        last_session_timestamp = response.data[0]['timestamp']
        last_session_time = datetime.fromisoformat(last_session_timestamp).replace(tzinfo=timezone.utc)
        
        # Determine if the last session is older than the session threshold
        if now_utc - last_session_time > session_threshold:
            return True
    else:
        # No existing session for this mobile number, which means we should start a new session
        return True
    
    # Existing session is within the threshold limit
    return False
    

def record_session(mobile):
    if should_start_new_session(mobile):
        # Insert or update the session's timestamp to the current time for this mobile number
        supabase_client.table("sessions")\
            .insert({"mobile": mobile, "timestamp": datetime.now(timezone.utc).isoformat()}, upsert=True)\
            .execute()
        # Return True to indicate that a new session was started
        return True
    # Return False to indicate that the existing session was kept alive
    return False

def store_message(user_id, role, content):
    data = {
        "user_id": user_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    supabase_client.table("conversations").insert(data).execute()

def get_conversation(user_id):
    five_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
    
    response = supabase_client.table("conversations")\
        .select("*")\
        .eq("user_id", user_id)\
        .gt("created_at", five_minutes_ago.isoformat())\
        .order("created_at")\
        .execute()
    
    return response.data





def send_whatsapp_location_request():
    url = f"https://graph.facebook.com/v15.0/{phone_number_id}/messages"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "type": "interactive",
        "to": mobile,
        "interactive": {
            "type": "location_request_message",
            "body": {
                "text": "Wanna find nearby stores?Share your Location😁."
            },
            "action": {
                "name": "send_location"
            }
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def mark_whatsapp_message_as_read(message_id):
    url = f'https://graph.facebook.com/v20.0/{phone_number_id}/messages'
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


if __name__ == "__main__":
    app.run(port=8000, debug=False)