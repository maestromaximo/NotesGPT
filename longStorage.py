import datetime
import itertools
import random
from dotenv import load_dotenv
import tiktoken
import os
import openai
import pinecone
#import torch
from PyPDF2 import PdfReader
import PyPDF2
from tqdm import tqdm
import pickle
import jsonpickle
import re
import sys
import asyncio
import itertools

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

import nltk
from rake_nltk import Rake
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk.tokenize import word_tokenize


import pytz

import telegram
# from telegram import Bot, Update
# from telegram.ext import Updater, CommandHandler, MessageHandler, filters, CallbackContext
# from telegram import Update
# from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext




import threading
import time

import tkinter as tk
from tkinter import filedialog

import pytesseract
from PIL import Image
from pdf2image import convert_from_path

import tiktoken


from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import wolframalpha

nltk.download("punkt")

class Course:
    def __init__(self, name, code, folder):
        self.name = name
        self.code = code
        self.folder = folder
        self.main_note = None

    def set_main_note(self, main_note):
        self.main_note = main_note

    def __str__(self):
        return f"Course: {self.name} ({self.code})\nFolder: {self.folder}\nMain Note: {self.main_note}"

class TableOfContents:
    def __init__(self, contents):
        self.contents = contents

class Content:
    def __init__(self, chapter, name, startPage, endPage=None):
        self.chapter = chapter
        self.name = name
        self.startPage = startPage
        self.endPage = endPage

class MainNote:
    def __init__(self, name, total_pages, lessons, tableOfContents = None):
        self.name = name
        self.total_pages = total_pages
        self.lessons = None #lessons
        self.tableOfContents = tableOfContents

    def __str__(self):
        return f"Main Note: {self.name}\nTotal Pages: {self.total_pages}"
    
    def add_transcript_to_latest_lesson(self, transcript):
        if self.lessons:
            self.lessons[-1].transcript = transcript
    
    def addTableOfContentsManually(self):
        runninString = ""
        print("Type \'$$$\' to finish")
        print()
        contents = []
        while runninString != "$$$":
            chapter = input("Chapter: ")
            name = input("Name: ")
            startPage = input("Page")
            cont = Content(chapter, name, startPage)
            contents.append(cont)
            print("Saved!")
            print()
            runninString = input("Stop?: ")
            print()
        self.tableOfContents = TableOfContents(contents)
        print("[completed]")
        print()




    
    
    
class Lesson:
    def __init__(self, name, content,summary, listOfSubjects,firstPage, lastPage, date, dueDate, ease_factor=2.5, interval=1, repetitions=0, transcript=None):
        self.name = name
        self.content = content
        self.summary = summary
        self.listOfSubjects = listOfSubjects
        self.firstPage = firstPage
        self.lastPage = lastPage
        self.date = datetime.date.today()
        self.ease_factor = ease_factor
        self.interval = interval
        self.repetitions = repetitions
        self.due_date = datetime.date.today() + datetime.timedelta(days=self.interval)
        self.transcript = transcript
    
    def update(self, quality):
        if quality < 3:
            self.repetitions = 0
            self.interval = 1
        else:
            self.repetitions += 1
            if self.repetitions == 1:
                self.interval = 1
            elif self.repetitions == 2:
                self.interval = 6
            else:
                self.interval *= self.ease_factor

        self.ease_factor += 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        self.ease_factor = max(1.3, self.ease_factor)
        self.due_date += datetime.timedelta(days=int(self.interval))

class FlashCard:
    def __init__(self, front, back):
        """front is question and back is answer by convention"""
        self.front = front
        self.back = back
    
    def checkAnswerIntelligently(self, answer, gpt4):
        preCheck = True if answer.lower() in self.back.lower() else False

        if preCheck:
            return True
        else:
            response = askGpt(f"Given this question from a flashcard \"{self.front}\" and this being the correct answer \"{self.back}\", would you say that this attempt to answer the question is correct  \"{answer}\"? Only answer with \"TRUE\" if is correct, else with \"FALSE\"", gpt4)
            trueOrFalse = True if "true" in response.lower() else False
            return trueOrFalse
        
    def intelligentTest(self):
        print(f"Question: {self.front}")
        answ = input("Answer: ")
        if self.checkAnswerIntelligently(answ, True):
            print(f"Correct! Here is the back: {self.back}")
            print()
            return True
        else:
            print(f"Sorry! not correct, here is the back: {self.back}")
            print()
            return False
        
class Homework:
    def __init__(self, CourseCode, description, difficulty=3, dueDate=None, numberOfQuestions=None):
        self.CourseCode = CourseCode
        self.description = description
        self.difficulty = difficulty
        self.dueDate = dueDate
        self.numberOfQuestions = numberOfQuestions

    def __str__(self):
        return f"CourseCode: {self.CourseCode}, Description: {self.description}, Difficulty: {self.difficulty}, DueDate: {self.dueDate}, NumberOfQuestions: {self.numberOfQuestions}"

def generate_flashcards(transcript, n_keywords=10, window_size=1):
    # Initialize RAKE
    rake = Rake()

    # Extract keywords from the transcript
    rake.extract_keywords_from_text(transcript)
    key_phrases = rake.get_ranked_phrases()[:n_keywords]

    # Filter key phrases based on POS tags
    key_phrases = filter_key_phrases(key_phrases)

    # Tokenize the transcript into sentences
    sentences = sent_tokenize(transcript)

    # Create flashcards
    flashcards = []
    for key_phrase in key_phrases:
        for i, sentence in enumerate(sentences):
            if key_phrase in sentence:
                front = f"What is {key_phrase}?"
                back = sentence

                # Add context from surrounding sentences if needed
                if window_size > 1:
                    context = []
                    for j in range(i - window_size + 1, i + window_size):
                        if j >= 0 and j < len(sentences) and j != i:
                            context.append(sentences[j])

                    back = " ".join(context) + " " + back

                flashcards.append(FlashCard(front, back))
                break

    return flashcards


def filter_key_phrases(key_phrases, pos_tags=('NN', 'NNS', 'NNP', 'NNPS')):
    filtered_key_phrases = []
    for phrase in key_phrases:
        words = word_tokenize(phrase)
        tagged_words = pos_tag(words)
        if any(tag in pos_tags for _, tag in tagged_words):
            filtered_key_phrases.append(phrase)
    return filtered_key_phrases


def read_homework_file():
    file_name = "telegram_messages.txt"
    homework_list = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(", ")
            CourseCode = parts[0].upper().strip()
            description = parts[1].strip()
            difficulty = int(parts[2])
            dueDate = datetime.datetime.now() + datetime.timedelta(days=int(parts[3]))
            numberOfQuestions = int(parts[4])
            homework = Homework(CourseCode, description, difficulty, dueDate, numberOfQuestions)
            homework_list.append(homework)
    with open(file_name, "w") as file:
        file.write("")
    return homework_list

def schedule_homework_events(homework_list):
    for homework in homework_list:
        num_days = (homework.dueDate.date() - datetime.datetime.now().date()).days
        num_questions = homework.numberOfQuestions
        difficulty = homework.difficulty

        if num_days <= 0:
            print(f"Skipping {homework.description}: due date has already passed.")
            continue

        questions_per_day = [0] * num_days
        total_questions = 0

        for i in range(num_questions):
            index = i % num_days if difficulty < 3 else num_days - 1 - (i % num_days)
            questions_per_day[index] += 1
            total_questions += 1

        current_question = 1
        for day, num_questions in enumerate(questions_per_day):
            if num_questions == 0:
                continue

            start_question = current_question
            end_question = current_question + num_questions - 1
            title = f"{homework.description} ({start_question}-{end_question}/{homework.numberOfQuestions})"
            event_date = datetime.datetime.now().date() + datetime.timedelta(days=day + 1)
            schedule_nearest_event(title, "Homework", "", event_date)

            current_question += num_questions

# def schedule_homework_events(homework_list):
#     for homework in homework_list:
#         num_days = (homework.dueDate.date() - datetime.datetime.now().date()).days
#         num_questions = homework.numberOfQuestions
#         difficulty = homework.difficulty

#         if num_days <= 0:
#             print(f"Skipping {homework.description}: due date has already passed.")
#             continue

#         questions_per_day = [0] * num_days
#         total_questions = 0

#         for i in range(num_questions):
#             index = i % num_days if difficulty < 3 else num_days - 1 - (i % num_days)
#             questions_per_day[index] += 1
#             total_questions += 1

#         for day, num_questions in enumerate(questions_per_day):
#             if num_questions == 0:
#                 continue

#             start_question = total_questions - num_questions + 1
#             end_question = total_questions
#             title = f"{homework.description} ({start_question}-{end_question}/{homework.numberOfQuestions})"
#             event_date = datetime.datetime.now().date() + datetime.timedelta(days=day + 1)
#             schedule_nearest_event(title, "Homework", "", event_date)

#             total_questions -= num_questions

def run_threaded_read_homework_file():
    print("in Homework")
    homework_list = read_homework_file()
    schedule_homework_events(homework_list)
    threading.Timer(300, run_threaded_read_homework_file).start()  # Schedule the function to run again after 5 minutes (300 seconds)

def summarize_transcript(transcript):
    tokenTotal = numTokensFromString(transcript)
    maxTokens = 2950
    ratio = 0.75
    maxWords = int(maxTokens / ratio)
    finalText = transcript
    words = transcript.split()
    sections = []
    
    if tokenTotal > maxTokens:
        for i in range(0, len(words), maxWords):
            section = " ".join(words[i:i + maxWords])
            sections.append(section)
        
        while totalSectionTokenCount(sections) > maxTokens:
            print("Summarizing transcript sections...")
            new_sections = []
            for i, section in tqdm(enumerate(sections), total=len(sections)):
                while numTokensFromString(section) > 2000:
                    section_words = section.split()
                    split_idx = len(section_words) // 2
                    section1 = " ".join(section_words[:split_idx])
                    section2 = " ".join(section_words[split_idx:])
                    section = section1
                    new_sections.append(section2)
                summarized_section = askGpt(f"Please summarize this transcript section. It is a part of a bigger transcript and needs to be shortened. Here is the transcript: {section}", gpt4=False)
                new_sections.append(summarized_section)
            sections = new_sections
            print("Total Token count: " + str(totalSectionTokenCount(sections)))
        
        finalText = " ".join(sections)
        finalText = askGpt(f"Please summarize this pieced up together transcript to the best of your ability: {finalText}", gpt4=False)
    return finalText

def totalSectionTokenCount(sectionedTranscript):
    summ = 0
    for section in sectionedTranscript:
        summ += numTokensFromString(section)
    return summ

def review_due_lessons(courses, gpt4=False):
    for course in courses:
        main_note = course.main_note
        if main_note is None:
            print(f"No main note found for course: {course.name}")
            continue
        
        lessons = main_note.lessons
        if lessons is None:
            print(f"No lessons found for main note: {main_note.name}")
            continue
        
        for lesson in lessons:
            if lesson.due_date <= datetime.date.today():
                # Generate examples

                print()
                print(f"||| {lesson.name} Review |||")
                print()
                print("Lets first try some generated examples:")
                print()
                subjects = ', '.join(lesson.listOfSubjects) if lesson.listOfSubjects is not None else 'No subjects provided'# and noExample = True
                prompt = f"Generate examples for the lesson '{lesson.name}' with the following topics: {subjects}"
                examples = askGptContext(prompt, lesson.summary, gpt4)
                # Ask user for rating
                print(f"Examples for lesson '{lesson.name}':\n{examples}")

                f = input("Press any key to contiue.")
                
                if(lesson.transcript == None):
                    print("Could not generate flashcards, no transcript detected")
                else:
                    flashcards = generate_flashcards(lesson.transcript) 
                    print()
                    print("Lets review terms now:")
                    print()

                    totalCards = len(flashcard)
                    correct = 0
                    for flashcard in flashcards:
                        boool = flashcard.intelligentTest()
                        if boool:
                            correct += 1
                            ratio = correct/totalCards*100
                            print(f"Percent right: {ratio}%")
                        else:
                            ratio = correct/totalCards*100
                            print(f"Percent right: {ratio}%")
                            
                rating = int(input(f"Rate the lesson '{lesson.name}' from 1 to 5: "))
                while rating < 1 or rating > 5:
                    rating = int(input("Invalid input. Please rate the lesson from 1 to 5: "))
                
                # Update lesson
                lesson.update(rating)
    save_courses(courses, DATA_FILE)

def waiting_wheel():
    while True:
        for char in "|/-\\":
            sys.stdout.write(f'\r{char}')
            sys.stdout.flush()
            time.sleep(0.1)




def reviewLessons(course):
    due_lessons = [lesson for lesson in course.lessons if lesson.due_date <= datetime.date.today()]

    if not due_lessons:
        print("No lessons to review today!")
        return

    lesson = random.choice(due_lessons)
    print(f"Review: {lesson.content}")
    quality = int(input("Quality (0-5): "))
    lesson.update(quality)
    print(f"Next review in {lesson.interval} days.")


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.api_key = os.getenv("PINECONE_API_KEY")
pinecome_env = os.getenv("PINECONE_ENV")

WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")
wolframClient = wolframalpha.Client(WOLFRAM_APP_ID)


pytesseract.pytesseract.tesseract_cmd = "C:/Users/aleja/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"##change this to your own location

bot_token = os.getenv("TELEGRAM_TOKEN")
user_id =9999 ##change this to your own

####bot = telegram.Bot(token=bot_token)


#bot.send_message(chat_id=user_id, text="Hello from the Python script!")

REMARKABLE_FOLD_ID = "1CpSr4xKre26VB6q-iJokOHIsCTa_Q_sC"
CALC4_FOLD_ID = "1OlcJzaGqDONcJu08eRvXyrgOuAwnGr-Y"
ELECTRO1_FOLD_ID = "1XXA4uiCMLb22gP_Ga6gVO9BbhzUW86Ko"
MEC_SPECIAL_FOLD_ID = "1DcMuDlY6hS1YmTHa8wsQsmuikoGM-tHf"
QUANTUM1_FOLD_ID = "1Qfiv_m90UY8SjabCkR9tqOo__qf0VhTO"
THERMO_FOLD_ID = "1g3o26yOM9cEFjFiZ6vgGN0UxhLZfyAyl"


code_to_folder = {
    "THR": THERMO_FOLD_ID,
    "CA4": CALC4_FOLD_ID,
    "QM1": QUANTUM1_FOLD_ID,
    "MAS": MEC_SPECIAL_FOLD_ID,
    "EM1": ELECTRO1_FOLD_ID
    }

courses = None

CLIENT_SECRET_FILE = os.getenv("CLIENT_SECRET_FILE")
#SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
#SCOPES = ['https://www.googleapis.com/auth/drive']
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/calendar']




DATA_FILE = "courses_data.json"
SAVING_FOLDER = "C:/Users/aleja/OneDrive/Escritorio/RemarkableGPT/mainSaving"

pinecone.init(api_key="YOUR TOKEN", environment=pinecome_env)
#pinecone.create_index("chunks", dimension=1536)
index = pinecone.Index("chunks")

def driveProtocol2():
    #time.sleep(600)
    creds = get_credentials()
    service = build('drive', 'v3', credentials=creds)
   

    
    cleanAndSortDrive(service)
    time.sleep(5)
    remove_duplicates2(service)
    holderList = []

    #REMARKABLE_FOLD = ["REMARKABLE","1CpSr4xKre26VB6q-iJokOHIsCTa_Q_sC"]
    CALC4_FOLD = ["CALC4","1OlcJzaGqDONcJu08eRvXyrgOuAwnGr-Y"]
    ELECTRO1_FOLD = ["ELECTRO1", "1XXA4uiCMLb22gP_Ga6gVO9BbhzUW86Ko"]
    MEC_SPECIAL_FOLD = ["MEC_SPECIAL", "1DcMuDlY6hS1YmTHa8wsQsmuikoGM-tHf"]
    QUANTUM1_FOLD = ["QUANTUM1", "1Qfiv_m90UY8SjabCkR9tqOo__qf0VhTO"]
    THERMO_FOLD = ["THERMO", "1g3o26yOM9cEFjFiZ6vgGN0UxhLZfyAyl"]

    #holderList.append(REMARKABLE_FOLD)
    holderList.append(CALC4_FOLD)
    holderList.append(ELECTRO1_FOLD)
    holderList.append(MEC_SPECIAL_FOLD)
    holderList.append(QUANTUM1_FOLD)
    holderList.append(THERMO_FOLD)

    for folderObject in holderList:

        download_pdf_notes(service, folderObject[1])
    print("Synced Drive")
        #time.sleep(600)

def process_transcripts(service, course):
    folder_id = code_to_folder[course.code]
    files = list_files_in_folder(service, folder_id)
    for file in files:
        if "transcript" in file['name'].lower():
            # Download the transcript file
            transcript = download_file_content(service, file['id'])
            
            # Add the transcript to the latest lesson in the course's main_note
            course.main_note.add_transcript_to_latest_lesson(transcript)
            
            # Delete the transcript file from Google Drive
            delete_file(service, file['id'])


def delete_file(service, file_id):
    """Deletes a file from Google Drive."""
    try:
        service.files().delete(fileId=file_id).execute()
    except errors.HttpError as error:
        print(f'An error occurred: {error}')
        return None
    print(f'File with ID {file_id} was deleted.')


def driveProtocol():
    time.sleep(600)
    creds = get_credentials()
    service = build('drive', 'v3', credentials=creds)
    courses = load_courses(DATA_FILE)
    ####asyncio.run(sendBotMessage("Starting Sync"))
   
    while True:
        
        cleanAndSortDrive(service)
        remove_duplicates2(service)
        holderList = []

        #REMARKABLE_FOLD = ["REMARKABLE","1CpSr4xKre26VB6q-iJokOHIsCTa_Q_sC"]
        CALC4_FOLD = ["CALC4","1OlcJzaGqDONcJu08eRvXyrgOuAwnGr-Y"]
        ELECTRO1_FOLD = ["ELECTRO1", "1XXA4uiCMLb22gP_Ga6gVO9BbhzUW86Ko"]
        MEC_SPECIAL_FOLD = ["MEC_SPECIAL", "1DcMuDlY6hS1YmTHa8wsQsmuikoGM-tHf"]
        QUANTUM1_FOLD = ["QUANTUM1", "1Qfiv_m90UY8SjabCkR9tqOo__qf0VhTO"]
        THERMO_FOLD = ["THERMO", "1g3o26yOM9cEFjFiZ6vgGN0UxhLZfyAyl"]

        #holderList.append(REMARKABLE_FOLD)
        holderList.append(CALC4_FOLD)
        holderList.append(ELECTRO1_FOLD)
        holderList.append(MEC_SPECIAL_FOLD)
        holderList.append(QUANTUM1_FOLD)
        holderList.append(THERMO_FOLD)

        for folderObject in holderList:

            download_pdf_notes(service, folderObject[1])
        print("Synced Drive")
        for course in courses:
            file_path = os.path.join(course.folder, course.main_note.name)
            newLesson =checkIfNewLesson(course.main_note, file_path)
            updateLessons(course)
            process_transcripts(service, course)
            if newLesson:
                course.main_note.lessons[-1] = updateNewestWithTranscript(course)
        print("Courses saved and synced")
        save_courses(courses, DATA_FILE)
        
        time.sleep(600)
        

timer_thread = threading.Thread(target=driveProtocol, daemon=True)



def numTokensFromString(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(str(string)))
    return num_tokens

def read_pdf_and_chunk(file_path, chunk_size):
    """Reads a PDF file and splits its content into chunks of the specified size."""
    pdf_reader = PdfReader(file_path)
    text = ""

    for page_num in tqdm(range(len(pdf_reader.pages)), desc="Processing PDF"):
        text += pdf_reader.pages[page_num].extract_text()

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def remove_non_ascii(text):
    """Removes non-ASCII characters from a given text string."""
    return ''.join([i if ord(i) < 128 else '' for i in text])

# def store_chunks(text_chunks):
#     for chunk in text_chunks:
#         chunk_embedding = get_embedding(chunk)
#         upsert_response = index.upsert(vectors=[{"id": chunk, "values": chunk_embedding}])
def store_chunks(chunks):
    """Stores the given chunks and their embeddings in Pinecone."""
    for chunk in tqdm(chunks, desc="Storing chunks"):
        chunk_embedding = get_embedding(chunk)
        # Clean the chunk to create an ASCII-based ID
        chunk_id = remove_non_ascii(chunk)
        upsert_response = index.upsert(vectors=[{"id": chunk_id, "values": chunk_embedding}])


def query_chunks(query, top_k=5):
    """Queries Pinecone for the top K most relevant chunks based on the input query."""
    query_embedding = get_embedding(query)
    chunk_scores = index.query(vector=query_embedding, top_k=top_k, include_values=True)
    return [match for match in chunk_scores['matches']]


def get_embedding(text):
    """Generates an embedding for the given text using OpenAI's API."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings

def askGptContextOLD(prompt, chunks, gpt4):
    """Asks GPT model a question with the provided context from relevant chunks"""
    conversation = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"Here is context to my next question from my book \"{chunks}\""}, {"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model= "gpt-4" if gpt4 else "gpt-3.5-turbo",
        messages=conversation,
        max_tokens=3500,
        n=1,
        stop=None,
        temperature=0.1,
    )

    message = response.choices[0].message["content"].strip()
    return message

def askGptOLD(prompt, gpt4):
    """Asks GPT model a question """
    conversation = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model= "gpt-4" if gpt4 else "gpt-3.5-turbo",
        messages=conversation,
        max_tokens=3500,
        n=1,
        stop=None,
        temperature=0.1,
    )

    message = response.choices[0].message["content"].strip()
    return message

def askGptContext(prompt, chunks, gpt4):
    conversation = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"Here is context to my next question from my book \"{chunks}\""}, {"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model= "gpt-4" if gpt4 else "gpt-3.5-turbo",
        messages=conversation,
        max_tokens=3500,
        n=1,
        stop=None,
        temperature=0.1,
    )

    message = response.choices[0].message["content"].strip()

    # Count tokens
    prompt_tokens = numTokensFromString(prompt)
    response_tokens = numTokensFromString(message)
    total_tokens = prompt_tokens + response_tokens

    # Calculate cost
    cost_per_token = 0.06 if gpt4 else 0.002
    cost = (total_tokens / 1000) * cost_per_token

    # Update the cost file
    updateCostFile(cost)

    return message

def askGpt(prompt, gpt4):
    conversation = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model= "gpt-4" if gpt4 else "gpt-3.5-turbo",
        messages=conversation,
        max_tokens=3500,
        n=1,
        stop=None,
        temperature=0.1,
    )

    message = response.choices[0].message["content"].strip()

    # Count tokens
    prompt_tokens = numTokensFromString(prompt)
    response_tokens = numTokensFromString(message)
    total_tokens = prompt_tokens + response_tokens

    # Calculate cost
    cost_per_token = 0.06 if gpt4 else 0.002
    cost = (total_tokens / 1000) * cost_per_token

    # Update the cost file
    updateCostFile(cost)

    return message

def updateCostFile(cost: float) -> None:
    """Updates the costTracking.txt file with the new cost."""
    if not os.path.exists("costTracking.txt"):
        with open("costTracking.txt", "w") as f:
            f.write("0")
    
    with open("costTracking.txt", "r") as f:
        current_cost = float(f.read().strip())

    new_cost = current_cost + cost

    with open("costTracking.txt", "w") as f:
        f.write(str(new_cost))

def get_credentials():
    """Obtains Google API credentials from the 'token.pickle' file or a new token if necessary."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def remove_duplicates(service):
    """Removes duplicate files from the specified Google Drive folders."""
    for folder_id in code_to_folder.values():
        query = f"'{folder_id}' in parents"
        results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        file_count = {}
        for item in items:
            # Extract file name without extension and " (#)" suffix
            base_name = re.sub(r'\s\(\d+\)|\.\w+$', '', item['name'])

            if base_name not in file_count:
                file_count[base_name] = []

            file_count[base_name].append(item)

        for base_name, files in file_count.items():
            if len(files) > 1:
                files.sort(key=lambda x: x['name'])

                # Delete the file without parentheses
                service.files().delete(fileId=files[0]['id']).execute()

                # Rename the file with parentheses
                new_name = re.sub(r'\s\(\d+\)', '', files[1]['name'])
                service.files().update(fileId=files[1]['id'], body={"name": new_name}).execute()
def remove_duplicates2(service):
    """Removes duplicate files from the specified Google Drive folders."""
    for folder_id in code_to_folder.values():
        query = f"'{folder_id}' in parents"
        results = service.files().list(q=query, fields="nextPageToken, files(id, name, modifiedTime)").execute()
        items = results.get('files', [])

        file_count = {}
        for item in items:
            # Extract file name without extension and " (#)" suffix
            base_name = re.sub(r'\s\(\d+\)|\.\w+$', '', item['name'])

            if base_name not in file_count:
                file_count[base_name] = []

            file_count[base_name].append(item)

        for base_name, files in file_count.items():
            if len(files) > 1:
                # Sort the files by modifiedTime in descending order
                files.sort(key=lambda x: x['modifiedTime'], reverse=True)

                # Keep the most recent file
                most_recent_file = files.pop(0)

                # Delete all other files with the same name
                for file in files:
                    service.files().delete(fileId=file['id']).execute()
                    print(f"Deleted file: {file['name']}")

                # Rename the most recent file by removing the " (#)" suffix if present
                new_name = re.sub(r'\s\(\d+\)', '', most_recent_file['name'])
                service.files().update(fileId=most_recent_file['id'], body={"name": new_name}).execute()
            elif len(files) == 1:  # If there's only one file with the base name, no need to delete anything
                continue



def get_page_count(pdf_file_path):
    """Returns the number of pages in a PDF file."""
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        return len(pdf_reader.pages)


def getDownloadFolderName(folder_id):
    """Returns the name of the folder corresponding to a given folder ID."""
    holderList = []

    REMARKABLE_FOLD = ["REMARKABLE","1CpSr4xKre26VB6q-iJokOHIsCTa_Q_sC"]
    CALC4_FOLD = ["CALC4","1OlcJzaGqDONcJu08eRvXyrgOuAwnGr-Y"]
    ELECTRO1_FOLD = ["ELECTRO1", "1XXA4uiCMLb22gP_Ga6gVO9BbhzUW86Ko"]
    MEC_SPECIAL_FOLD = ["MEC_SPECIAL", "1DcMuDlY6hS1YmTHa8wsQsmuikoGM-tHf"]
    QUANTUM1_FOLD = ["QUANTUM1", "1Qfiv_m90UY8SjabCkR9tqOo__qf0VhTO"]
    THERMO_FOLD = ["THERMO", "1g3o26yOM9cEFjFiZ6vgGN0UxhLZfyAyl"]

    holderList.append(REMARKABLE_FOLD)
    holderList.append(CALC4_FOLD)
    holderList.append(ELECTRO1_FOLD)
    holderList.append(MEC_SPECIAL_FOLD)
    holderList.append(QUANTUM1_FOLD)
    holderList.append(THERMO_FOLD)

    for folder in holderList:
        #print(folder[1])
        if folder[1] == folder_id:
            return folder[0]
        else:
            continue
        

def move_file(service, file_id, new_folder_id):
    """Moves a file in Google Drive from its current location to a new folder."""
    file = service.files().get(fileId=file_id, fields='parents').execute()
    previous_parents = ",".join(file.get('parents'))
    file = service.files().update(
        fileId=file_id,
        addParents=new_folder_id,
        removeParents=previous_parents,
        fields='id, parents'
    ).execute()


def sort_files_by_code(service, code_to_folder):
    """Sorts files in Google Drive based on their code and moves them to corresponding folders."""
    root_folder_id = 'root'
    files = list_files_in_folder(service, root_folder_id)

    for file in files:
        file_name = file['name']
        file_id = file['id']
        code = file_name[:3]

        if code in code_to_folder:
            destination_folder_id = code_to_folder[code]
            move_file(service, file_id, destination_folder_id)
            print(f"Moved file '{file_name}' to folder with ID: {destination_folder_id}")

def list_files_in_folder(service, folder_id):
    """Lists all files in a specified Google Drive folder."""
    query = f"'{folder_id}' in parents and trashed = false"  # Exclude trashed files
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    return items


def cleanAndSortDrive(service):
    """Cleans and sorts files in Google Drive based on their code."""
    #root_folder_id = 'root'
    
    sort_files_by_code(service, code_to_folder)


##piclke
def save_courses(courses, filename):
    with open(filename, 'w') as outfile:
        json_data = jsonpickle.encode(courses)
        outfile.write(json_data)

def load_courses(filename):
    if not os.path.exists(filename):
        return []

    with open(filename, 'r') as infile:
        json_data = infile.read()
        courses = jsonpickle.decode(json_data)
    return courses



##pickle end

def download_pdf_notes(service, folder_id, file_name=None):
    """Downloads PDF notes from a specified Google Drive folder."""
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed = false"  # Exclude trashed files
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    
    subFoldName = getDownloadFolderName(folder_id)
    download_folder = SAVING_FOLDER + "/" + subFoldName + "/"
    if not items:
        print('No PDF files found.')
    else:
        # Create the download folder if it doesn't exist
        os.makedirs(download_folder, exist_ok=True)

        for item in items:
            if file_name is None or file_name == item['name']:
                print(f"Downloading PDF: {item['name']}")
                file_id = item['id']
                request = service.files().get_media(fileId=file_id)
                response = request.execute()

                # Save the file in the specified folder
                with open(os.path.join(download_folder, item['name']), "wb") as pdf_file:
                    pdf_file.write(response)
            else:
                print(f"Available PDF: {item['name']}")

def download_file_content(service, file_id):
    """Downloads the content of a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    response = request.execute()
    return response



def extract_text_from_page(pdf_path, page_number):
    """Extracts text from a specific page of a PDF file using OCR."""
    images = convert_from_path(pdf_path, dpi=600, first_page=page_number, last_page=page_number)
    if not images:
        return ""
    
    image = images[0]
    custom_config = r'-c preserve_interword_spaces=1 --psm 6 --oem 3 -l eng+equ'
    text = pytesseract.image_to_string(image, config=custom_config)

    return text

def getLessonTitle(pdf_path, page_number):
    """Extracts text from a specific page of a PDF file using OCR."""
    images = convert_from_path(pdf_path, dpi=600, first_page=page_number, last_page=page_number)
    if not images:
        return ""
    
    image = images[0]
    percentage_width = 0.52
    percentage_height = 0.07317

    cropped_image = crop_upper_right(image, percentage_width, percentage_height)

    #custom_config = r'-c preserve_interword_spaces=1 --psm 6 --oem 3 -l eng+equ'
    text = pytesseract.image_to_string(cropped_image)#, config=custom_config

    textt = askGpt(f"here is a OCR transcription of a title, please fix it if it has errors only and return ONLY the title. if it has no errors also ONLY return the title, you must say nothing else other than the fixed title. Here is the OCR title \"{text}\"", True)
    return textt

def crop_upper_right(image, percentage_width, percentage_height):
    width, height = image.size
    left = width * (1 - percentage_width)
    upper = 0
    right = width
    lower = height * percentage_height
    cropped_image = image.crop((left, upper, right, lower))
    return cropped_image

def read_pdf_and_chunk_by_page(file_path, page_number=None):
    """Reads a specific page of a PDF file and splits its content into chunks."""
    text = extract_text_from_page(file_path, page_number) if page_number else ""

    chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    return chunks

def initializeCourses():#def __init__(self, name,listOfSubjects,firstPage, lastPage, date, dueDate
    courses = []
    holderList = []

    #REMARKABLE_FOLD = ["REMARKABLE","1CpSr4xKre26VB6q-iJokOHIsCTa_Q_sC"]
    CALC4_FOLD = ["CALC4","1OlcJzaGqDONcJu08eRvXyrgOuAwnGr-Y"]
    ELECTRO1_FOLD = ["ELECTRO1", "1XXA4uiCMLb22gP_Ga6gVO9BbhzUW86Ko"]
    MEC_SPECIAL_FOLD = ["MEC_SPECIAL", "1DcMuDlY6hS1YmTHa8wsQsmuikoGM-tHf"]
    QUANTUM1_FOLD = ["QUANTUM1", "1Qfiv_m90UY8SjabCkR9tqOo__qf0VhTO"]
    THERMO_FOLD = ["THERMO", "1g3o26yOM9cEFjFiZ6vgGN0UxhLZfyAyl"]

    #holderList.append(REMARKABLE_FOLD)
    holderList.append(CALC4_FOLD)
    holderList.append(ELECTRO1_FOLD)
    holderList.append(MEC_SPECIAL_FOLD)
    holderList.append(QUANTUM1_FOLD)
    holderList.append(THERMO_FOLD)

    subFoldName = getDownloadFolderName(THERMO_FOLD[1])
    download_folder = os.path.join(SAVING_FOLDER, subFoldName)
    cr1 = Course("Thermodynamics", "THR", download_folder)
    courses.append(cr1)

    subFoldName = getDownloadFolderName(QUANTUM1_FOLD[1])
    download_folder = os.path.join(SAVING_FOLDER, subFoldName)
    cr2 = Course("Quantum 1", "QM1", download_folder)
    courses.append(cr2)

    subFoldName = getDownloadFolderName(MEC_SPECIAL_FOLD[1])
    download_folder = os.path.join(SAVING_FOLDER, subFoldName)
    cr3 = Course("Mec and Special", "MAS", download_folder)
    courses.append(cr3)

    subFoldName = getDownloadFolderName(QUANTUM1_FOLD[1])
    download_folder = os.path.join(SAVING_FOLDER, subFoldName)
    cr4 = Course("Electrodynamics 1", "EM1", download_folder)
    courses.append(cr4)

    subFoldName = getDownloadFolderName(QUANTUM1_FOLD[1])
    download_folder = os.path.join(SAVING_FOLDER, subFoldName)
    cr5 = Course("Calculus 4", "CA4", download_folder)
    courses.append(cr5)

    i = 0
    for course in courses:
        courses[i].folder
        for file in os.listdir(courses[i].folder):
            if "main" in file:
                file_path = os.path.join(course.folder, file)
                numPages = get_page_count(file_path)
                note = MainNote(file, numPages, None)
                lessonss = []
                lesson = getInitialLesson(file_path, numPages)
                lessonss.append(lesson)
                note.lessons = lessonss
                courses[i].set_main_note(note)
        i += 1
    
    return courses

def getInitialLesson(file_path, numPages):

    content = ""

    for i in range(numPages):
        content += extract_text_from_page(file_path, i)

    title = getLessonTitle(file_path, 1)
    question = f"I have used OCR to get the text from a lesson from my notes, it will have grammatical errors; however, I need you to please list to me a list of as many topics as you may detect on the notes, find all of them and return them on this format [topic1, topic2, topic3, ...] and so on, Only answer with the list, nothing else. The title of the lesson is \"{title}\" and here is the content: {content}"
    answer = askGpt(question, gpt4=True)
    topics = extract_topics_from_response(answer)
    question2 = f"Please provide a summary to this OCR transcription, keep in mind it will have many grammatical errors. I will also provide the name of the lesson and a list of topics inside. Content: \"{content}\"; title: \"{title}\";list of topics: \"{topics}\""
    gptContent = askGpt(question2, gpt4=False)

    theLesson = Lesson(title, content, gptContent, topics, 1, numPages, None, None, 2.5, 1, 0, None)

    return theLesson

def getNewestLesson(course):

    file_path = os.path.join(course.folder, course.main_note.name)
    lessons = course.main_note.lessons
    lastLessonNum = len(lessons) - 1
    numPages = get_page_count(file_path)

    startPage = lessons[lastLessonNum].lastPage + 1

    content = ""
    count = 0
    while count < (numPages - (startPage -1)):
        content += extract_text_from_page(file_path, startPage + count)
        count += 1

    title = getLessonTitle(file_path, startPage)
    question = f"I have used OCR to get the text from a lesson from my notes, it will have grammatical errors; however, I need you to please list to me a list of as many topics as you may detect on the notes, find all of them and return them on this format [topic1, topic2, topic3, ...] and so on, Only answer with the list, nothing else. The title of the lesson is \"{title}\" and here is the content: {content}"
    answer = askGpt(question, gpt4=True)
    topics = extract_topics_from_response(answer)
    question2 = f"Please provide a summary to this OCR transcription, keep in mind it will have many grammatical errors. I will also provide the name of the lesson and a list of topics inside. Content: \"{content}\"; title: \"{title}\";list of topics: \"{topics}\""
    gptContent = askGpt(question2, gpt4=False)

    theLesson = Lesson(title, content, gptContent, topics, startPage, numPages, None, None, 2.5, 1, 0, None)
    
    return theLesson

def updateNewestWithTranscript(course):#    def __init__(self, name, content,summary, listOfSubjects,firstPage, lastPage, date, dueDate, ease_factor=2.5, interval=1, repetitions=0, transcript=None):

    
    lessons = course.main_note.lessons
    
    if lessons[-1].transcript:

        summarizedTranscript = summarize_transcript(lessons[-1].transcript)
        title = askGpt(f"Given this summarized transcript for my lesson, what would be an accurate title for the lesson, return only the title: {summarizedTranscript}", gpt4=False)
        question = f"Here is a summary of my transcribed lesson, it may have grammatical errors; however, I need you to please list to me a list of as many topics as you may detect on the summary, find all of them and return them on this format [topic1, topic2, topic3, ...] and so on, Only answer with the list, nothing else. The title of the lesson is \"{title}\" and here is the content: {summarizedTranscript}"
        answer = askGpt(question, gpt4=True)
        topics = extract_topics_from_response(answer)
        
        lessons[-1].name = title
        lessons[-1].summary = summarizedTranscript
        lessons[-1].listOfSubjects = topics

    
    return lessons[-1]


def extract_topics_from_response(response):
    # Find the opening and closing square brackets
    start_index = response.find('[')
    end_index = response.find(']')

    # If the brackets are found, extract the text between them
    if start_index != -1 and end_index != -1:
        topics_str = response[start_index+1:end_index]
        topics_list = [topic.strip() for topic in topics_str.split(',')]
        return topics_list
    else:
        print("Error: Square brackets not found in the response")
        return []

def updateLessons(course):
    file_path = os.path.join(course.folder, course.main_note.name)
    if (checkIfNewLesson(course.main_note, file_path)):
        ###asyncio.run(sendBotMessage("The new Lessons are being saved!"))
        lesson = getNewestLesson(course)
        course.main_note.lessons.append(lesson)



def checkIfNewLesson(mainNote, filePath):
    
    totalPage = get_page_count(filePath)
    lessons = mainNote.lessons
    lastLessonNum = len(lessons) - 1
    if (lessons[lastLessonNum].lastPage != totalPage):
        return True
    else:
        return False

def checkUpRun(courses):#
    review_due_lessons(courses)

    

# async def sendBotMessage(message):
#     await bot.send_message(chat_id=user_id, text=message)


def list_upcoming_events():
    """Lists the next 35 events from the user's primary calendar."""
    creds = get_credentials()
    calendar_service = build('calendar', 'v3', credentials=creds)
    now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    events_result = calendar_service.events().list(calendarId='primary', timeMin=now,
                                                  maxResults=35, singleEvents=True,
                                                  orderBy='startTime').execute()
    events = events_result.get('items', [])

    if not events:
        print('No upcoming events found.')
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        print(f"{start}: {event['summary']}")

def create_event(start_time, end_time, summary, description, location=""):
    """Creates a new event in the user's primary calendar."""

    creds = get_credentials()
    calendar_service = build('calendar', 'v3', credentials=creds)

    event = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'America/Toronto',
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'America/Toronto',
        },
    }

    event = calendar_service.events().insert(calendarId='primary', body=event).execute()
    print(f'Event created: {event.get("htmlLink")}')

def create_eventOLD(start_time, end_time, summary, description, location=""):
    """Creates a new event in the user's primary calendar."""

    creds = get_credentials()
    calendar_service = build('calendar', 'v3', credentials=creds)

    event = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'America/Los_Angeles',
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'America/Los_Angeles',
        },
    }

    event = calendar_service.events().insert(calendarId='primary', body=event).execute()
    print(f'Event created: {event.get("htmlLink")}')
    ####asyncio.run(sendBotMessage(f'Event created for {summary}: {event.get("htmlLink")}'))

def schedule_nearest_event(event_title, event_description, event_location, preferred_date=None):
    if not preferred_date:
        preferred_date = datetime.date.today()
        
    weekday = preferred_date.weekday()

    # If the preferred date is a weekend, adjust to the following Monday
    if weekday >= 5:
        preferred_date += datetime.timedelta(days=(7 - weekday))

    # Check for available slots starting from the preferred date
    event_start = None
    for days_offset in itertools.chain(range(0, 7), range(-1, -8, -1)):
        current_date = preferred_date + datetime.timedelta(days=days_offset)
        weekday = current_date.weekday()

        # Skip weekends
        if weekday >= 5:
            continue

        # Check for available slots during the day
        for hour in range(11, 21):  # 11am to 9pm
            start_time = datetime.datetime.combine(current_date, datetime.time(hour, 0))
            end_time = start_time + datetime.timedelta(hours=1)

            if is_time_slot_available(start_time, end_time):
                event_start = start_time
                end_time = event_start + datetime.timedelta(hours=1)  # Add this line
                create_event(event_start, end_time, event_title, event_description, event_location)
                return  # Exit the function after scheduling the event


def schedule_nearest_eventOLD(event_title, event_description, event_location, preferred_date=None):
    if not preferred_date:
        preferred_date = datetime.date.today()
        
    weekday = preferred_date.weekday()

    # If the preferred date is a weekend, adjust to the following Monday
    if weekday >= 5:
        preferred_date += datetime.timedelta(days=(7 - weekday))

    # Check for available slots starting from the preferred date
    event_start = None
    for days_offset in itertools.chain(range(0, 7), range(-1, -8, -1)):
        current_date = preferred_date + datetime.timedelta(days=days_offset)
        weekday = current_date.weekday()

        # Skip weekends
        if weekday >= 5:
            continue

        # Check for available slots during the day
        for hour in range(11, 21):  # 11am to 9pm
            start_time = datetime.datetime.combine(current_date, datetime.time(hour, 0))
            end_time = start_time + datetime.timedelta(hours=1)

            if is_time_slot_available(start_time, end_time):
                event_start = start_time
                break

        if event_start:
            break

    if event_start:
        print("Tried")
        end_time = event_start + datetime.timedelta(hours=1)  # Add this line
        create_event(event_start, end_time, event_title, event_description, event_location)
    else:
        print("Could not find a suitable time slot for the event.")


# def schedule_nearest_event(event_title, event_description, event_location, preferred_date=None):
#     if not preferred_date:
#         preferred_date = datetime.date.today()
        
#     weekday = preferred_date.weekday()

#     # If the preferred date is a weekend, adjust to the following Monday
#     if weekday >= 5:
#         preferred_date += datetime.timedelta(days=(7 - weekday))

#     # Check for available slots starting from the preferred date
#     event_start = None
#     for days_offset in itertools.chain(range(0, 7), range(-1, -8, -1)):
#         current_date = preferred_date + datetime.timedelta(days=days_offset)
#         weekday = current_date.weekday()

#         # Skip weekends
#         if weekday >= 5:
#             continue

#         # Check for available slots during the day
#         for hour in range(11, 21):  # 11am to 9pm
#             start_time = datetime.datetime.combine(current_date, datetime.time(hour, 0))
#             end_time = start_time + datetime.timedelta(hours=1)

#             if is_time_slot_available(start_time, end_time):
#                 event_start = start_time
#                 break

#         if event_start:
#             break

#     if event_start:
#         create_event(event_title, event_description, event_location, event_start)
#     else:
#         print("Could not find a suitable time slot for the event.")
def is_time_slot_available(start_time, end_time):
    """Check if a time slot is available in the user's primary calendar."""

    creds = get_credentials()
    calendar_service = build('calendar', 'v3', credentials=creds)

    # Convert start_time and end_time to timezone-aware datetime objects
    start_time = start_time.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-4)))
    end_time = end_time.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-4)))

    events_result = calendar_service.events().list(calendarId='primary', timeMin=start_time.isoformat(), timeMax=end_time.isoformat(), singleEvents=True, orderBy='startTime').execute()
    events = events_result.get('items', [])

    if not events:
        return True
    else:
        return False



def is_time_slot_availableOLD(start_time, end_time):
    # Convert start_time and end_time to RFC3339 format
    start_time_rfc3339 = start_time.isoformat() + "-00:00"
    end_time_rfc3339 = end_time.isoformat() + "-00:00"

    # Call the Google Calendar API to retrieve the list of events in the specified time range
    service = build('calendar', 'v3', credentials=get_credentials())
    events_result = service.events().list(calendarId='primary', timeMin=start_time_rfc3339, timeMax=end_time_rfc3339, singleEvents=True, orderBy='startTime').execute()
    events = events_result.get('items', [])

    # If there are no events in the specified time range, the time slot is available
    return len(events) == 0

def schedule_due_dates(course):
    """Will look for lessons and if due dates fall, then it will schedule an event for that date"""
    lessons_due = {}

    # Collect due dates of all lessons
    for lesson in course.main_note.lessons:
        due_date = lesson.due_date
        if due_date in lessons_due:
            lessons_due[due_date].append(lesson)
        else:
            lessons_due[due_date] = [lesson]

    # Schedule events for days with 3 or more lessons due
    for due_date, lessons in lessons_due.items():
        if len(lessons) >= 3:
            event_title = f"{len(lessons)} lessons due"
            event_description = "We will go over: " + ", ".join([lesson.name for lesson in lessons])
            event_location = "Home"

            # Schedule the event
            # Schedule the event
            schedule_nearest_event(event_title, event_description, event_location, preferred_date=due_date)

def get_last_run_time():
    with open("last_run.txt", "r") as f:
        timestamp = f.read()
        if not timestamp:
            return None
        return datetime.datetime.fromisoformat(timestamp)

def update_last_run_time():
    with open("last_run.txt", "w") as f:
        f.write(datetime.datetime.now().isoformat())

def should_run_schedule_due_dates():
    last_run_time = get_last_run_time()
    if not last_run_time:
        return True

    time_since_last_run = datetime.datetime.now() - last_run_time
    if time_since_last_run >= datetime.timedelta(days=2):
        return True
    else:
        return False
    
def runPineConeProtocol():
    prompter1 = query_chunks("test")
    prompter2 = query_chunks("What are differential equations")
    prompter3 = query_chunks("What is mathematics")
    print("[Pinecone index updated]")

def get_page_content(url):
    if url is None:
        print("Error: Invalid URL (None)")
        return ""

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: Unable to fetch page content (status code {response.status_code})")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find the content div or the article tag (for alternative page structures)
    content = soup.find("div", {"id": "mw-content-text"}) or soup.find("article")

    if content:
        return content.text.strip()
    else:
        print("Error: Unable to find content element on the page.")
        return ""



def extractUrlFromResponse(response):
    url_pattern = r'https?://[^\s]+'
    match = re.search(url_pattern, response)
    return match.group(0) if match else None

# def is_valid_url(url):
#     try:
#         result = urlparse(url)
#         return all([result.scheme, result.netloc])
#     except ValueError:
#         return False

# def extractUrlFromResponse(response):
#     words = response.split()
#     for word in words:
#         if is_valid_url(word):
#             return word
#     return None

def search_proofwikiOLD(query, max_results=10):
    # Replace spaces with '+' in the query
    query = query.replace(' ', '+')
    
    # Generate the search URL
    url = f"https://proofwiki.org/w/index.php?search={query}"
    
    # Fetch the search results page
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: Unable to fetch search results (status code {response.status_code})")
        return []
    
    # Check for exact result (URL starts with /wiki/)
    parsed_url = urlparse(response.url)
    if parsed_url.path.startswith('/wiki/'):
        return [response.url]

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract the search results
    search_results = soup.find_all("li", class_="mw-search-result")
    
    # Extract the first 10 results' links
    links = []
    for result in search_results[:max_results]:
        link = result.find("a")
        links.append("https://proofwiki.org" + link["href"])
    
    return links

def smartSearchProofWiki(query, max_results=10, SUPERQuery=None):
    """USES GPT4"""
    # Replace spaces with '+' in the query
    if query == None:
        
        query = askGpt(f"given this question\"{SUPERQuery}\" what is a term I can do a query search for that could answer the question, do not answer with anything else other than the term, please only answer with \"[term here]\", the term sorrounded by square brackets", gpt4= True)
    query = query.replace(' ', '+')
    
    # Generate the search URL
    url = f"https://proofwiki.org/w/index.php?search={query}"
    
    # Fetch the search results page
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: Unable to fetch search results (status code {response.status_code})")
        return []
    
    # Check for exact result (URL starts with /wiki/)
    parsed_url = urlparse(response.url)
    if parsed_url.path.startswith('/wiki/'):
        return get_page_content(response.url)
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract the search results
    search_results = soup.find_all("li", class_="mw-search-result")
    
    # Extract the first 10 results' links
    links = []
    for result in search_results[:max_results]:
        link = result.find("a")
        links.append("https://proofwiki.org" + link["href"])

    if SUPERQuery:
        
        gptJudgement = askGpt(f"given this query\"{SUPERQuery}\" what would the link that is most likely to have an answer to it be within this list \"{links}\", return only the link", gpt4=True)
        return get_page_content(extractUrlFromResponse(gptJudgement))
    else:
        
        gptJudgement = askGpt(f"given this query \"{query}\", what would the link that is most likely to have an answer to it be within this list \"{links}\", return only the link", gpt4=True)
        linkk = extractUrlFromResponse(gptJudgement)
        return get_page_content(linkk)
    
def search_proofwiki(query, max_results=10, returnList=True):
    # Replace spaces with '+' in the query
    query = query.replace(' ', '+')
    
    # Generate the search URL
    url = f"https://proofwiki.org/w/index.php?search={query}"
    
    # Fetch the search results page
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: Unable to fetch search results (status code {response.status_code})")
        return [] if returnList else ""
    
    # Check for exact result (URL starts with /wiki/)
    parsed_url = urlparse(response.url)
    if parsed_url.path.startswith('/wiki/'):
        return [response.url] if returnList else get_page_content(response.url)

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract the search results
    search_results = soup.find_all("li", class_="mw-search-result")
    
    # Extract the first 10 results' links
    links = []
    for result in search_results[:max_results]:
        link = result.find("a")
        links.append("https://proofwiki.org" + link["href"])
    
    return links if returnList else get_page_content(links[0])


def main():
    """The main function that runs the entire script, handling Google Drive operations and interacting with Pinecone and GPT."""


    # print("tes test start")
    # pdfLem = "C:/Users/aleja/OneDrive/Documentos/Second Year/MATH 235/Assg 4/assg 4 file.pdf"
    # print(extract_text_from_page(pdfLem, 2))

    # print("tes test end")

###################google

    creds = get_credentials()
    service = build('drive', 'v3', credentials=creds)
    
    
    # summary = 'Nearest Event'
    # description = 'This is a sample event scheduled on the nearest available time slot.'
    # location = '123 Main St, Anytown, USA'

    # schedule_nearest_event(summary, description, location)
    
    driveProtocol2()
    timer_thread.start()
    run_threaded_read_homework_file()
    ##rememeber courses
    courses = load_courses(DATA_FILE)

    # If courses is empty, create courses and set main notes (only on the first run)
    if not courses:
        print("INNNNNNNNNNN")
        courses = initializeCourses()
        # Save courses to file
        save_courses(courses, DATA_FILE)
    print("courses have been loaded.")

    for course in courses:
        file_path = os.path.join(course.folder, course.main_note.name)
        newLesson =checkIfNewLesson(course.main_note, file_path)
        updateLessons(course)
        process_transcripts(service, course)
        if newLesson:
            course.main_note.lessons[-1] = updateNewestWithTranscript(course)

    if should_run_schedule_due_dates():
        for course in courses:
            schedule_due_dates(course)
            runPineConeProtocol()
            update_last_run_time()
    
###################google
    print("what would you like to do?")
    print("Load a pdf (1), Query DB (2), Idle Run (3), Run checkUp (4) Close (9)")
    print()
    loader = input("Option: ")

    if(loader == "1"):
        
        #pdf_path = "C:/Users/aleja/OneDrive/Documentos/Second Year/AMATH 251/AMATH_251_Course_Notes_September_20_2021.pdf"
        pdf_path = input("copy paste path here: ")
        pdf_chunks = read_pdf_and_chunk(pdf_path, 200)
        
        # Store the chunks in Pinecone
        store_chunks(pdf_chunks)
    elif(loader == "2"):
        print("Query mode")
        query = input("Input Query: ")
        relevant_chunk_scores = query_chunks(query)
        relevant_chunks = " ".join([chunk['id'] for chunk in relevant_chunk_scores])



        #print("relevant: "+ relevant_chunks)
        #print("done")
        response = askGptContext(query, relevant_chunks, False)

        print(response)

        wolframQuestion = input("Would you like to verify this information with Wolfram Alpha? (yes/no): ")

        if wolframQuestion == "yes":
            processQuestion = askGpt(f"Look at this text \"{response}\" we need to verify its contents with Wolfram Alpha, based on this text give me a list of questions that I can ask Wolfram Alpha API to verify the information, give them to me on this format [question1, question2, question3]", gpt4=True)
            listOfQuestions = extract_topics_from_response(processQuestion)

            # Initialize a list to store Wolfram Alpha answers
            wolfram_answers = []

            # Iterate through the list of questions and ask Wolfram Alpha API
            for question in listOfQuestions:
                try:
                    res = wolframClient.query(question)
                    answer = next(res.results).text
                    wolfram_answers.append(answer)
                    print(f"Question: {question}\nAnswer: {answer}\n")
                except (StopIteration, AttributeError):
                    print(f"Question: {question}\nAnswer: No answer found\n")

            # Convert the list of Wolfram Alpha answers to a string
            wolfram_answers_str = " ".join(wolfram_answers)

            # Ask GPT if the text is accurate based on Wolfram Alpha answers
            gpt_question = f"Given this information from Wolfram Alpha \"{wolfram_answers_str}\", is what this text says accurate \"{response}\"?"
            gpt_answer = askGpt(gpt_question, gpt4=True)
            
            print("GPT's evaluation:", gpt_answer)
        else:
            print("Closing...")
    elif(loader == "3"):
        count = 0
        try:
            waiting_wheel()
        except KeyboardInterrupt:
            sys.stdout.write('\rDone! ')
            sys.stdout.flush()
    elif(loader == "4"):
        checkUpRun(courses)
    else:
        # for coursee in courses:
        #     print(coursee.main_note.lessons[-1].topics)
        print("Closing...")
        time.sleep(1)


    save_courses(courses, DATA_FILE)
    

if __name__ == "__main__":
    main()



    
    
