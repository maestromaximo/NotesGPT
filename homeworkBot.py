
import datetime
from datetime import datetime, timedelta
import logging
import time
from dotenv import load_dotenv
import os
import tiktoken
import openai
import jsonpickle
import re
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
CLIENT_SECRET_FILE = os.getenv("CLIENT_SECRET_FILE")
#SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
#SCOPES = ['https://www.googleapis.com/auth/drive']
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_credentials():
    """Obtains Google API credentials from the 'token.pickle' file or a new ~token if necessary."""
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

    def __str__(self):
        return "\n".join([str(content) for content in self.contents])

    def fill_end_pages(self):
        for i, content in enumerate(self.contents):
            if i + 1 < len(self.contents):
                next_content = self.contents[i + 1]
                if content.startPage == next_content.startPage:
                    content.endPage = content.startPage
                else:
                    content.endPage = next_content.startPage - 1
            else:
                content.endPage = content.startPage


class Content:
    def __init__(self, chapter, name, startPage, endPage=None):
        self.chapter = chapter
        self.name = name
        self.startPage = startPage
        self.endPage = endPage

    def __str__(self):
        if self.endPage:
            return f"{self.chapter} {self.name} (pp. {self.startPage}-{self.endPage})"
        else:
            return f"{self.chapter} {self.name} (p. {self.startPage})"


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

def load_courses(filename):
    if not os.path.exists(filename):
        return []

    with open(filename, 'r') as infile:
        json_data = infile.read()
        courses = jsonpickle.decode(json_data)
    return courses

DATA_FILE = "courses_data.json"
courses = load_courses(DATA_FILE)


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def botStart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def botHelpCommand(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Hi rememeber this are the course codes! \"QM1, THR, MAS, EM1, CA4\" and the \n order is \"Course_code, description (title), dificulty from 1-5, due date on days from now, number of questions\"")


async def deleteData(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    
    await update.message.reply_text("DataBase deleted")
    with open("telegram_messages.txt", "w") as file:
        file.write("")

async def prepare_for_today_lessons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Get today's date
    today = datetime.today()
    print("recieved today")
    # DATA_FILE = "courses_data.json"
    # courses = load_courses(DATA_FILE)
     # Get today's date and time
    now = datetime.now()

    # Get the start of today (midnight)
    start_of_today = datetime(now.year, now.month, now.day)

    # Convert start_of_today to an RFC3339 timestamp
    start_time = start_of_today.isoformat() + "Z"

    # Get the end of today (23:59:59)
    end_of_today = start_of_today + timedelta(days=1, seconds=-1)

    # Convert end_of_today to an RFC3339 timestamp
    end_time = end_of_today.isoformat() + "Z"

    creds = get_credentials()
    calendar_service = build('calendar', 'v3', credentials=creds)
    # Get all events from today's calendar
    events_result = calendar_service.events().list(calendarId='primary', timeMin=start_time, timeMax=end_time, singleEvents=True, orderBy='startTime').execute()
    events = events_result.get('items', [])

    # Loop over all events
    for event in events:
        event_name = event['summary']

        # Check if any of the course codes is in the event name
        for course in courses:
            if course.code in event_name and "LEC" in event_name:
                print("in")
                # Get the chapter from last lesson
                last_chapter = get_lesson_chapter([course], True)
                last_chapter = last_chapter.strip("[]").strip()  # Remove brackets

                # Find the next chapter in the Table of Contents
                found_last_chapter = False
                next_chapter = None
                for content in course.main_note.tableOfContents.contents:
                    if found_last_chapter:
                        next_chapter = content.name
                        break
                    if content.name == last_chapter or content.chapter == last_chapter:
                        found_last_chapter = True
                
                if next_chapter:
                    # Ask GPT-4 for key pointers for the next chapter
                    prompt = f'I am going to my {course.name} class, we are going to learn about {next_chapter} today, give me a few key pointers that will help me grasp everything better before I go into the lesson, include math and one example with solution if possible'
                    response = askGpt(prompt, False)
                    await update.message.reply_text(response)
                    #print(response)
    print("out")


async def botEcho(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    message_text = update.message.text
    await update.message.reply_text("Got your request! and its been sent to procesing")
    print(message_text)

    # Save the message with the current date and time to a text file
    
    with open("telegram_messages.txt", "a") as file:
        file.write(f"{message_text}\n")
def askGpt(prompt, gpt4):
    conversation = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    
    # Calculate available tokens for the response
    prompt_tokens = numTokensFromString(prompt)
    max_allowed_tokens = 4000  # Set the maximum allowed tokens
    available_tokens_for_response = max_allowed_tokens - prompt_tokens

    # Ensure the available tokens for the response is within the model's limit
    if available_tokens_for_response < 1:
        raise ValueError("The input query is too long. Please reduce the length of the input query.")
    
    max_retries = 4
    for _ in range(max_retries + 1):  # This will try a total of 5 times (including the initial attempt)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4" if gpt4 else "gpt-3.5-turbo",
                messages=conversation,
                max_tokens=available_tokens_for_response,
                n=1,
                stop=None,
                temperature=0.1,
            )

            message = response.choices[0].message["content"].strip()

            # Count tokens
            response_tokens = numTokensFromString(message)
            total_tokens = prompt_tokens + response_tokens

            # Calculate cost
            cost_per_token = 0.06 if gpt4 else 0.002
            cost = (total_tokens / 1000) * cost_per_token

            # Update the cost file
            updateCostFile(cost)

            return message
        
        except Exception as e:
            if _ < max_retries:
                print(f"Error occurred: {e}. Retrying {_ + 1}/{max_retries}...")
                time.sleep(1)  # You can adjust the sleep time as needed
            else:
                raise
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


def numTokensFromString(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(str(string)))
    return num_tokens
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
def get_lesson_chapter(courses, gpt4 = False):

    for course in courses:
        main_note = course.main_note
        if main_note and main_note.lessons:
            last_lesson = main_note.lessons[-1]
            lesson_name = last_lesson.name
            topics = ', '.join(last_lesson.listOfSubjects)
            
            # Create a string representation of the chapters
            chapters_str = "\n".join([f"{content.chapter}: {content.name}" for content in main_note.tableOfContents.contents])
            
            prompt = f'I have a lesson titled "{lesson_name}" on my notes, and it goes through these topics "{topics}" given this information, which chapter do you think this lesson is from given this table of contents:\n{chapters_str}\nOnly answer with the chapter and surround the chapter with "[ ]" you must do it this way.'
            
            answer = askGpt(prompt, False)
            finalAnswer = extract_topics_from_response(answer)
            print(finalAnswer[0])
            return finalAnswer[0]

def load_table_of_contents(contents_str):
    contents = []
    lines = contents_str.split("\n")
    
    for line in lines:
        match = re.match(r"(\d+(?:\.\d+)?(?:\.\d+)?)(.*?)\.*(\d+)", line)
        
        if match:
            chapter = match.group(1)
            name = match.group(2).strip()
            startPage = int(match.group(3))
            contents.append(Content(chapter, name, startPage))
            
    return TableOfContents(contents)

# Example table of contents
# CalcTableString = '''1 Curves & Vector Fields 1
# 1.1 Curves in Rn. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
# 1.1.1 Curves as vector-valued functions . . . . . . . . . . . . . . . . . . . . 1
# 1.1.2 Reparametrization of a curve . . . . . . . . . . . . . . . . . . . . . . 7
# 1.1.3 Limits . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
# 1.1.4 Derivatives . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
# 1.1.5 Arclength . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
# 1.2 Vector fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
# 1.2.1 Examples from physics . . . . . . . . . . . . . . . . . . . . . . . . . . 17
# 1.2.2 Field lines of a vector field . . . . . . . . . . . . . . . . . . . . . . . . 21
# 2 Line Integrals & Green’s Theorem 27
# 2.1 Line integral of a scalar field . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
# 2.1.1 Motivation and definition . . . . . . . . . . . . . . . . . . . . . . . . 27
# 2.1.2 Applications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
# 2.2 Line integral of a vector field . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
# 2.2.1 Motivation and definition . . . . . . . . . . . . . . . . . . . . . . . . 34
# 2.2.2 Applications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
# 2.2.3 Some technical matters . . . . . . . . . . . . . . . . . . . . . . . . . . 38
# 2.3 Path-independent line integrals . . . . . . . . . . . . . . . . . . . . . . . . . 41
# 2.3.1 First Fundamental Theorem for Line Integrals . . . . . . . . . . . . . 42
# 2.3.2 Second Fundamental Theorem for Line Integrals . . . . . . . . . . . . 45
# 2.3.3 Conservative (i.e. gradient) vector fields . . . . . . . . . . . . . . . . 46
# 2.4 Green’s Theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
# 2.4.1 The theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
# 2.4.2 Existence of a potential in R2. . . . . . . . . . . . . . . . . . . . . . 53
# 2.5 Vorticity and circulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
# 3 Surfaces & Surface Integrals 65
# 3.1 Parametrized surfaces . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
# 3.1.1 Surfaces as vector-valued functions . . . . . . . . . . . . . . . . . . . 65
# 3.1.2 The tangent plane . . . . . . . . . . . . . . . . . . . . . . . . . . . . 69
# 3.1.3 Surface area . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71
# 3.1.4 Orientation of a surface . . . . . . . . . . . . . . . . . . . . . . . . . 73
# 3.2 Surface Integrals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
# 3.2.1 Scalar fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
# 3.2.2 Vector fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
# 3.2.3 Properties of surface integrals . . . . . . . . . . . . . . . . . . . . . . 79
# 4 Gauss’ and Stokes’ Theorems 81
# 4.1 The vector differential operator ∇ . . . . . . . . . . . . . . . . . . . . . . . . 81
# 4.1.1 Divergence and curl of a vector field . . . . . . . . . . . . . . . . . . . 81
# 4.1.2 Identities involving ∇ . . . . . . . . . . . . . . . . . . . . . . . . . . . 83
# 4.1.3 Expressing ∇ in curvilinear coordinates . . . . . . . . . . . . . . . . . 86
# 4.2 Gauss’ Theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 93
# 4.2.1 The theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 93
# 4.2.2 Conservation laws and PDEs . . . . . . . . . . . . . . . . . . . . . . . 98
# 4.2.3 The Generalized Divergence Theorem . . . . . . . . . . . . . . . . . . 101
# 4.3 Stokes’ Theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 102
# 4.3.1 The theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 102
# 4.3.2 Faraday’s law . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 107
# 4.3.3 The physical interpretation of ∇ × F . . . . . . . . . . . . . . . . . . 109
# 4.4 The Potential Theorems . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 109
# 4.4.1 Irrotational vector fields . . . . . . . . . . . . . . . . . . . . . . . . . 110
# 4.4.2 Divergence-free vector fields . . . . . . . . . . . . . . . . . . . . . . . 112
# 5 Fourier Series and Fourier Transforms 117
# 5.1 Fourier Series . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 117
# 5.1.1 Calculating Fourier coefficients . . . . . . . . . . . . . . . . . . . . . 119
# 5.1.2 Pointwise convergence of a Fourier series . . . . . . . . . . . . . . . . 123
# 5.1.3 Symmetry properties . . . . . . . . . . . . . . . . . . . . . . . . . . . 128
# 5.1.4 Complex form of the Fourier series . . . . . . . . . . . . . . . . . . . 133
# 5.2 Convergence of series of functions . . . . . . . . . . . . . . . . . . . . . . . . 138
# 5.2.1 A deficiency of pointwise convergence . . . . . . . . . . . . . . . . . . 139
# 5.2.2 The maximum norm and mean square norm . . . . . . . . . . . . . . 141
# 5.2.3 Uniform and Mean Square Convergence . . . . . . . . . . . . . . . . . 145
# 5.2.4 Termwise integration of series . . . . . . . . . . . . . . . . . . . . . . 151
# 5.3 A Second Look at Fourier Series . . . . . . . . . . . . . . . . . . . . . . . . . 153
# 5.3.1 Uniform and mean square convergence . . . . . . . . . . . . . . . . . 153
# 5.3.2 Integration of Fourier series . . . . . . . . . . . . . . . . . . . . . . . 155
# 5.4 The Fourier transform & Fourier integral . . . . . . . . . . . . . . . . . . . . 157
# 5.4.1 The definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 157
# 5.4.2 Calculating Fourier transforms using the definition . . . . . . . . . . 160
# 5.4.3 Properties of the Fourier transform . . . . . . . . . . . . . . . . . . . 163
# 5.4.4 Parseval’s formula for a non-periodic function . . . . . . . . . . . . . 167
# 5.4.5 Relation between the continuous spectrum and the discrete spectrum 168
# 5.4.6 Things are simpler in the frequency domain . . . . . . . . . . . . . . 170
# 5.5 Digitized signals – a glimpse . . . . . . . . . . . . . . . . . . . . . . . . . . . 172
# 5.5.1 The sampling theorem . . . . . . . . . . . . . . . . . . . . . . . . . . 172
# 5.5.2 The discrete Fourier transform (DFT) . . . . . . . . . . . . . . . . . 174'''
# electroTableString = """1 Electrostatic field in vacuum 7
# 1.1 Electric Interactions: Basic Examples . . . . . . . . . . . . . . . . . . . 7
# 1.1.1 Insulators and Conductors . . . . . . . . . . . . . . . . . . . . . 7
# 1.2 Electric Pendulum . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
# 1.3 Electric Charge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
# 1.3.1 Summation of Electric Charges . . . . . . . . . . . . . . . . . . 9
# 1.4 Coulomb’s Law . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
# 1.5 Superposition Principle . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
# 1.5.1 Example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
# 1.6 Charge Conservation Principle . . . . . . . . . . . . . . . . . . . . . . . 14
# 1.7 Electrostatic Field . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
# 1.8 Continuous charge distributions . . . . . . . . . . . . . . . . . . . . . . 18
# 1.8.1 Physical meaning of a continuous charge distribution . . . . . . 19
# 1.9 Electric Field Generated by a Generic Charge Distribution . . . . . . . 23
# 1.10 Gauss’ Theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
# 1.11 Line Integral of a Vector Field on an Oriented Curve . . . . . . . . . . 31
# 1.12 Irrotational Property of the Electrostatic Field . . . . . . . . . . . . . . 35
# 1.13 Symmetry Arguments = Irrotational Property . . . . . . . . . . . . . . 37
# 1.14 Electrostatic Field Properties in Local Form . . . . . . . . . . . . . . . 50
# 1.14.1 Gauss’ Theorem in Local Form . . . . . . . . . . . . . . . . . . 50
# 1.15 The Irrotational Property in Local Form. . . . . . . . . . . . . . . . . . 53
# 1.16 Gauss’ theorem and the irrotational property of E~ . . . . . . . . . . . . 54
# 1.17 The Divergence Theorem in Detail . . . . . . . . . . . . . . . . . . . . 61
# 1.18 Electrostatic Field E~ Generated by Point-Like Charges . . . . . . . . . 65
# 1.19 Physics History Inverted: Coulomb’s Theorem and the Superposition Property of E~ Derived from Gauss’ Law and the Irrotational Principle of E~ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
# 1.19.1 Laws vs. Theorems in Physics. . . . . . . . . . . . . . . . . . . 74
# 1.19.2 Gauss and Faraday. . . . . . . . . . . . . . . . . . . . . . . . . . 75
# 1.19.3 Coulomb. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79
# 1.19.4 Back to history. . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
# 2 The Electrostatic Potential 89
# 2.1 The electrostatic potential. . . . . . . . . . . . . . . . . . . . . . . . . . 89
# 2.1.1 The potential of a point-like charge. . . . . . . . . . . . . . . . . 89
# 2.1.2 Work of the Field Forces. . . . . . . . . . . . . . . . . . . . . . . 90
# 2.2 Potential of a Generic Charge Distribution. . . . . . . . . . . . . . . . . 91
# 2.3 The Electrostatic Potential of Charged Rings, Shells, and Spheres. . . . 95
# 2.3.1 The Electrostatic Potential of a Charged Ring on the Ring’s Axis. 95
# 2.3.2 The Electrostatic Potential of a Charged Shell at any Point P in Space. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 96
# 2.3.3 The Electrostatic Potential of a Charged Sphere with Known Field E~ at any Point in Space. . . . . . . . . . . . . . . . . . . . 99
# 2.4 Poisson and Laplace Equations. . . . . . . . . . . . . . . . . . . . . . . 103
# 2.5 Energy of a System of Point-Like Charges. . . . . . . . . . . . . . . . . 105
# 2.6 Energy density of an electrostatic field. . . . . . . . . . . . . . . . . . . 110
# 2.7 Electrostatic Potential of a Double Infinite Layer. . . . . . . . . . . . . 112
# 2.8 The Potential Generated by Unlimited Charge Distributions. . . . . . . 115
# 2.8.1 The Potential of a Single Infinite Layer. . . . . . . . . . . . . . 115
# 2.8.2 The Potential of a Charged Disk on the Disk’s Axis. . . . . . . . 117
# 2.8.3 The Potential of a Single Infinite Layer Revisited. . . . . . . . . 119
# 3 The Electrostatic Field in Presence of Conductors in Vacuum. 123
# 3.1 Conductors and Insulators. . . . . . . . . . . . . . . . . . . . . . . . . . 123
# 3.2 Electrostatic Equilibrium in Homogeneous Conductors. . . . . . . . . . 125
# 3.3 Field Calculation in Presence of Conductors. . . . . . . . . . . . . . . . 131
# 3.3.1 Conducting Sphere. . . . . . . . . . . . . . . . . . . . . . . . . . 132
# 3.4 The Field in Hollow Conductors: Electrostatic Shields. . . . . . . . . . 134
# 3.5 Capacitance. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 141
# 3.5.1 System of Two Conductors: Capacitors. . . . . . . . . . . . . . 142
# 3.5.2 Parallel Plate Capacitor. . . . . . . . . . . . . . . . . . . . . . . 148
# 3.5.3 Electrostatic Energy of a System of Charged Conductors. . . . . 154
# 4 Electric Current 157
# 4.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 157
# 4.2 Interactions between Moving Charges . . . . . . . . . . . . . . . . . . . 158
# 4.3 The Electric Current . . . . . . . . . . . . . . . . . . . . . . . . . . . . 161
# 4.4 Different Types of Current . . . . . . . . . . . . . . . . . . . . . . . . . 163
# 4.5 Current Density and Continuity Equation . . . . . . . . . . . . . . . . 164
# 4.6 Stationary Current . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 167
# 4.7 The Electric Field in Conductors with a Stationary Current . . . . . . 169
# 4.8 The Electromotive Force Acting in a Circuit with Current . . . . . . . 172
# 4.9 Ohm’s and Joule’s Law in Integral Form . . . . . . . . . . . . . . . . . 178
# 4.10 Resistance of Conductors with Current . . . . . . . . . . . . . . . . . . 182
# 4.10.1 Cylindrical Conductor with Longitudinal Current . . . . . . . . 182
# 4.10.2 Hollow Cylindrical Conductor with Radial Current . . . . . . . 183
# 5 The Magnetostatic Field in Vacuum 187
# 5.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 187
# 5.2 The Fundamental Laws of Magnetostatic . . . . . . . . . . . . . . . . . 188
# 5.3 Magnetic Field Generated by Simple Current Distribution . . . . . . . 193
# 5.3.1 Infinite Straight Line with Stationary Current . . . . . . . . . . 193
# 5.4 The Laws of Magnetostatics in Local Form . . . . . . . . . . . . . . . . 199
# 5.5 Fields with Vector Potential . . . . . . . . . . . . . . . . . . . . . . . . 207
# 5.6 Vector Potential of B~ and Laplace’s Elementary Theorem . . . . . . . . 208
# 5.7 Coefficients of Self and Mutual Inductance of Circuits . . . . . . . . . . 212
# 5.8 Selected examples on self and mutual inductance . . . . . . . . . . . . . 221
# 5.8.1 Toroidal Solenoid . . . . . . . . . . . . . . . . . . . . . . . . . . 221
# 5.8.2 Infinite Straight Solenoid . . . . . . . . . . . . . . . . . . . . . . 232"""
# thermoTableString = """1.1 Thermal Equilibrium 1
# 1.2 The Ideal Gas 6
# 1.3 Equipartition Energy 14
# 1.4 Heat and Work 17
# 1.5 Compression Work 20
# 1.6 Heat Capacities 28
# 1.7 Rates of Processes 37
# 2 The Second Law 49
# 2.1 Two State System 49
# 2.2 The Einstein Model of a Solid 53
# 2.3 Interacting Systems 56
# 2.4 Large Systems 60
# 2.5 The Ideal Gas with the Second Law 68
# 2.6 Entropy 74
# 3 Interactions and Implications 85
# 3.1 Temperature 85
# 3.2 Entropy and Heat 92
# 3.3 Paramagnetism 98
# 3.4 Mechanical Equilibrium and Pressure 108
# 3.5 Diffusive Equilibrium and Chemical Potential 115
# 3.6 Summary and a Look Ahead 120
# 4 Engines and Refrigerators 122
# 4.1 Heat Engines 122
# 4.2 Refrigerators 127
# 4.3 Real Heat Engines 131
# 4.4 Real Refrigerators 137
# 5 Free Energy and Chemical Thermodynamics 149
# 5.1 Free Energy as Available Work 149
# 5.2 Free Energy as a Force toward Equilibrium 161
# 5.3 Phase Transformations of Pure Substances 166
# 5.4 Phase Transformations of Mixtures 186
# 5.5 Dilute Solutions 200
# 5.6 Chemical Equilibrium 208
# 6 Boltzman Statistics 220
# 6.1 The Boltzmann Factor 220
# 6.2 Average Values 229
# 6.3 The Equipartition Theorem 238
# 6.4 The Maxwell Speed Distribution 242
# 6.5 Partition Functions and Free Energy 247
# 6.6 Partition Functions for Composite Systems
# 6.7 Ideal Gas Revisited 251
# 7 Quantum Statistics 257
# 7.1 The Gibbs Factor 257
# 7.2 Bosons and Fermions 262
# 7.3 Degenerate Fermi Gases 271
# 7.4 Blackbody Radiation 288
# 7.5 Debye Theory of Solids
# 7.6 Bose Einstein Condensation 315
# 8 Systems of Interacting Particle 327
# 8.1 Weakly Intercating Gases 328
# 8.2 The Ising Model of a Ferromagnet 339
# """
# quantumTableString = """1 The Wave Function 3
# 1.1 The Schrödinger Equation 3
# 1.2 The Statistical Interpretation 3
# 1.3 Probability 8
# 1.3.1 Discrete Variables 8
# 1.3.2 Continuous Variables 11
# 1.4 Normalization 14
# 1.5 Momentum 16
# 1.6 The Uncertainty Principle 19
# Further Problems on Chapter 1 20
# 2 Time-Independent Schrödinger Equation 25
# 2.1 Stationary States 25
# 2.2 The Infinite Square Well 31
# 2.3 The Harmonic Oscillator 39
# 2.3.1 Algebraic Method 40
# 2.3.2 Analytic Method 48
# 2.4 The Free Particle 55
# 2.5 The Delta-Function Potential 61
# 2.5.1 Bound States and Scattering States 61
# 2.5.2 The Delta-Function Well 63
# 2.6 The Finite Square Well 70
# 3 Formalism 91
# 3.1 Hilbert Space 91
# 3.2 Observables 94
# 3.2.1 Hermitian Operators 94
# 3.2.2 Determinate States 96
# 3.3 Eigenfunctions of a Hermitian Operator 97
# 3.3.1 Discrete Spectra 98
# 3.3.2 Continuous Spectra 99
# 3.4 Generalized Statistical Interpretation 102
# 3.5 The Uncertainty Principle 105
# 3.5.1 Proof of the Generalized Uncertainty Principle 105
# 3.5.2 The Minimum-Uncertainty Wave Packet 108
# 3.5.3 The Energy-Time Uncertainty Principle 109
# 3.6 Vectors and Operators 113
# 3.6.1 Bases in Hilbert Space 113
# 3.6.2 Dirac Notation 117
# 3.6.3 Changing Bases in Dirac Notation 121
# 4 Quantum Mechanics in Three Dimensions 131
# 4.1 The Schrödinger Equation 131
# 4.1.1 Spherical Coordinates 132
# 4.1.2 The Angular Equation 134
# 4.1.3 The Radial Equation 138
# 4.2 The Hydrogen Atom 143
# 4.2.1 The Radial Wave Function 144
# 4.2.2 The Spectrum of Hydrogen 155
# 4.3 Angular Momentum 157
# 4.3.1 Eigenvalues 157
# 4.3.2 Eigenfunctions 162
# 4.4 Spin 165
# 4.4.1 Spin 1/2 167
# 4.4.2 Electron in a Magnetic Field 172
# 4.4.3 Addition of Angular Momenta 176
# 4.5 Electromagnetic Interactions 181
# 4.5.1 Minimal Coupling 181
# 4.5.2 The Aharonov–Bohm Effect 182
# 5 Identical Particles 198
# 5.1 Two-Particle Systems 198
# 5.1.1 Bosons and Fermions 201
# 5.1.2 Exchange Forces 203
# 5.1.3 Spin 206
# 5.1.4 Generalized Symmetrization Principle 207
# 5.2 Atoms 209
# 5.2.1 Helium 210
# 5.2.2 The Periodic Table 213
# 5.3 Solids 216
# 5.3.1 The Free Electron Gas 216
# 5.3.2 Band Structure 220
# 6 Symmetries & Conservation Laws 232
# 6.1 Introduction 232
# 6.1.1 Transformations in Space 232
# 6.2 The Translation Operator 235
# 6.2.1 How Operators Transform 235
# 6.2.2 Translational Symmetry 238
# 6.3 Conservation Laws 242
# 6.4 Parity 243
# 6.4.1 Parity in One Dimension 243
# 6.4.2 Parity in Three Dimensions 244
# 6.4.3 Parity Selection Rules 246
# 6.5 Rotational Symmetry 248
# 6.5.1 Rotations About the z Axis 248
# 6.5.2 Rotations in Three Dimensions 249
# 6.6 Degeneracy 252
# 6.7 Rotational Selection Rules 255
# 6.7.1 Selection Rules for Scalar Operators 255
# 6.7.2 Selection Rules for Vector Operators 258
# 6.8 Translations in Time 262
# 6.8.1 The Heisenberg Picture 264
# 6.8.2 Time-Translation Invariance 266
# 7 Time-Independent Perturbation Theory 279
# 7.1 Nondegenerate Perturbation Theory 279
# 7.1.1 General Formulation 279
# 7.1.2 First-Order Theory 280
# 7.1.3 Second-Order Energies 284
# 7.2 Degenerate Perturbation Theory 286
# 7.2.1 Two-Fold Degeneracy 286
# 7.2.2 “Good” States 291
# 7.2.3 Higher-Order Degeneracy 294
# 7.3 The Fine Structure of Hydrogen 295
# 7.3.1 The Relativistic Correction 296
# 7.3.2 Spin-Orbit Coupling 299
# 7.4 The Zeeman Effect 304
# 7.4.1 Weak-Field Zeeman Effect 305
# 7.4.2 Strong-Field Zeeman Effect 307
# 7.4.3 Intermediate-Field Zeeman Effect 309
# 7.5 Hyperfine Splitting in Hydrogen 311
# 8 The Variational Principle 327
# 8.1 Theory 327
# 8.2 The Ground State of Helium 332
# 8.3 The Hydrogen Molecule Ion 337
# 8.4 The Hydrogen Molecule 341
# 9 The WKB Approximation 354
# 9.1 The “Classical” Region 354
# 9.2 Tunneling 358
# 9.3 The Connection Formulas 362
# 10 Scattering 376
# 10.1 Introduction 376
# 10.1.1 Classical Scattering Theory 376
# 10.1.2 Quantum Scattering Theory 379
# 10.2 Partial Wave Analysis 380
# 10.2.1 Formalism 380
# 10.2.2 Strategy 383
# 10.3 Phase Shifts 385
# 10.4 The Born Approximation 388
# 10.4.1 Integral Form of the Schrödinger Equation 388
# 10.4.2 The First Born Approximation 391
# 10.4.3 The Born Series 395
# 11 Quantum Dynamics 402
# 11.1 Two-Level Systems 403
# 11.1.1 The Perturbed System 403
# 11.1.2 Time-Dependent Perturbation Theory 405
# 11.1.3 Sinusoidal Perturbations 408
# 11.2 Emission and Absorption of Radiation 411
# 11.2.1 Electromagnetic Waves 411
# 11.2.2 Absorption, Stimulated Emission, and Spontaneous Emission 412
# 11.2.3 Incoherent Perturbations 413
# 11.3 Spontaneous Emission 416
# 11.3.1 Einstein’s A and B Coefficients 416
# 11.3.2 The Lifetime of an Excited State 418
# 11.3.3 Selection Rules 420
# 11.4 Fermi’s Golden Rule 422
# 11.5 The Adiabatic Approximation 426
# 11.5.1 Adiabatic Processes 426
# 11.5.2 The Adiabatic Theorem 428
# 12 Afterword 446
# 12.1 The EPR Paradox 447
# 12.2 Bell’s Theorem 449
# 12.3 Mixed States and the Density Matrix 455
# 12.3.1 Pure States 455
# 12.3.2 Mixed States 456
# 12.3.3 Subsystems 458
# 12.4 The No-Clone Theorem 459
# 12.5 Schrödinger’s Cat 461
# """
# masTableString = """1.1 Matrices, Vectors, and Vector CalculusIntroduction 1 
# 1.2 Concept of a Scalar 2 
# 1.3 Coordinate Transformations 3 
# 1.4 Properties of Rotation Matrices 6 
# 1.5 Matrix Operations 9 
# 1.6 Further Definitions 12 
# 1.7 Geometrical Significance of Transformation Matrices 14 
# 1.8 Definitions of a Scalar and a Vector in Terms of Transformation Properties 20 
# 1.9 Elementary Scalar and Vector Operations 20 
# 1.10 Scalar Product of Two Vectors 21 
# 1.11 Unit Vectors 23 
# 1.12 Vector Product of Two Vectors 25 
# 1.13 Differentiation of a Vector with Respect to a Scalar 29 
# 1.14 Examples of Derivatives—Velocity and Acceleration 30 
# 1.15 Angular Velocity 34 
# 1.16 Gradient Operator 37 
# 1.17 Integration of Vectors 40 
# 2.1 Newtonian Mechanics—Single Particle Introduction 48 
# 2.2 Newton's Laws 49 
# 2.3 Frames of Reference 53 
# 2.4 The Equation of Motion for a Particle 55
# 6.4 The "Second Form" of the Euler Equation 216 
# 6.5 Functions with Several Dependent Variables 218 
# 6.6 Euler Equations When Auxiliary Conditions Are Imposed 219 
# 6.7 The eight Notation 224
# 7.1 Hamilton's Principle—Lagrangian and Hamiltonian Dynamics 228 
# 7.2 Hamilton's Principle 229 
# 7.3 Generalized Coordinates 233 
# 7.4 Lagrange's Equations of Motion in Generalized Coordinates 237 
# 7.5 Lagrange's Equations with Undetermined Multipliers 248 
# 7.6 Equivalence of Lagrange's and Newton's Equations 254 
# 7.7 Essence of Lagrangian Dynamics 257 
# 7.8 A Theorem Concerning the Kinetic Energy 258 
# 7.9 Conservation Theorems Revisited 260 
# 7.10 Canonical Equations of Motion—Hamiltonian Dynamics 265 
# 7.11 Some Comments Regarding Dynamical Variables and Variational Calculations in Physics 272 
# 7.12 Phase Space and Liouville's Theorem (Optional) 274 
# 7.13 Virial Theorem (Optional) 277
# 8.1 Central-Force Motion introduction 287 
# 8.2 Reduced Mass 287 
# 8.3 Conservation Theorems—First Integrals of the Motion 289 
# 8.4 Equations of Motion 291 
# 8.5 Orbits in a Central Field 295 
# 8.6 Centrifugal Energy and the Effective Potential 296 
# 8.7 Planetary Motion—Kepler's Problem 300 
# 8.8 Orbital Dynamics 305 
# 8.9 Apsidal Angles and Precession (Optional) 312 
# 8.10 Stability of Circular Orbits (Optional) 316
# 9.1 Dynamics of a System of Particles 328 
# 9.2 Center of Mass 329 
# 9.3 Linear Momentum of the System 331
# 12.9 The Loaded String 498
# 13.1 Continuous Systems Waves Introduction 512 
# 13.2 Continuous String as a Limiting Case of the Loaded String 513 
# 13.3 Energy of a Vibrating String 516 
# 13.4 Wave Equation 520 
# 13.5 Forced and Damped Motion 522 
# 13.6 General Solutions of the Wave Equation 524 
# 13.7 Separation of the Wave Equation 527 
# 13.8 Phase Velocity, Dispersion, and Attenuation 533 
# 13.9 Group Velocity and Wave Packets 538 
# 14.1 Special Theory of Relativity Introduction 546 
# 14.2 Galilean Invariance 547 
# 14.3 Lorentz Transformation 548 
# 14.4 Experimental Verification of the Special Theory 555 
# 14.5 Relativistic Doppler Effect 558 
# 14.6 Twin Paradox 561 
# 14.7 Relativistic Momentum 562 
# 14.8 Energy 566 
# 14.9 Spacetime and Four-Vectors 569 
# 14.10 Lagrangian Function in Special Relativity 578 
# 14.11 Relativistic Kinematics 579"""

# # Load the table of contents into a TableOfContents object
# CalcTable = load_table_of_contents(CalcTableString)
# CalcTable.fill_end_pages()
# electroTable = load_table_of_contents(electroTableString)
# electroTable.fill_end_pages()
# thermoTable = load_table_of_contents(thermoTableString)
# thermoTable.fill_end_pages()
# quantumTable = load_table_of_contents(quantumTableString)
# quantumTable.fill_end_pages()
# masTable = load_table_of_contents(masTableString)
# masTable.fill_end_pages()



def save_courses(courses, filename):
    with open(filename, 'w') as outfile:
        json_data = jsonpickle.encode(courses)
        outfile.write(json_data)
def main() -> None:
    """Start the bot."""
    
    
    # courses[0].main_note.tableOfContents = thermoTable
    # courses[1].main_note.tableOfContents = quantumTable
    # courses[2].main_note.tableOfContents = masTable
    # courses[3].main_note.tableOfContents = electroTable
    # courses[4].main_note.tableOfContents = CalcTable

    # save_courses(courses, DATA_FILE)
    
    #get_lesson_chapter()
    # Create the Application and pass it your bot's token.
    token = "6262438080:AAFZJvAqdcWK7aXjId1QgXxUkDupIAA_Cwk"
    application = Application.builder().token(token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", botStart))
    application.add_handler(CommandHandler("help", botHelpCommand))
    application.add_handler(CommandHandler("deleteData", deleteData))
    application.add_handler(CommandHandler("today", prepare_for_today_lessons))
    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, botEcho))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(5)
    print("stoped")

if __name__ == "__main__":
    main()
