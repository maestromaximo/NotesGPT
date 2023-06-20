import json
import base64
import datetime
import tkinter as tk
from tkinter import ttk
import struct
import datetime
import os
import calendar
import time
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import jsonpickle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

def load_courses(filename):
    if not os.path.exists(filename):
        return []

    with open(filename, 'r') as infile:
        json_data = infile.read()
        courses = jsonpickle.decode(json_data)
    return courses

DATA_FILE = "courses_data.json"
courses = load_courses(DATA_FILE)

def deserialize_date(dct):
    if "py/object" in dct and dct["py/object"] == "datetime.date":
        date_bytes = base64.b64decode(dct["__reduce__"][1][0])
        timestamp, = struct.unpack(">i", date_bytes)  # Use big-endian byte order
        return datetime.date.fromtimestamp(timestamp)
    return dct



def decode_transcript(encoded_transcript):
    return base64.b64decode(encoded_transcript.encode('utf-8')).decode('utf-8')


def load_courses_data(file_path):
    with open(file_path, 'r') as f:
        data = json.loads(f.read(), object_hook=deserialize_date)
    return data


def display_spending_details(file_path, threshold):
    current_month = datetime.date.today().month
    current_year = datetime.date.today().year
    month_days = calendar.monthrange(current_year, current_month)[1]
    reset_timestamp = time.mktime(datetime.date(current_year, current_month, month_days).timetuple())

    with open(file_path, 'r') as f:
        spent = float(f.read().strip())

    spending_window = tk.Toplevel()
    spending_window.title("ChatGPT Spending")

    spending_label = tk.Label(spending_window, text=f"Total spent this month: ${spent:.2f}\n\nMonthly spending limit: ${threshold}", font=('Arial', 14))
    spending_label.pack(padx=20, pady=20)

    # Add a bar chart
    figure = plt.Figure(figsize=(5, 3), dpi=100)
    chart = figure.add_subplot(1, 1, 1)
    chart.bar(["Spent", "Threshold"], [spent, threshold], color=["blue", "orange"])
    chart.set_ylim([0, max(spent, threshold) * 1.5])

    for i, v in enumerate([spent, threshold]):
        chart.text(i, v, f"${v:.2f}", ha="center", va="bottom", fontsize=10)

    chart_canvas = FigureCanvasTkAgg(figure, spending_window)
    chart_canvas.draw()
    chart_canvas.get_tk_widget().pack(padx=20, pady=20)

     # Update daily spending history
    update_daily_spending_history('daily_spending_history.txt', spent)
    history_dates, history_spending = get_daily_spending_history('daily_spending_history.txt')
    predicted_monthly_spending = predict_monthly_spending(history_dates, history_spending)

    spending_label.config(text=f"Total spent this month: ${spent:.2f}\n\nPredicted spending this month: ${predicted_monthly_spending:.2f}\n\nMonthly spending limit: ${threshold}")

    if spent >= threshold:
        warning_label = tk.Label(spending_window, text="You have reached or exceeded your monthly spending limit!", fg="red", font=('Arial', 14, 'bold'))
        warning_label.pack(pady=10)
    
    if time.time() > reset_timestamp:
        with open(file_path, 'w') as f:
            f.write("0")

def display_lessons(course_data):
    lessons_window = tk.Toplevel()
    lessons_window.title(f"Lessons for {course_data['name']}")

    tree = ttk.Treeview(lessons_window, columns=('Name', 'Summary'), show='headings')
    tree.heading('Name', text='Lesson Name')
    tree.heading('Summary', text='Lesson Summary')
    tree.pack(fill=tk.BOTH, expand=1)

    lessons = course_data['main_note']['lessons']
    for lesson in lessons:
        tree.insert('', 'end', text=lesson['name'], values=(lesson['name'], lesson['summary']))

    def on_lesson_click(event):
        item_id = tree.identify_row(event.y)
        if not item_id:
            return
        item = tree.item(item_id)
        lesson = None
        for l in lessons:
            if l['name'] == item['values'][0]:
                lesson = l
                break
        if lesson:
            display_lesson_details(lesson)

    tree.bind('<Double-1>', on_lesson_click)



def display_lesson_details(lesson_data):
    details_window = tk.Toplevel()
    details_window.title(f"Details for {lesson_data['name']}")

    content = tk.Text(details_window, wrap=tk.WORD)
    content.insert(tk.END, f"Name: {lesson_data['name']}\n\n")
    datee =lesson_data['date'].strftime('%Y-%m-%d')
    dueDate =lesson_data['due_date'].strftime('%Y-%m-%d')
    firstPage = 1
    lastPage = -1
    for course in courses:
        
        for lesson in course.main_note.lessons:
            if lesson.name == lesson_data['name']:
                datee = lesson.date
                dueDate = lesson.due_date
                firstPage = lesson.firstPage
                lastPage = lesson.lastPage
    content.insert(tk.END, f"First Page: {firstPage}\n\n")
    content.insert(tk.END, f"Last Page: {lastPage}\n\n")
    content.insert(tk.END, f"Date: {datee}\n\n")
    content.insert(tk.END, f"Due Date: {dueDate}\n\n")
    
    content.insert(tk.END, f"Summary: {lesson_data['summary']}\n\n")
    ###########
    
    ###########
    #content.insert(tk.END, f"Date: {lesson_data['date'].strftime('%Y-%m-%d')}\n\n")
    #content.insert(tk.END, f"Due Date: {lesson_data['due_date'].strftime('%Y-%m-%d')}\n\n")

    content.insert(tk.END, f"List of Subjects: {', '.join(lesson_data['listOfSubjects'])}\n\n")
    
    if lesson_data['transcript'] is not None:
        decoded_transcript = decode_transcript(lesson_data['transcript']['py/b64'])
        content.insert(tk.END, f"Transcript:\n{decoded_transcript}\n\n")
    else:
        content.insert(tk.END, f"No transcript available for this lesson.\n\n")
    content.insert(tk.END, f"Content: {lesson_data['content']}\n\n")
    content.config(state=tk.DISABLED)
    content.pack(fill=tk.BOTH, expand=1)

def refresh_data(tree, file_path):
    data = load_courses_data(file_path)
    tree.delete(*tree.get_children())  # Clear the existing data in the Treeview
    for item in data:
        tree.insert('', 'end', text=item['name'], values=(item['name'], item['code'], item['folder']))
    tree.after(10000, lambda: refresh_data(tree, file_path))  # Schedule the function to run again after 10 seconds

def display_courses_data(data):
    root = tk.Tk()
    root.title("Courses Data")

    
    tree = ttk.Treeview(root, columns=('Name', 'Code', 'Folder'), show='headings')
    
    tree.heading('Name', text='Course Name')
    tree.heading('Code', text='Course Code')
    tree.heading('Folder', text='Course Folder')
    tree.pack(fill=tk.BOTH, expand=1)

    
    for item in data:
        tree.insert('', 'end', text=item['name'], values=(item['name'], item['code'], item['folder']))
    spending_button = tk.Button(root, text="Check Spending", command=lambda: display_spending_details('costTracking.txt', 4))
    spending_button.pack(pady=10)
    refresh_data(tree, file_path)  # Call the refresh_data function initially

    def on_course_click(event):
        
        item_id = tree.identify_row(event.y)
        if not item_id:
            return
        item = tree.item(item_id)
        course = None
        for c in data:
            if c['name'] == item['values'][0]:
                course = c
                break
        if course:
            display_lessons(course)

    tree.bind('<Button-1>', on_course_click)

    root.mainloop()


import datetime

def update_daily_spending_history(file_path, spent):
    today = datetime.date.today()
    lines = []

    # Open the file in 'r+' mode to read the existing lines
    with open(file_path, 'r+') as f:
        lines = f.readlines()

    # Check the last line and decide if you need to append or overwrite
    if lines:
        last_date, last_spent = lines[-1].strip().split(',')
        last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()
        if today == last_date:
            # If today is the same as the last date, overwrite the last line
            lines[-1] = f"{today},{spent}\n"
            with open(file_path, 'w') as f:
                f.writelines(lines)
        else:
            # If today is a different date, append a new line
            with open(file_path, 'a') as f:
                f.write(f"{today},{spent}\n")
    else:
        # If the file was empty, just append a new line
        with open(file_path, 'a') as f:
            f.write(f"{today},{spent}\n")


def get_daily_spending_history(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        dates = []
        spending = []
        for line in lines:
            date, spent = line.strip().split(',')
            dates.append(datetime.datetime.strptime(date, "%Y-%m-%d").date())
            spending.append(float(spent))
    return dates, spending

def predict_monthly_spending(dates, spending):
    if len(spending) < 2:
        return spending[-1] if spending else 0

    days_elapsed = [(date - dates[0]).days for date in dates]
    X = np.array(days_elapsed).reshape(-1, 1)
    y = np.array(spending)

    model = LinearRegression().fit(X, y)
    days_in_month = calendar.monthrange(datetime.date.today().year, datetime.date.today().month)[1]
    prediction = model.predict([[days_in_month - 1]])

    return prediction[0]

if __name__ == "__main__":
    file_path = 'courses_data.json'
    courses_data = load_courses_data(file_path)
    display_courses_data(courses_data)
