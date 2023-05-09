
import logging
from dotenv import load_dotenv
import os


from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

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

async def botEcho(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    message_text = update.message.text
    await update.message.reply_text("Got your request! and its been sent to procesing")
    print(message_text)

    # Save the message with the current date and time to a text file
    
    with open("telegram_messages.txt", "a") as file:
        file.write(f"{message_text}\n")

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    token = "YOUR TOKEN"
    application = Application.builder().token(token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", botStart))
    application.add_handler(CommandHandler("help", botHelpCommand))
    application.add_handler(CommandHandler("deleteData", deleteData))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, botEcho))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(5)
    print("stoped")

if __name__ == "__main__":
    main()