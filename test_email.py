import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")

def send_test_email():
    msg = EmailMessage()
    msg["Subject"] = "✅ Test Email from Accident Detection System"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_TO
    msg.set_content("This is a test email to check Gmail SMTP setup.")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("📧 Test email sent successfully!")
    except Exception as e:
        print(f"⚠️ Email sending failed: {e}")

if __name__ == "__main__":
    send_test_email()
