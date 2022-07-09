from flask import Flask
appFlask = Flask(__name__)


@appFlask.route('/home')
def home():
    return "We are learning HTTPS @ EduCBA"

if __name__ == "__main__":
    appFlask.run(ssl_context='adhoc')