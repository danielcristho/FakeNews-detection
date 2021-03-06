from flask import Flask, request
import Prediction
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)
@app.route('/sms', methods=['POST'])
def sms():
    resp = MessagingResponse()
    inbMsg = request.values.get('Body')
    pred, confidence = Prediction.detectingFakeNews(inbMsg)

    resp.message(
        f'The news headline you entered is {pred[0]!r} and corresponds to {confidence[0][1]!r}.')
    return str(resp)

if __name__ == '__main__':
    app.run()