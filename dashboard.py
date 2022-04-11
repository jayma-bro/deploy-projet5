import pandas as pd
import streamlit as st
import requests
import streamlit.components.v1 as components

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='GET', headers=headers, url=model_uri, json=data)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    RAY_SERVE_URI = 'http://localhost:8020/predict'

    st.title('question de StackOverflow')

    title = st.text_input('Titre')

    body = st.text_area('Corp de la question')

    predict_btn = st.button('Prédire')
    if predict_btn:
        data = {'title': title, 'body': body}
        pred: str = request_prediction(RAY_SERVE_URI, data)['result']
        html = ''
        for tag in pred.split():
            html += "<button style='font-size: 1.5em; border-radius: 1em; border-color: blue; border-width: 3px; padding: 3px 8px; background-color: darkblue; color: antiquewhite; margin: 5px;'>" + tag + "</button>"
        components.html(html)


if __name__ == '__main__':
    main()
