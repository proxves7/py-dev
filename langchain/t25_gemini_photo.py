import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Image, display
import google.generativeai as genai
from dotenv import dotenv_values

config = dotenv_values(".env")

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=config.get("GOOGLE_API_KEY")
)

user_messages = []
user_input = "圖片中的生物是什麼？"
user_messages.append({"type": "text", "text": user_input + "請使用繁體中文回答。"})
image_url = "https://i.ibb.co/KyNtMw5/IMG-20240321-172354614-AE.jpg"
user_messages.append({"type": "image_url", "image_url": image_url})
human_messages = HumanMessage(content=user_messages)
result = model.invoke([human_messages])

print("Q: " + user_input)
print("A: " + result.content)

# Display the image
# display(Image(url=image_url))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from io import BytesIO
img = mpimg.imread(BytesIO(requests.get(image_url).content), format='jpg')

plt.imshow(img)
plt.axis('off')
plt.show()