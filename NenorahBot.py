import streamlit as st
from datetime import datetime
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Set up the model and tokenizer
mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

# Define function to display messages with a border
def display_message(sender, message):
    time = datetime.now().strftime("%H:%M:%S")
    if sender == "user":
        st.write(f"<div style='float:left;border:1px solid #ccc;border-radius:5px;background-color:#ddd;color:#000;padding:10px;margin-top:15px;width:40%;'><b>You:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.write(f"<div style='float:right;border:1px solid #ccc;border-radius:5px;background-color:#eee;color:#222;padding:10px;margin-bottom:10px;margin-top:5px;width:40%;'><b>NenorahBot:</b> {message}</div>", unsafe_allow_html=True)

# Define main function for chat app
def chat_app():
    # Set app title and chatbot image
    st.image("https://th.bing.com/th/id/R.081b885f0872ddfd530007a70a0ed590?rik=q3yfESphKfG7Xw&riu=http%3a%2f%2fexperthubrobotics.com%2fportals%2f0%2fImages%2fRobots%2fHire-A-Bot-HRHub-BOT-Final.png&ehk=Wszs0cFOSaNiLOygBcqfUsG6tprtToL5APmVE%2bdWgA4%3d&risl=&pid=ImgRaw&r=0", width=200, use_column_width=False)
    st.markdown("<h1 style='text-align:center;'>Chat with NenorahBot</h1>", unsafe_allow_html=True)

    # Add some margin to push the input box below the chat messages
    st.markdown("<br>", unsafe_allow_html=True)

    # Create input box for user to enter messages
    user_input = st.text_input("Enter message", value="", key="user_input")

    # Create button to submit user message
    if st.button("Send Message"):
        display_message("user", user_input)

        # Encode user input and generate response
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        response_ids = model.generate(input_ids)
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Replace this with your bot's response
        display_message("bot", response_text)

    # Add some margin to separate the input box from the chat messages
    st.markdown("<br>", unsafe_allow_html=True)

chat_app()
