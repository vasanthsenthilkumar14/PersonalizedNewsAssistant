# Create a new conda environment named 'news_bot' with Python 3.9
conda create -n news_bot python=3.9

# Activate the environment
conda activate news_bot

# Install required packages (assuming you need these common packages)
pip install -r requirements.txt

# Run the news_chatbot file
python news_chatbot.py