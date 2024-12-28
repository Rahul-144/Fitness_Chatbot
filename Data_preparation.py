import requests
from bs4 import BeautifulSoup
import pandas as pd

# Read exercise URLs from a text file
with open('Chatbot/exercise_urls.txt', 'r') as file:
    urls = [line.strip() for line in file.readlines()]

# Function to scrape a single exercise page
def scrape_exercise(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title_tag = soup.find('h1', {'class': 'text-[2.125rem]/10 font-bold text-main-black dark:d-main-black'})
        title = title_tag.get_text(strip=True) if title_tag else "Title not found"

        # Extract description
        description_tag = soup.find('p', { 'class':"text-base/6 text-main-black dark:d-main-black font-normal whitespace-pre-wrap"})
        description = description_tag.get_text(strip=True) if description_tag else "Description not found"

        # Extract target muscle groups
        muscle_groups = []
        muscle_group_tags = soup.select("div.flex.items-center.flex-col > p.text-base\\/6")
        for tag in muscle_group_tags:
            muscle_groups.append(tag.get_text(strip=True))

        # Extract steps (if present)
        steps = []
        if description != "Description not found":
            for line in description.split('.'):
                line = line.strip()
                if ':' in line:  # Look for step-like content based on a colon
                    steps.append(line)
        
        # Fallback for non-step text
        if not steps:
            steps = [description]

        return {
            "title": title,
            "description": description,
            "muscle_groups": ", ".join(muscle_groups),  # Convert list to string
            "steps": " | ".join(steps)  # Join steps with '|' separator
        }
    else:
        return {"title": "Failed to fetch", "description": "N/A", "muscle_groups": "N/A", "steps": "N/A"}
# Scrape all exercises and organize data
all_exercises = []

for url in urls:
    exercise_data = scrape_exercise(url)
    exercise_data['url'] = url  # Add URL for reference
    all_exercises.append(exercise_data)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(all_exercises)

# Save to a CSV file
df.to_csv('exercise_dataset.csv', index=False)