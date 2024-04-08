
![Logo](logo1.gif)

# AI Explorer NYC

Utilizing AI algorithms to craft personalized itineraries, considering diverse preferences and budgetary constraints for an enhanced travel experience


## 🛠️ Installation Using Conda

To install the project using conda, first navigate to the path where you want to save the files (make sure to change 'your/path' to the real path)
```bash
  cd your/path
```

Then, clone the GitHub repository:
```bash
  git clone https://github.com/LianFi15/AI-Explorer-NYC.git
```
Next, nevigate to the cloned directory: 
(make sure to change 'your/path' to the real path)
```bash
  cd your/path/AI-Explorer-NYC
```

Next, create a conda environment and install the dependencies(this part may take a few minutes):
```bash
  conda create --name NYC-Explorer-env --file spec-file.txt
```
Then, activate the conda environment:
```bash
  conda activate NYC-Explorer-env
```
Next, make sure streamlit is installed in your conda environment:
```bash
  conda install streamlit
```

At last, run the streamlit app and enjoy!
```bash
  streamlit run main.py
```
## 🎨 Setting Up Streamlit App Theme and Colors
Once the Streamlit app is running, ensure that the 'app theme, colors and fonts' option is set to light. Otherwise, the buttons and font may not be visible.

Here's how to do it:

First, select the settings by clicking on the three dots in the top right corner of the Streamlit app.

![](screen_shots/screenshot1.png)

Next, choose the "app theme, colors and fonts" and set it to white. 

![](screen_shots/screenshot2.png)

## 📋 Project Overview
Our project tackles the challenge of trip planning for visitors to New York City. Our solution is an AI-driven trip planner that enables users to input their stay duration and preferences, such as budget, preferred activities, group size, transportation choices, and desired number of activities per day. The system then generates an initial schedule, carefully balancing the inclusion of these attractions. Users can also rank activities, providing feedback for dynamic schedule adjustments.

### 🚀 Features
* Dynamic Schedule Generation: Utilizes Genetic Algorithms (GA) to optimize schedules based on user preferences and budget constraints.

* Personalized Recommendations: Employs KNN Collaborative Filtering for recommending activities based on the preferences of similar users.

* User Feedback Processing: Integrates NLTK Sentiment Analysis to process textual user feedback and extract sentiment information regarding various aspects of the trip.

### 🔒 Privacy Policy
To prioritize user privacy, we provide a comprehensive privacy document that informs users about the data collected, its usage, and what might be shared with others.

### 📝 Contribution and Feedback
At the end of their trip, users can contribute by sharing their experiences through text, enriching the community and providing valuable feedback for further system refinement. Users will also have the option to upload photos of their experiences, adding a personal touch to the system and providing a visual representation of their adventures.
