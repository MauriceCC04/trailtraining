TrailTraining: Personal Trail-Running Data Pipeline + LLM Coaching

Overview

This was developed as a project to turn the data collected from mine (or someone else's) Garmin watch and their strava into something useful. It first downloads the Garmin metrics including sleep, Heart Rate, Heart Rate Variability and more using the GarminDB library in python and turns them into json files formatted chronologically. Then it uses the strava api to download the list of historical activities chronologically and pairs them day by day in a json file with the Garmin data. It was designed methodically to make it readable and understandable by an LLM. Then with the OpenAI api, additionally it allows for the user to generate personalized training reports for recovery, meal plans and training plans, of which the prompt can be manually edited to fit one's specific training goals (if necessary). This project is done in python and can be used via a command line interface to 1. initially download ones data into a folder 2. incrementally update their data and generate new json files.

Motivation & Goals

I designed this project because after I was gifted a Garmin watch one year, I started to collect data on myself which was going unused. My main goal was to build a usable tool that takes recovery metrics from Garmin and activity data from Strava to produce readable guidance on training each week.

Data Sources & Processing

The pipeline pulls two main categories of data:

1. Data collected from the Garmin watch (excluding activities) via the GarminDB library. By using the GarminDB library one can incrementally update their data without having to donwload every day again. After download, the processing step cleans it by removing irrelevant fields (stuff which either has not been properly collected or tells no useful information) and creates 3 JSON files that could be used for analysis by an LLM.
- formatted_personal_data.json contains some of the user's constant biometric data like age, height and weight
- shortened_rhr.json contains the user's resting heart rate data over a shortened time frame
- shortened_sleep.json contains the user's sleep data including how long they slept and how 'restorative'
 it was

2. Data from the Strava website via an API application (the user must unfortunately create). The Readme provides more details on creation of the strava API app. This is also downloaded and then stored in a JSON for later use.

Integration & Output Artifacts

A dedicated combine step merges the Garmin data's sleep json and the strava activities to create a day by day chronological json called combined_summary.json. This contains for every day, the sleep data and which activity happened on which day tying directly activity to day. It also includes the time of day that an activity happened because that may also be relevant in analysis, especially if there are multiple activities per day.
In the end, the implementation was a bit messy because when working with the optimal way to prompt the LLM, I was testing with other json files, thus the creation of a processing directory. Additionally, I did not personally use the json of pure resting heart rate data because apparently, that was also included as an element of ones sleep. One thing to note is that if the user achieved a resting heart rate lower outside of sleep it would not be accounted for.

LM Coaching Extension

Once combined_summary.json and formatted_personal_data.json exist, the project can create a Coaching Brief using LLM integration via the OpenAI api. I have chosen 5.2 thinking because it seems to work the best as of right now (1 January 2026) but this will probably change later. The prompts enforce rules like prioritizing recent days (because some random event 2 years ago is probably not relevant). The user can request 3 different coach briefs via the CLI as documented in the README: training plan generation, recovery status or meal plan generation.

Results, Limitations, and Future Work

The main result is a pipeline that works (at least for me) to take one user's data collected from their Garmin watch and Strava activities and produce actual readable guidance from it, like that of an expert coach. The main limitations are reliance on external services like GarminDB, Strava authentication, the fact that the user must create their own API App and of course LLM hallucinations or mistakes. Future work could be enhancing the prompts, experimenting with the best way to let an LLM read the JSON (how to format the JSON, etc) and removing (currently) unneeded steps like the RHR json.
