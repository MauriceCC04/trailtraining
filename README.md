Full User Guide:

First you need to create an API application on strava. 
Record the client ID and secret for later.

First download and install from this repo
then on mac:
Do the following

install GarminDB as per the instructions according to the installation guide

Open terminal and run the following:

mkdir -p ~/.GarminDb

cat > ~/.GarminDb/GarminConnectConfig.json <<'EOF'
{ "db": { "type" : "sqlite" }, "garmin": { "domain" : "garmin.com" }, "credentials": { "user" : "YOUR_GARMIN_EMAIL", "secure_password" : false, "password" : "YOUR_GARMIN_PASSWORD", "password_file" : null }, "data": { "weight_start_date" : "12/31/2019", "sleep_start_date" : "12/31/2019", "rhr_start_date" : "12/31/2019", "monitoring_start_date" : "12/31/2019", "download_latest_activities" : 25, "download_all_activities" : 1000 }, "directories": { "relative_to_home" : true, "base_dir" : "HealthData", "mount_dir" : "/Volumes/GARMIN" }, "enabled_stats": { "monitoring" : true, "steps" : true, "itime" : true, "sleep" : true, "rhr" : true, "weight" : true, "activities" : true }, "course_views": { "steps" : [] }, "modes": { }, "activities": { "display" : [] }, "settings": { "metric" : false, "default_display_activities" : ["walking", "running", "cycling"] }, "checkup": { "look_back_days" : 90 } }
EOF

export TRAILTRAINING_BASE_DIR="$HOME/HealthData"

cd /.../Trailrunning-Training-Project
This is the directory you installed it under

# create + activate venv
python3 -m venv .venv
source .venv/bin/activate

# install deps + install your package (editable)
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# point the project at your local data directory (change if you want)
export TRAILTRAINING_BASE_DIR="$HOME/activity data"

# Now here use the credentials from your strava api application
export STRAVA_CLIENT_ID=""
export STRAVA_CLIENT_SECRET=""

export STRAVA_REDIRECT_URI="http://127.0.0.1:5000/authorization"
export TRAILTRAINING_BASE_DIR="$HOME/trailtraining-data"

# and here just your garmin log in information
export GARMIN_EMAIL=""
export GARMIN_PASSWORD=""

# run the full pipeline
trailtraining run-all

it takes a while to download your garmin data at first
but the next updates are incremental and should not take nearly as long

Instead, if you use intervals.icu, you can run it via intervals.icu's API instead. You must go and get your credentials but it is otherwise simple:

instead of setting the garmin credentials, you set the following:
export INTERVALS_API_KEY=""
export INTERVALS_ATHLETE_ID=""

export OLDEST="2023-01-01" #this is a sample date you can change
export NEWEST="2026-02-27" #another sample date you can change
node scripts/intervals_fetch_wellness.mjs

trailtraining run-all-intervals

this is much faster than the garmin download and but does not include all wellness data

Aftewards, to run the LLM based coaching extension you must do the following:

1. go to open ai api key website
2. add money
3. create api key
4. copy that api key into the following ""
5. and run the following:
export OPENAI_API_KEY=""
export TRAILTRAINING_LLM_MODEL="gpt-5.2"
export TRAILTRAINING_REASONING_EFFORT="medium"   # none|low|medium|high|xhigh
export TRAILTRAINING_VERBOSITY="medium"          # low|medium|high

trailtraining coach --prompt training-plan
trailtraining coach --prompt recovery-status
trailtraining coach --prompt meal-plan

training-plan generates a training plan
recovery-status analyzes your fatigue and recovery
meal-plan generates you a personalized meal plan based on your training and recovery data

additionally,
you could run the following:
if you need to specify the path
trailtraining coach --prompt recovery-status --input /path/to/prompting/
# or a zip export:
trailtraining coach --prompt recovery-status --input /path/to/prompting_bundle.zip

to make it easier,
i have set up a file called .env.local which includes all the local parameters (Login data, api keys, etc)
and run the following instead:
source .env.local

one thing to note is that i have set up this to only download data from December 2023.
This is an arbitrary date I have chosen to ensure that it is not too much data (both for me to download and store or not to confuse the LLM)

Additionally, of course the LLM prompting can also be changed (for different sports or to edit to a user's tastes)

Additionally, if you want to  use multiple users, you have to make sure that in the /.GarminDB directory the json is updated to the new user.