import base64

import streamlit as st
from datetime import date
import pandas as pd

import os
import csv
from streamlit import session_state as ss
from algorithms import run_genetic_algorithm,read_feedback_from_csv,insert_data,read_csv_to_list,\
    save_user_feedback_to_csv,delete_old_files,remove_prefix,adjusted_preferences_KNN,sum_prices, \
    read_csv_to_dict,save_dict_to_csv,save_user_reviews

def tooltip(hover_text, tooltip_text):
    tooltip_html = f"""
    <style>
    .tooltip {{
      position: relative;
      display: inline-block;
    }}

    .tooltip .tooltiptext {{
      visibility: hidden;
      width: 400px; /* wider tooltip */
      background-color: #d3d3d3; /* lighter grey background */
      color: #000000; /* black font */
      text-align: center;
      border-radius: 6px;
      padding: 5px;
      position: absolute;
      z-index: 1;
      top: -30px;
      left: 50%;
      margin-left: -100px; /* adjust to center the tooltip */
    }}

    .tooltip:hover .tooltiptext {{
      visibility: visible;
    }}
    </style>

    <div class="tooltip"><b>{hover_text}</b> <!-- Bold hover text -->
      <span class="tooltiptext">{tooltip_text}</span>
    </div>
    """
    return tooltip_html
try:
    museums_df = pd.read_csv('museums_data.csv')
    zoos_df = pd.read_csv('zoos_data.csv')
    restaurants_df = pd.read_csv('restaurants_data.csv')
    parks_df = pd.read_csv('parks_df.csv')
    reviews_df = pd.read_csv("users_data/user_reviews.csv",encoding='latin-1')
    def image_to_base64(image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode()

    # Specify the file path of your local image
    image_path = "background.jpg"

    # Convert the image to base64
    image_base64 = image_to_base64(image_path)

    # Use the base64-encoded image in the background CSS
    background_image = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url('data:image/png;base64,{image_base64}');
        background-size: 100vw 100vh;  /* This sets the size to cover 100% of the viewport width and height */
        background-position: center;  
        background-repeat: no-repeat;
    }}
    </style>
    """
    logo_path = "logo1.gif"
    custom_width = 300

    # Read the GIF file as binary data
    with open(logo_path, "rb") as file:
        gif_bytes = file.read()

    # Encode the GIF data as base64
    gif_base64 = base64.b64encode(gif_bytes).decode("utf-8")

    # Generate HTML to embed the GIF with the specified width
    html_code = f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/gif;base64,{gif_base64}" alt="logo" style="width:{custom_width}px;">
    </div>
    """

    # Display the HTML code
    st.markdown(html_code, unsafe_allow_html=True)
    st.markdown(background_image, unsafe_allow_html=True)
    users_df= pd.read_csv("users_data/users.csv")

    st.write('<span style="font-size: 30px; font-weight: bold;">Let\'s Start Planning Your Trip!✨</span>', unsafe_allow_html=True)
    st.write('<span style="font-size: 20px;font-weight: bold;">We\'ll create your personalized trip itinerary based on your feedback and insights from similar users, ensuring a tailored and unforgettable experience.</span>', unsafe_allow_html=True)

    user_privacy1= st.empty()
    user_privacy2 = st.empty()
    user_privacy3 = st.empty()
    user_privacy4 = st.empty()
    user_privacy5 = st.empty()

    user_privacy1.write('❗User Privacy Notice:')
    user_privacy2.write('Your privacy is important to us. When you use our services, we may collect data locally on our devices '
             'to improve our services. Rest assured, your personal information remains confidential and is never '
             'shared with other users or third parties. Any data collected is used internally only. We are committed '
             'to protecting your privacy and ensuring the security of your information.')
    user_privacy3.write('Thank you for trusting us with your data.')
    user_privacy4.write('Sincerely,AI Explorer NYC🌍')

    privacy_button= user_privacy5.checkbox("Agree")

    if privacy_button:
        user_privacy1.empty()
        user_privacy2.empty()
        user_privacy3.empty()
        user_privacy4.empty()
        user_privacy5.empty()

        user1= st.empty()
        user2 = st.empty()
        user3 = st.empty()
        user4 = st.empty()
        user5 = st.empty()
        user6 = st.empty()
        user7 = st.empty()


        user1.write(
            '<span style="font-size: 20px; font-weight: bold; ">First, let\'s check if you already used our system! Please insert your details below:</span>',
            unsafe_allow_html=True)

        # Collect user input for first name
        user2.markdown(tooltip("insert User Name and User ID(hover for more info)", "Insert your full name and a unique user ID"),
                    unsafe_allow_html=True)
        first_name = user3.text_input("First Name:")
        last_name = user4.text_input("Last Name:")
        user6.markdown(
            "<p style='font-size: 16px; color: red;'><b>User ID must be a number</b></p>",
            unsafe_allow_html=True)

        user_id = user5.text_input("User ID:")
        show_trip_planning_options = user7.checkbox("Show Trip Planning Options")


        if user_id:  # Check if the input is not empty
                user_id = int(user_id)
                if show_trip_planning_options:
                    user1.empty()
                    user2.empty()
                    user3.empty()
                    user4.empty()
                    user5.empty()
                    user6.empty()
                    user7.empty()

                    past_activities = []
                    flag_new_user = False

                    def get_past_activities(user_ID):
                        past_activities = []
                        museums_feedback = pd.read_csv("users_data/muesums_feedback.csv")
                        parks_feedback = pd.read_csv("users_data/parks_feedback.csv")
                        restaurants_feedback = pd.read_csv("users_data/restaurant_feedback.csv")
                        zoos_feedback = pd.read_csv("users_data/zoos_feedbacks.csv")

                        museums_result = museums_feedback[museums_feedback['ID'] == user_ID]
                        parks_result = parks_feedback[parks_feedback['ID'] == user_ID]
                        restaurants_result = restaurants_feedback[restaurants_feedback['ID'] == user_ID]
                        zoos_result = zoos_feedback[zoos_feedback['ID'] == user_ID]

                        museums_dict = museums_result.to_dict("records")
                        parks_dict = parks_result.to_dict("records")
                        restaurants_dict = restaurants_result.to_dict("records")
                        zoos_dict = zoos_result.to_dict("records")

                        if len(museums_dict) != 0:
                            past_activities.append(museums_dict[0])
                        if len(parks_dict) != 0:
                            past_activities.append(parks_dict[0])
                        if len(restaurants_dict) != 0:
                            past_activities.append(restaurants_dict[0])
                        if len(zoos_dict) != 0:
                            past_activities.append(zoos_dict[0])

                        return past_activities


                    if ((users_df['First Name'] == first_name) & (users_df['Last Name'] == last_name) & (
                            users_df['ID'] == int(user_id))).any():

                        st.write(
                            f'<span style="font-size: 20px; font-weight: bold;">Nice to see you again {first_name}! Hope you\'ll have an amazing trip!</span>',
                            unsafe_allow_html=True)

                        past_activities = get_past_activities(int(user_id))
                    else:
                        st.write(
                            '<span style="font-size: 18px;">Always nice to meet new people! Hope you\'ll enjoy!</span>',
                            unsafe_allow_html=True)
                        new_row = {"First Name": first_name, "Last Name": last_name, "ID": user_id}
                        flag_new_user = True

                    museums_df = pd.read_csv('museums_data.csv')
                    zoos_df = pd.read_csv('zoos_data.csv')
                    restaurants_df = pd.read_csv(
                        'restaurants_data.csv')
                    parks_df = pd.read_csv('parks_df.csv')

                    user_preferences = {}

                    # Initialize list to store trip history
                    trip_history = []

                    # Initialize dictionary to store feedback
                    feedbacks = {}
                    history_set = set()


                    # Function to save the daily schedule to CSV
                    def save_to_csv(daily_schedule, day):
                        filename = f'trip_day_{day + 1}.csv'
                        if not os.path.exists(filename):  # Check if the file already exists
                            with open(filename, 'w', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows([[activity] for activity in daily_schedule])


                    def save_feedback_to_csv(ss, day):
                        filename = f'feedback_day_{day}.csv'
                        if not os.path.exists(filename):  # Check if the file already exists
                            with open(filename, 'w', newline='') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=ss.keys())
                                writer.writeheader()
                                writer.writerow(ss)
                        save_user_reviews(filename)

                    def restaurant_pref(user_preferences,meals_pref,num_pref,type_pref,vegan_pref,veg_pref,
                                        other_pref,cuisine_choose,cuisine_pref):

                        meals_pref.write(
                            '<span style="font-size: 18px; font-weight: bold; ">How many meals would you like to eat?</span>',
                            unsafe_allow_html=True)
                        num_meals = num_pref.number_input("Insert Here:", step=1, value=1,
                                                    key='num_meals_key_updated')
                        user_preferences["num_restaurants"] = num_meals


                        type_pref.write(
                            '<span style="font-size: 18px; font-weight: bold;">Any preferences?</span>',
                            unsafe_allow_html=True)
                        vegan = vegan_pref.checkbox("Update to Vegan")
                        vegetarian = veg_pref.checkbox("Update to Vegetarian")
                        other = other_pref.checkbox("Doesn't matter")

                        user_preferences["vegan"] = vegan
                        user_preferences["vegetarian"] = vegetarian
                        user_preferences["other"] = other

                        cuisine_choose.write(
                            '<span style="font-size: 18px; font-weight: bold;">Choose your favorite cuisines:</span>',
                            unsafe_allow_html=True)
                        cuisine_options_updated = ["Italian", "Mexican", "Japanese", "Indian", "Thai",
                                                   "Chinese",
                                                   "Mediterranean",
                                                   "French"]
                        selected_cuisines = cuisine_pref.multiselect("Select one or more options:", cuisine_options_updated)

                        #user_preferences["restaurants"] = True
                        user_preferences["cuisine_description"] = selected_cuisines
                        if len(selected_cuisines) == 0:
                            flag_cuisine = True
                            user_preferences["cuisine_description"] = cuisine_options_updated
                            user_preferences["vegan"] = True
                            user_preferences["vegetarian"] = True
                            user_preferences["other"] = True

                        return user_preferences



                    # Date range selection
                    st.subheader("How many days will you stay in NYC?📅")
                    start_date = st.date_input("Start date", date.today(), min_value=date.today())
                    end_date = st.date_input("End date", date.today(), min_value=date.today())


                    st.write("Selected start date:", start_date)
                    st.write("Selected end date:", end_date)
                    st.write("Total Days:", (end_date - start_date).days +1)

                    st.subheader("How much do you plan to spend during your trip?💵")
                    options_budget = ["Less than 500 USD", "Between 500 and 1000 USD", "Between 1000 and 2000 USD",
                                      "Between 2000 and 5000 USD",
                                      "More than 5000 USD"]
                    selected_option_budget = st.selectbox("Consider Stay and Activities", options_budget)

                    budget_mapping = {
                        "Less than 500 USD": 500,
                        "Between 500 and 1000 USD": 1000,
                        "Between 1000 and 2000 USD": 2000,
                        "Between 2000 and 5000 USD": 5000,
                        "More than 5000 USD": 1000000
                    }

                    user_preferences["budget"] = budget_mapping[selected_option_budget]

                    st.subheader("How do you plan to get around on your trip? 👣🚕")
                    options_get_around = ["By foot", "Metro or Cab", "Some by foot and some by transportation",
                                          "Haven't decided yet"]
                    selected_option_get_around = st.selectbox("Consider Stay and Activities", options_get_around)
                    user_preferences["preferred_transportation"] = selected_option_get_around

                    if selected_option_get_around in ["By foot", "Some by foot and some by transportation",
                                                      "Haven't decided yet"]:

                        st.write(
                            '<span style="font-size: 18px; font-weight: bold; ">How much distance would you like to go by foot?</span>',
                            unsafe_allow_html=True)
                        options_distance = ["1 KM", "Between 1 and 5 KM", "Between 5 and 10 KM", "More than 10 KM"]
                        selected_option_distance = st.selectbox("Consider Stay and Activities", options_distance)

                        walk_distance_mapping = {
                            "1 KM": 1,
                            "Between 1 and 5 KM": 5,
                            "Between 5 and 10 KM": 10,
                            "More than 10 KM": 200  # Assuming "More than 10 KM" corresponds to 10
                        }

                        user_preferences["walking_distance"] = walk_distance_mapping[selected_option_distance]
                    else:
                        user_preferences["walking_distance"] = 0

                    st.subheader("Choose your trip preferences")
                    st.markdown(tooltip("You can choose more than one (hover for more info)",
                                        "You must choose at least one option"),
                                unsafe_allow_html=True)
                    museums = st.checkbox("Museums 🏛️")
                    parks = st.checkbox("Parks 🌳🌸")
                    zoos = st.checkbox("Zoos🐘🦁🦒")


                    if museums:
                        user_preferences["museums"] = True
                    else:
                        user_preferences["museums"] = False

                    if parks:
                        user_preferences["parks"] = True
                    else:
                        user_preferences["parks"] = False

                    if zoos:
                        user_preferences["zoos"] = True
                    else:
                        user_preferences["zoos"] = False

                    st.subheader("Are you interested in exploring local restaurants during your trip?🍽️")
                    restaurants = st.checkbox("Yes! I would love to visit restaurants")


                    num_meals = 0  # Initialize num_meals with a default value
                    flag_cuisine = False
                    if restaurants:
                        st.write(
                            '<span style="font-size: 18px; font-weight: bold; ">How many meals would you like to eat?</span>',
                            unsafe_allow_html=True)
                        num_meals = st.number_input("Insert Here:", step=1, value=1, key='num_meals_key')
                        user_preferences["num_restaurants"] = num_meals

                        st.write(
                            '<span style="font-size: 18px; font-weight: bold;">Any preferences?</span>',
                            unsafe_allow_html=True)
                        st.markdown(tooltip("Select your preferences(hover for more info)",
                                            "If you won't choose, we'll suggest all the options we have"),
                                    unsafe_allow_html=True)
                        vegan = st.checkbox("Vegan")
                        vegetarian = st.checkbox("Vegetarian")
                        other = st.checkbox("Doesn't mind")

                        user_preferences["vegan"] = vegan
                        user_preferences["vegetarian"] = vegetarian
                        user_preferences["other"] = other

                        st.write(
                            '<span style="font-size: 18px; font-weight: bold;">Choose your favorite cuisines:</span>',
                            unsafe_allow_html=True)
                        cuisine_options = ["Italian", "Mexican", "Japanese", "Indian", "Thai", "Chinese", "Mediterranean",
                                           "French"]
                        selected_cuisines = st.multiselect("Select one or more:", cuisine_options)

                        user_preferences["restaurants"] = True
                        user_preferences["cuisine_description"] = selected_cuisines
                        if len(selected_cuisines) == 0:
                            flag_cuisine = True
                            user_preferences["cuisine_description"] = cuisine_options
                            user_preferences["vegan"] = True
                            user_preferences["vegetarian"] = True
                            user_preferences["other"] = True

                    else:
                        user_preferences["cuisine_description"] = []
                        user_preferences["num_restaurants"] = 0
                        user_preferences["restaurants"] = False

                    st.subheader("How many activities per day would you like to do?🎉🤸‍♂️")
                    options_activities = st.number_input("Insert Here:", step=1, value=1, key='num_activities_key')
                    user_preferences["num_activities"] = options_activities

                    st.subheader("📷 Capture the Magic: Explore Moments Shared by Our Community! 🌟")
                    photos_folder = 'photos'
                    shared_photos = os.listdir(photos_folder)
                    photo_container = st.container()
                    images = []
                    with photo_container:
                        for i, photo in enumerate(shared_photos):
                            if photo.endswith(".jpg") or photo.endswith(".jpeg") or photo.endswith(".png"):
                                images.append(os.path.join(photos_folder, photo))
                    st.image(images, width=200)


                    if (end_date - start_date).days < 0:
                        total_days = -1 * ((end_date - start_date).days +1)
                    else:
                        total_days = (end_date - start_date).days +1
                    total_cost = 0
                    if len(ss) > 3:
                        try:
                            save_feedback_to_csv(ss, ss['current_day'])
                            lst_day_plan = read_csv_to_list(ss['current_day'])
                            feedback_dict = read_feedback_from_csv(ss['current_day'])
                            insert_data(lst_day_plan, user_id, feedback_dict, museums_df, restaurants_df, parks_df, zoos_df)
                            if flag_new_user:
                                users_df.loc[len(users_df)] = new_row
                                users_df.to_csv('users_data/users.csv', index=False)
                        except :
                            pass

                    current_day = st.session_state.get('current_day', 0)
                    if current_day == total_days:
                        total_cost = sum_prices(total_days+1)
                        delete_old_files(total_days)
                        if os.path.exists("user_preferences.csv"):
                            os.remove("user_preferences.csv")
                        st.write("End of trip. No more days to plan.")
                        st.header("Your Trip to NYC is over!🗽🚕🎉")
                        st.write(f"Total Cost of Your Trip: {total_cost} USD")
                        st.subheader("We would like to hear about your experience 🌟🗣️")
                        st.write("Please provide feedback on what you liked, disliked, and any suggestions for improvement.")
                        user_feedback = st.text_area("Your Feedback:", "")

                        # Button to submit feedback
                        if st.button("Submit Feedback"):
                            save_user_feedback_to_csv(user_id, user_feedback)
                            print(f"user feedback:{user_feedback}")
                            st.success("Thank you for sharing your feedback! We appreciate it.")
                        # File uploader for photos
                        st.subheader("Share Your Trip Experience!📸🗺️")
                        st.write("Upload photos from your trip to help others see and plan their own amazing experiences.")
                        uploaded_files = st.file_uploader("Choose photos from your trip", accept_multiple_files=True,
                                                          type=["jpg", "jpeg", "png"])

                        # Button to submit feedback and uploaded photos
                        if st.button("Share Photos and Feedback"):
                            st.success("Thank you for sharing your photos and feedback! Your experiences will inspire others.")
                            photos_folder = "photos"
                            os.makedirs(photos_folder, exist_ok=True)
                            # Iterate over the uploaded files and save them to the "photos" folder
                            for i, uploaded_file in enumerate(uploaded_files):
                                file_extension = uploaded_file.name.split(".")[-1]  # Get file extension
                                file_path = os.path.join(photos_folder, f"photo_{i}_{user_id}.{file_extension}")
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())

                    elif current_day == 0 and not museums and not parks and not zoos and not restaurants:
                        st.markdown(
                            "<p style='font-size: 18px; color: red;'><b>❗You must choose at least one trip preference</b></p>",
                            unsafe_allow_html=True)

                    elif current_day == 0 and not museums and not parks and not zoos:
                        st.markdown(
                            "<p style='font-size: 18px; color: red;'><b>❗You must choose at least one trip preference other than restaurants</b></p>",
                            unsafe_allow_html=True)

                    elif current_day==0 and st.button(f"Plan day {current_day + 1}"):
                        user_preferences["more_museums"] = False
                        user_preferences["more_zoos"] = False
                        user_preferences["more_parks"] = False
                        user_preferences["rest_updates"]= False

                        save_dict_to_csv(user_preferences,'user_preferences.csv')

                        if restaurants and flag_cuisine:
                            st.markdown(
                                "<p style='font-size: 18px;'><b>You didn't choose your favourite cuisine, so we'll suggest all types!🍕🍔🍟</b></p>",
                                unsafe_allow_html=True)
                        schedule_day = run_genetic_algorithm(user_preferences, museums_df, zoos_df, restaurants_df,
                                                             parks_df, past_activities)

                        schedule_per_day = [schedule_day]
                        current_day = st.session_state.get('current_day', 0)
                        st.session_state['current_day'] = current_day + 1
                        submit_feedback = False
                        if current_day < total_days:
                            st.subheader(f"Trip Schedule for Day {current_day + 1}:")
                            st.write(
                                '<span style="font-size: 18px; font-weight: bold;">The reason we believe these attractions are a great match is that similar users with preferences like yours have genuinely enjoyed them 🧳✈️</span>',
                                unsafe_allow_html=True)
                            save_to_csv(schedule_per_day[0], current_day)
                            my_dict = {}
                            submit_feedback = False
                            with st.form(key='feedback_form'):
                                for i, current_activity in enumerate(schedule_per_day[0]):
                                    st.write(
                                        f'<span style="font-size: 20px; font-weight: bold; color : #0053a5 ; ">Attraction {i+1}</span>',
                                        unsafe_allow_html=True)
                                    activity = current_activity[0] + current_activity[1]
                                    name_of_activity = current_activity[0].split(":")[1].strip()
                                    name_of_df = current_activity[0].split(":")[0].strip()
                                    history_set.add(remove_prefix(current_activity[0]))
                                    st.write(
                                        f'<span style="font-size: 18px; font-weight: bold;"> {name_of_df}</span>',
                                        unsafe_allow_html=True)
                                    st.write(name_of_activity)
                                    st.write(current_activity[1].replace(",", "").strip())

                                    if name_of_df == "Museum":
                                        df = museums_df
                                    elif name_of_df == "Zoo":
                                        df = zoos_df
                                    elif name_of_df == "Restaurant":
                                        df = restaurants_df
                                    elif name_of_df == "Park":
                                        df = parks_df
                                    museum_info = df[df['name'] == name_of_activity]['Information'].values[0]
                                    st.write(
                                        '<span style="font-size: 18px; font-weight: bold;">Information</span>',
                                        unsafe_allow_html=True)
                                    st.write(museum_info)

                                    st.write(
                                        '<span style="font-size: 18px; font-weight: bold;">Reviews from other users:</span>',
                                        unsafe_allow_html=True)
                                    reviews_for_activity = reviews_df[reviews_df['activity'] == name_of_activity][
                                        'review']
                                    flag_review_1 = True
                                    if len(reviews_for_activity) == 0:
                                        flag_review_1 = False
                                        st.write("We currently do not have any reviews available for this activity.")
                                    check_nulls = []
                                    for review in reviews_for_activity:
                                        check_nulls.append(review)
                                    flag_review = True
                                    for review in check_nulls:
                                        if not pd.isnull(review):
                                            flag_review = False
                                    if flag_review == False:
                                        #st.text("Reviews provided by other users:")
                                        for review in reviews_for_activity:
                                            if not pd.isnull(review):  # Check if review is not NaN
                                                st.write(review)
                                    if flag_review and flag_review_1:
                                        st.write("We currently do not have any reviews available for this activity.")

                                    feedback_key = f"feedback_{current_day}_{activity}"  # Generate feedback key
                                    feedback_value = st.slider(f"How much did you enjoy this activity?", 1, 5,
                                                               key=feedback_key)  # Use feedback key to retrieve slider value
                                    my_dict[(current_day, activity)] = feedback_value  # Store feedback in dictionary
                                    st.write(
                                        '<span style="font-size: 16px; font-style: italic; color : blue;">Your feedback matters! Please consider adding a review to help enhance the experience for others🌈</span>',
                                        unsafe_allow_html=True)
                                    user_review = st.text_area("Your review:",
                                                               key=f"review_{current_day}_{name_of_activity}")

                                submit_feedback = st.form_submit_button("Submit Feedback")

                    elif current_day>0 :
                        new_pref = st.empty()
                        new_att_title = st.empty()
                        new_attractions_num= st.empty()
                        new_mus = st.empty()
                        new_parks= st.empty()
                        new_zoos= st.empty()
                        new_pref.subheader("Let us know if your preferences have changed today🔄")
                        user_preferences_updated= read_csv_to_dict('user_preferences.csv')
                        new_att_= new_att_title.checkbox("Change the attractions number per day")
                        if new_att_:
                            options_activities = new_attractions_num.number_input("Insert Here:", step=1, value=1, key='num_activities_key_new')
                            user_preferences_updated["num_activities"] = options_activities

                        more_museums=False
                        if user_preferences_updated['museums']:
                            more_museums= new_mus.checkbox("Add more museums tomorrow", value=more_museums)
                            user_preferences_updated["more_museums"] = more_museums

                        more_parks=False
                        if user_preferences_updated['parks']:
                            more_parks= new_parks.checkbox("Add more parks tomorrow")
                            user_preferences_updated["more_parks"] = more_parks

                        more_zoos=False
                        if user_preferences_updated['zoos']:
                            more_zoos= new_zoos.checkbox("Add more zoos tomorrow")
                            user_preferences_updated["more_zoos"] = more_zoos

                        rest_updates=False
                        rest_pref= st.empty()
                        meals_pref= st.empty()
                        num_pref= st.empty()
                        type_pref= st.empty()
                        vegan_pref= st.empty()
                        veg_pref=st.empty()
                        other_pref=st.empty()
                        cuisine_choose=st.empty()
                        cuisine_pref=st.empty()
                        if user_preferences_updated['restaurants']:
                            if rest_pref.checkbox("I want to change my restaurant preferences"):
                                user_preferences_updated= restaurant_pref(user_preferences_updated,meals_pref,num_pref,type_pref,vegan_pref,
                                                veg_pref,other_pref,cuisine_choose,cuisine_pref)

                                user_preferences_updated['restaurants']= True
                                user_preferences_updated['rest_updates'] = True
                                rest_updates=True

                        less_data= st.empty()
                        note_less_data=st.empty()
                        less_museum= st.empty()
                        less_parks= st.empty()
                        less_zoos= st.empty()
                        mues_markdown=st.empty()
                        parks_markdown=st.empty()
                        zoos_markdown=st.empty()
                        less_data.subheader("Let us know if there are activities you don't want to visit anymore 🚫")
                        note_less_data.markdown(
                            "<p style='font-size: 18px; color: red;'><b>Make sure not to deselect all options </b></p>",
                            unsafe_allow_html=True)
                        dont_visit_museums=False
                        if user_preferences_updated['museums']:
                            dont_visit_museums = less_museum.checkbox("I don't want to visit Museums")
                            user_preferences_updated["museums_less"] = dont_visit_museums

                        dont_visit_parks=False
                        if user_preferences_updated['parks'] :
                            dont_visit_parks = less_parks.checkbox("I don't want to visit Parks")
                            user_preferences_updated["parks_less"] = dont_visit_parks

                        dont_visit_zoos=False
                        if user_preferences_updated['zoos']:
                            dont_visit_zoos = less_zoos.checkbox("I don't want to visit Zoos")
                            user_preferences_updated["zoos_less"] = dont_visit_zoos

                        if dont_visit_museums and more_museums:
                            mues_markdown.markdown(
                                "<p style='font-size: 18px; color: red;'><b>❗You can't deselect an activity and request to do more of it </b></p>",
                                unsafe_allow_html=True)

                        if dont_visit_zoos and more_zoos:
                            zoos_markdown.markdown(
                                "<p style='font-size: 18px; color: red;'><b>❗You can't deselect an activity and request to do more of it </b></p>",
                                unsafe_allow_html=True)

                        if dont_visit_parks and more_parks:
                            parks_markdown.markdown(
                                "<p style='font-size: 18px; color: red;'><b>❗You can't deselect an activity and request to do more of it </b></p>",
                                unsafe_allow_html=True)


                        more_data= st.empty()
                        add_museum= st.empty()
                        add_parks= st.empty()
                        add_zoos= st.empty()

                        activities_bool = [user_preferences_updated['museums'], user_preferences_updated['zoos'],user_preferences_updated['parks'],
                                           user_preferences_updated['restaurants']]
                        if all(activities_bool):
                            st.markdown(
                                    "<p style='font-size: 18px;'><b>You currently have all the options for our activities, but stay tuned! We'll add more in the future</b></p>",
                                    unsafe_allow_html=True)

                        more_data.subheader("Let us know if you wish to try activities you didn't try!🌟")
                        if not user_preferences_updated['museums']:
                            add_new_museums = add_museum.checkbox("I want to visit Museums during my trip")
                            user_preferences_updated["museums_new"] = add_new_museums

                        if not user_preferences_updated['parks']:
                            add_new_parks = add_parks.checkbox("I want to visit Parks during my trip")
                            user_preferences_updated["parks_new"] = add_new_parks

                        if not user_preferences_updated['zoos']:
                            add_new_zoos = add_zoos.checkbox("I want to visit Zoos during my trip")
                            user_preferences_updated["zoos_new"] = add_new_zoos

                        rest_pref_new= st.empty()
                        meals_pref_new= st.empty()
                        num_pref_new= st.empty()
                        type_pref_new= st.empty()
                        vegan_pref_new= st.empty()
                        veg_pref_new=st.empty()
                        other_pref_new=st.empty()
                        cuisine_choose_new=st.empty()
                        cuisine_pref_new=st.empty()
                        if not user_preferences_updated['restaurants']:
                                if rest_pref_new.checkbox("I want to visit Restaurants during my trip"):
                                    user_preferences_updated= restaurant_pref(user_preferences_updated,meals_pref_new,
                                                                              num_pref_new,type_pref_new,vegan_pref_new,
                                                                              veg_pref_new,other_pref_new,cuisine_choose_new,
                                                                              cuisine_pref_new)
                        save_dict_to_csv(user_preferences_updated,'user_preferences.csv')

                        if st.button(f"Plan day {current_day + 1}"):
                            user_preferences_updated= read_csv_to_dict('user_preferences.csv')
                            if user_preferences_updated['num_restaurants']>0 and len(user_preferences_updated['cuisine_description'])>0:
                                user_preferences_updated['restaurants']= True
                            if 'museums_less' in user_preferences_updated:
                                user_preferences_updated['museums']=not user_preferences_updated['museums_less']
                            if 'parks_less' in user_preferences_updated:
                                user_preferences_updated['parks']=not user_preferences_updated['parks_less']
                            if 'zoos_less' in user_preferences_updated:
                                user_preferences_updated['zoos']= not user_preferences_updated['zoos_less']
                            if 'restaurants_less' in user_preferences_updated:
                                user_preferences_updated['restaurants'] = not user_preferences_updated['restaurants_less']
                            if 'museums_new' in user_preferences_updated:
                                user_preferences_updated['museums'] = user_preferences_updated['museums_new']
                            if 'parks_new' in user_preferences_updated:
                                user_preferences_updated['parks']=user_preferences_updated['parks_new']
                            if 'zoos_new' in user_preferences_updated:
                                user_preferences_updated['zoos']=user_preferences_updated['zoos_new']


                            schedule_day = adjusted_preferences_KNN(user_preferences_updated, user_id, museums_df, zoos_df,
                                                                    restaurants_df, parks_df,current_day+1)

                            schedule_per_day = [schedule_day]

                            current_day = st.session_state.get('current_day', 0)
                            st.session_state['current_day'] = current_day + 1
                            submit_feedback = False
                            if current_day < total_days:
                                save_dict_to_csv(user_preferences_updated, 'user_preferences.csv')
                                st.subheader(f"Trip Schedule for Day {current_day + 1}:")
                                save_to_csv(schedule_per_day[0], current_day)
                                st.write(
                                    '<span style="font-size: 18px; font-weight: bold;">We believe these attractions are a great match for you because similar users with comparable feedback from previous days have enjoyed them✨</span>',
                                    unsafe_allow_html=True)
                                save_to_csv(schedule_per_day[0], current_day)
                                my_dict = {}
                                submit_feedback = False

                                with st.form(key='feedback_form'):
                                    for i,current_activity in enumerate(schedule_per_day[0]):

                                        st.write(
                                            f'<span style="font-size: 20px; font-weight: bold;  color : #0053a5; ">Attraction {i+1}</span>',
                                            unsafe_allow_html=True)
                                        activity = current_activity[0] + current_activity[1]
                                        name_of_activity = current_activity[0].split(":")[1].strip()
                                        name_of_df = current_activity[0].split(":")[0].strip()

                                        history_set.add(remove_prefix(current_activity[0]))
                                        st.write(
                                            f'<span style="font-size: 18px; font-weight: bold;"> {name_of_df}</span>',
                                            unsafe_allow_html=True)
                                        st.write(name_of_activity)
                                        st.write(current_activity[1].replace(",", "").strip())


                                        if name_of_df == "Museum":
                                            df = museums_df
                                        elif name_of_df == "Zoo":
                                            df = zoos_df
                                        elif name_of_df == "Restaurant":
                                            df = restaurants_df
                                        elif name_of_df == "Park":
                                            df = parks_df


                                        museum_info = df[df['name'] == name_of_activity]['Information'].values[0]
                                        st.write(
                                            '<span style="font-size: 18px; font-weight: bold;">Information</span>',
                                            unsafe_allow_html=True)
                                        st.write(museum_info)

                                        st.write(
                                            '<span style="font-size: 18px; font-weight: bold;">Reviews from other users:</span>',
                                            unsafe_allow_html=True)
                                        reviews_for_activity = reviews_df[reviews_df['activity'] == name_of_activity][
                                            'review']
                                        flag_review_1 = True
                                        if len(reviews_for_activity) == 0:
                                            flag_review_1 = False
                                            st.write(
                                                "We currently do not have any reviews available for this activity.")
                                        check_nulls = []
                                        for review in reviews_for_activity:
                                            check_nulls.append(review)
                                        flag_review = True
                                        for review in check_nulls:
                                            if not pd.isnull(review):
                                                flag_review = False
                                        if flag_review == False:
                                            for review in reviews_for_activity:
                                                if not pd.isnull(review):  # Check if review is not NaN
                                                    st.write(review)
                                        if flag_review and flag_review_1:
                                            st.write(
                                                "We currently do not have any reviews available for this activity.")

                                        feedback_key = f"feedback_{current_day}_{activity}"  # Generate feedback key
                                        feedback_value = st.slider(f"How much did you enjoy this activity?", 1, 5,
                                                                   key=feedback_key)  # Use feedback key to retrieve slider value
                                        my_dict[(current_day, activity)] = feedback_value  # Store feedback in dictionary
                                        st.write(
                                            '<span style="font-size: 16px; font-style: italic; color : blue;">Your feedback matters! Please consider adding a review to help enhance the experience for others🌈</span>',
                                            unsafe_allow_html=True)
                                        user_review = st.text_area("Your review:",
                                                                   key=f"review_{current_day}_{name_of_activity}")


                                    submit_feedback = st.form_submit_button("Submit Feedback")
                                    new_pref.empty()
                                    new_mus.empty()
                                    new_parks.empty()
                                    new_zoos.empty()
                                    rest_pref.empty()
                                    meals_pref.empty()
                                    add_museum.empty()
                                    add_parks.empty()
                                    add_zoos.empty()
                                    more_data.empty()
                                    num_pref.empty()
                                    type_pref.empty()
                                    vegan_pref.empty()
                                    veg_pref.empty()
                                    other_pref.empty()
                                    cuisine_pref.empty()
                                    cuisine_choose.empty()
                                    rest_pref_new.empty()
                                    meals_pref_new.empty()
                                    num_pref_new.empty()
                                    type_pref_new.empty()
                                    vegan_pref_new.empty()
                                    veg_pref_new.empty()
                                    other_pref_new.empty()
                                    cuisine_choose_new.empty()
                                    cuisine_pref_new.empty()
                                    new_att_title.empty()
                                    new_attractions_num.empty()
                                    less_data.empty()
                                    less_museum.empty()
                                    less_parks.empty()
                                    less_zoos.empty()
                                    mues_markdown.empty()
                                    parks_markdown.empty()
                                    zoos_markdown.empty()
                                    note_less_data.empty()



except Exception as e:
    print(e)
    st.error("An error occurred. Please contact the developer of this website")
