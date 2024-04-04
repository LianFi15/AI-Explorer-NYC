import random
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import os
import csv
import ast
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Genetic algorithm parameters
population_size = 50
mutation_rate = 0.2
num_generations = 100

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth using Haversine formula."""
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def calculate_distance(activity, user_location):
    """Calculate the distance between an activity and the user's location."""
    activity_location = (activity['latitude'], activity['longitude'])
    return haversine_distance(*activity_location, *user_location)


def initialize_population(preferences, museums, zoos, restaurants, parks, past_activities):
    population = []
    for _ in range(population_size):
        day_plan = []
        remaining_activities= preferences["num_activities"]
        remaining_budget = preferences["budget"]
        remaining_restaurants = preferences["num_restaurants"]

        while remaining_activities >=0 :
            # Determine the maximum ticket price allowed based on the remaining budget
            max_ticket_price = remaining_budget / (preferences["num_activities"])

            # Select a random activity based on preferences and remaining budget
            if preferences["museums"]:
                museum = random.choice(museums + past_activities)
                if museum not in day_plan and museum.get('ticket_price', 0) <= max_ticket_price:
                    day_plan.append(museum)
                    remaining_budget -= museum.get('ticket_price', 0)
                    remaining_activities -=1

            if preferences["zoos"]:
                zoo = random.choice(zoos + past_activities)
                if zoo not in day_plan and zoo.get('ticket_price', 0) <= max_ticket_price:
                    day_plan.append(zoo)
                    remaining_budget -= zoo.get('ticket_price', 0)
                    remaining_activities -= 1

            if preferences["parks"]:
                park = random.choice(parks + past_activities)
                if park not in day_plan and park.get('ticket_price', 0) <= max_ticket_price:
                    day_plan.append(park)
                    remaining_budget -= park.get('ticket_price', 0)
                    remaining_activities -= 1

            # Break the loop if there's not enough budget left for any activity
            if remaining_budget <= 0:
                break

        if preferences["restaurants"]:
            max_ticket_price = remaining_budget / (preferences["num_activities"])
            # Filter restaurants based on preferred cuisine description
            preferred_cuisines = preferences["cuisine_description"]
            filtered_restaurants = [r for r in restaurants if r["cuisine_description"] in preferred_cuisines]

            # Initialize a list to store the filtered restaurants based on dietary preferences
            filtered_by_diet = []

            # Check if any dietary preferences are selected
            if preferences["vegan"]:
                filtered_by_diet.extend([r for r in filtered_restaurants if r["is_vegan"] == 1])
            if preferences["vegetarian"]:
                filtered_by_diet.extend([r for r in filtered_restaurants if r["is_vegetarian"] == 1])
            if preferences["other"]:
                filtered_by_diet.extend(
                    [r for r in filtered_restaurants if r["is_vegan"] == 0 and r["is_vegetarian"] == 0])

            while remaining_restaurants >= 0:
                # Select a random restaurant from the filtered list or past activities
                if filtered_by_diet:
                    restaurant = random.choice(filtered_by_diet + past_activities)
                else:
                    # If no specific dietary preference is selected, choose from all filtered restaurants
                    restaurant = random.choice(filtered_restaurants + past_activities)

                # Add the selected restaurant to the day plan and update remaining budget
                if restaurant not in day_plan and restaurant.get('ticket_price', 0) <= max_ticket_price:
                    day_plan.append(restaurant)
                    remaining_budget -= restaurant['ticket_price']
                    remaining_restaurants -= 1

        population.append(day_plan)

    return population


def mutation(day_plan, preferences):
    if len(day_plan) < 2:
        return day_plan  # Skip mutation if there are fewer than 2 activities

    # Randomly select two distinct positions to swap
    position1, position2 = random.sample(range(len(day_plan)), 2)

    # Perform swap mutation if the positions are distinct and activities are different
    if position1 != position2:
        activity1 = day_plan[position1]
        activity2 = day_plan[position2]
        if activity1 != activity2:
            day_plan[position1] = activity2
            day_plan[position2] = activity1

    # Calculate total cost after mutation
    total_cost_after = sum(activity.get('ticket_price', 0) for activity in day_plan)

    # Adjust ticket prices if total cost exceeds the budget
    if total_cost_after > preferences["budget"]:
        excess_cost = total_cost_after - preferences["budget"]
        num_activities = len(day_plan)
        excess_cost_per_activity = excess_cost / num_activities

        # Reduce ticket prices of activities proportionally, ensuring minimum ticket price constraint
        for activity in day_plan:
            new_price = max(activity['ticket_price'] - excess_cost_per_activity, 0)
            activity['ticket_price'] = new_price

    return day_plan



def crossover(parent1, parent2):
    if len(parent1) < 2 or len(parent2) < 2:
        return parent1, parent2  # No crossover if either parent is too short

    # Select a random crossover point
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)

    # Create children by swapping segments between parents
    child1 = parent1[:crossover_point] + [activity for activity in parent2 if activity not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [activity for activity in parent1 if activity not in parent2[:crossover_point]]

    return child1, child2


def calculate_fitness(day_plan, preferences):
    total_cost = sum(activity.get('ticket_price', 0) for activity in day_plan)

    # Check if total cost exceeds the budget
    if total_cost > preferences["budget"]:
        return 0  # Assign fitness of 0 if the day plan exceeds the budget

    # Calculate total distance from activities to the user's location
    user_location = (0, 0)  # Assume user is at the origin for simplicity
    total_distance = sum(calculate_distance(activity, user_location) for activity in day_plan)

    # Fitness penalized if the cost exceeds the budget
    fitness = preferences["budget"] - total_cost

    # Adjust fitness based on preferred transportation mode and walking distance
    if preferences["preferred_transportation"] == "By foot" and total_distance > preferences["walking_distance"]:
        fitness *= 0.5  # Penalize fitness if total distance exceeds walking distance

    return fitness


def selection(population, preferences):
    selected_parents = []
    tournament_size = min(5, len(population))  # Ensure tournament size is within bounds

    for _ in range(len(population)):
        # Perform tournament selection
        tournament_contestants = random.choices(population, k=tournament_size)
        winner = max(tournament_contestants, key=lambda x: calculate_fitness(x, preferences))
        selected_parents.append(winner)

    return selected_parents


# Genetic algorithm
def genetic_algorithm(preferences,museums,zoos,restaurants,parks,past_activities):
    population = initialize_population(preferences,museums,zoos,restaurants,parks,past_activities)
    for generation in range(num_generations):
        # Selection
        selected_parents = selection(population,preferences)

        # Crossover
        new_population = []
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])

        # Mutation
        for day_plan in new_population:
            if random.random() < mutation_rate:
                day_plan = mutation(day_plan,preferences)

        # Update population
        population = new_population

    # Select the best day plan from the final population
    best_day_plan = max(population, key=lambda x: calculate_fitness(x, preferences))
    return best_day_plan

def has_duplicates(lst):
    seen = set()
    unique_list = []
    for item in lst:
        if item["name"] not in seen:
            unique_list.append(item)
            seen.add(item["name"])
    return len(unique_list) < len(lst), unique_list

def check_activity_type(activity, museums, zoos, restaurants,parks):
    if activity.strip() in museums['name'].str.strip().values:
        return "Museum:"
    elif activity.strip() in zoos['name'].str.strip().values:
        return "Zoo:"
    elif activity.strip() in restaurants['name'].str.strip().values:
        return "Restaurant:"
    else:
        return "Park:"


def create_final_plan(trip_plan, num_restaurants, num_other_activities):
    restaurant_count = sum(1 for activity, _ in trip_plan if activity.startswith("Restaurant"))

    new_list = []
    for activity, price in trip_plan:
        if activity.startswith("Restaurant") and num_restaurants > 0:
            new_list.append((activity, price))
            num_restaurants -= 1
        elif num_other_activities > 0:
            new_list.append((activity, price))
            num_other_activities -= 1

    return new_list



def run_genetic_algorithm(preferences,museums_df,zoos_df,restaurants_df,parks_df, past_activities):
    # Updated sample data representing museums, parks, and restaurants
    museums = museums_df.to_dict("records")
    zoos = zoos_df.to_dict("records")
    restaurants = restaurants_df.to_dict("records")
    parks = parks_df.to_dict("records")


    day_plan = genetic_algorithm(preferences,museums,zoos,restaurants,parks,past_activities)

    day_plan_titles=[]
    for activity in day_plan:
        type=check_activity_type(activity['name'],museums_df,zoos_df,restaurants_df,parks_df)
        full_type_activity= type + activity['name']
        full_price = " ,Price:"+ str(activity['ticket_price'])
        day_plan_titles.append((full_type_activity,full_price))

    final_plan = create_final_plan(day_plan_titles,preferences['num_restaurants'], preferences['num_activities'])
    return final_plan


# Function to read feedback from CSV
def read_feedback_from_csv(day):
    filename = f'feedback_day_{day}.csv'
    feedback_data = {}
    if os.path.exists(filename):
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for activity, rating in row.items():
                    if activity.startswith("feedback_"):  # Check if the column represents feedback
                        feedback_activity = activity.split("_", 2)[2].split(" ,")[0]  # Extract the activity name
                        feedback_data[feedback_activity] = int(rating)  # Store feedback rating
    return feedback_data

def filter_museums_base_on_feedback(user_preferences,museums_df,current_day):
    if user_preferences['museums']:
        liked_museum_types = set()
        disliked_museum_types = set()
        visited_museums = set()
        for i in range(1,current_day+1):
            feedback_data = read_feedback_from_csv(i)
            museum_feedback = {activity: rating for activity, rating in feedback_data.items() if activity.startswith("Museum:")}
            museum_feedback_cleaned = {museum.split(":", 1)[1]: rating for museum, rating in museum_feedback.items()}

            if len(museum_feedback) > 0:
                for museum, rating in museum_feedback_cleaned.items():
                    visited_museums.add(museum)
                    if rating > 3:
                        liked_museum_types.add(museums_df[museums_df['name'] == museum]['Museum Type'].iloc[0])
                    else:
                        disliked_museum_types.add(museums_df[museums_df['name'] == museum]['Museum Type'].iloc[0])

            # Filter museums_df to exclude museums with types present in disliked_museum_types
        filtered_museums_df = museums_df[~museums_df['Museum Type'].isin(disliked_museum_types)]
        filtered_museums_df = filtered_museums_df[~filtered_museums_df['name'].isin(visited_museums)]
        return filtered_museums_df

    else:
        return museums_df

def read_csv_to_list(day):
    filename = f'trip_day_{day}.csv'
    result = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for item in row:
                result.append(ast.literal_eval(item))
    return result


def check_in_dict(dict_feedback,name):
    for item in dict_feedback:
        if name in item:
            return dict_feedback[item]

    return 0


def insert_data(lst, user_ID,dict_feedback,museums_df,restaurants_df,parks_df,zoos_df):
    first_values = [t[0] for t in lst]
    museums = []
    restaurants = []
    parks = []
    zoos = []

    # Iterate over each item in the output list
    for item in first_values:
        # Split the item into its type and name
        category, name = item.split(":", 1)
        feedback_val = check_in_dict(dict_feedback,name)
        # Append the item to the corresponding category list
        if category == 'Museum':
            museums.append((name,feedback_val))
        elif category == 'Restaurant':
            restaurants.append((name,feedback_val))
        elif category == 'Park':
            parks.append((name,feedback_val))
        elif category == 'Zoo':
            zoos.append((name,feedback_val))

    museums_new_values = museums_df[museums_df['name'].str.strip().isin([activity[0].strip() for activity in museums])]
    museums_corresponding_values = museums_new_values[['name', 'Museum Type', 'longitude', 'latitude', 'ticket_price']]
    museums_corresponding_values.insert(0,'ID',user_ID)
    feedback = [t[1] for t in museums]
    museums_corresponding_values.insert(6,'feedback',feedback)

    restaurants_new_values = restaurants_df[restaurants_df['name'].str.strip().isin([activity[0].strip() for activity in restaurants])]
    restaurants_corresponding_values = restaurants_new_values[['name', 'cuisine_description','is_vegan', 'is_vegetarian','longitude', 'latitude', 'ticket_price']]
    restaurants_corresponding_values.insert(0,'ID',user_ID)
    feedback = [t[1] for t in restaurants]
    restaurants_corresponding_values.insert(8,'feedback',feedback)

    parks_new_values = parks_df[parks_df['name'].str.strip().isin([activity[0].strip() for activity in parks])]
    parks_corresponding_values = parks_new_values[['name','latitude', 'longitude', 'ticket_price']]
    parks_corresponding_values.insert(0, 'ID', user_ID)
    feedback = [t[1] for t in parks]
    parks_corresponding_values.insert(2,'feedback',feedback)

    zoos_new_values = zoos_df[zoos_df['name'].str.strip().isin([activity[0].strip() for activity in zoos])]
    zoos_corresponding_values = zoos_new_values[['name','latitude', 'longitude', 'ticket_price']]
    zoos_corresponding_values.insert(0, 'ID', user_ID)
    feedback = [t[1] for t in zoos]
    zoos_corresponding_values.insert(2,'feedback',feedback)

    museums_feedback= pd.read_csv("users_data/muesums_feedback.csv")
    parks_feedback= pd.read_csv("users_data/parks_feedback.csv")
    zoos_feedback = pd.read_csv("users_data/zoos_feedbacks.csv")
    restaurants_feedback= pd.read_csv("users_data/restaurant_feedback.csv")


    if not restaurants_new_values.empty:
        concatenated_rest = pd.concat([restaurants_feedback, restaurants_corresponding_values], ignore_index=True)
        concatenated_rest.to_csv("users_data/restaurant_feedback.csv",index=False)

    if not museums_corresponding_values.empty:
        concatenated_musemus = pd.concat([museums_feedback, museums_corresponding_values], ignore_index=True)
        concatenated_musemus.to_csv('users_data/muesums_feedback.csv',index=False)


    if not parks_corresponding_values.empty:
        concatenated_parks = pd.concat([parks_feedback, parks_corresponding_values], ignore_index=True)
        concatenated_parks.to_csv("users_data/parks_feedback.csv",index=False)

    if not zoos_corresponding_values.empty:
        concatenated_zoos = pd.concat([zoos_feedback, zoos_corresponding_values], ignore_index=True)
        concatenated_zoos.to_csv("users_data/zoos_feedbacks.csv",index=False)

nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

def classify_sentiment(feedback):
    # Initialize NLTK sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    # Define keywords related to museums, restaurants, zoos, and parks
    museum_keywords = ['museum', 'museums', 'gallery', 'exhibit']
    restaurant_keywords = ['restaurants', 'restaurant', 'dining', 'eatery']
    zoo_keywords = ['zoo', 'aquarium', 'safari']
    park_keywords = ['parks', 'park', 'garden', 'nature']

    # Analyze sentiment using NLTK
    sentiment_score = analyzer.polarity_scores(feedback)

    # Check if any museum, restaurant, zoo, or park keywords are present in the feedback
    museum_related = any(keyword in feedback.lower() for keyword in museum_keywords)
    restaurant_related = any(keyword in feedback.lower() for keyword in restaurant_keywords)
    zoo_related = any(keyword in feedback.lower() for keyword in zoo_keywords)
    park_related = any(keyword in feedback.lower() for keyword in park_keywords)

    # Classify sentiment based on compound score
    if sentiment_score['compound'] >= 0.05:
        return {'museum': 'positive' if museum_related else 'natural',
                'restaurant': 'positive' if restaurant_related else 'natural',
                'zoo': 'positive' if zoo_related else 'natural',
                'park': 'positive' if park_related else 'natural',
                'general':'positive'}
    elif sentiment_score['compound'] <= -0.05:
        return {'museum': 'negative' if museum_related else 'natural',
                'restaurant': 'negative' if restaurant_related else 'natural',
                'zoo': 'negative' if zoo_related else 'natural',
                'park': 'negative' if park_related else 'natural',
                'general':'negative'}
    else:
        return {'museum': 'positive' if museum_related else 'natural',
                'restaurant': 'positive' if restaurant_related else 'natural',
                'zoo': 'positive' if zoo_related else 'natural',
                'park': 'positive' if park_related else 'natural',
                'general':'positive'}

def save_user_reviews(feedback_file_name):
    dataset = pd.read_csv(feedback_file_name)

    # Extract activity and review columns dynamically
    activity_columns = [col for col in dataset.columns if col.startswith("review_")]
    activities = [col.split("_")[-1] for col in activity_columns]
    reviews = dataset[activity_columns].iloc[0].tolist()

    # Create a list of dictionaries for the new data
    new_data = [{"activity": act, "review": rev} for act, rev in zip(activities, reviews)]
    filename = "users_data/user_reviews.csv"
    fieldnames = ["activity", "review"]

    # Append data to CSV file using csv.DictWriter
    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()
            # Write new data
            for data in new_data:
                writer.writerow(data)
        print("Data appended successfully.")
    except Exception as e:
        print(f"Error occurred: {e}")
    df = pd.read_csv(filename)
    df.dropna(subset=["review"],inplace=True)
    df.drop_duplicates(subset=["activity", "review"], inplace=True)
    df.to_csv(filename, index=False)
def save_user_feedback_to_csv(user_id, user_feedback):
    filename = 'users_data/user_feedback.csv'
    fieldnames = ['ID', 'Textual Feedback', 'Museum Feedback', 'Restaurant Feedback', 'Zoo Feedback', 'Park Feedback','sentimental analysis']

    # Check if the file exists
    file_exists = os.path.isfile(filename)

    # Classify sentiment of the feedback
    sentiment_results = classify_sentiment(user_feedback)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is newly created
        if not file_exists:
            writer.writeheader()

        # Write user feedback and classification results
        writer.writerow({
            'ID': user_id,
            'Textual Feedback': user_feedback,
            'Museum Feedback': sentiment_results['museum'],
            'Restaurant Feedback': sentiment_results['restaurant'],
            'Zoo Feedback': sentiment_results['zoo'],
            'Park Feedback': sentiment_results['park'],
            'sentimental analysis': sentiment_results['general']
        })

def delete_old_files(num_days):
    for i in range(1, num_days+1):
        trip_file_name = f"trip_day_{i}.csv"
        feedback_file_name = f"feedback_day_{i}.csv"

        if os.path.exists(trip_file_name):
            os.remove(trip_file_name)
            print(f"Deleted file: {trip_file_name}")
        if os.path.exists(feedback_file_name):
            os.remove(feedback_file_name)
            print(f"Deleted file: {feedback_file_name}")

def sum_prices(num_files):
    """
    Calculates the total price from multiple CSV files with restaurant data.
    """

    total_price = 0
    for i in range(1, num_files):
        filename = f'trip_day_{i}.csv'

        try:
            with open(filename, 'r') as file:
                data = file.readlines()

            for entry in data:
                try:
                    # Split on commas, handle leading/trailing spaces
                    parts = entry.strip().split(",")
                    restaurant = parts[0]  # Assuming restaurant name is the first part
                    price_str = parts[2].split(":")[-1].strip()[:-1].split("'")[0]
                    price = float(price_str)
                    total_price += price
                except (IndexError, ValueError):
                    print(f"Error processing entry: {entry.strip()} in file: {filename}")
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")

    return round(total_price,2)


def remove_prefix(input_string):
    colon_index = input_string.find(':')
    if colon_index != -1:
        return input_string[colon_index + 1:].strip()
    else:
        return input_string.strip()


def save_dict_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = data.keys() if data else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(data)

def read_csv_to_dict(filename):
    data_dict = {}
    with open(filename, 'r') as file:
        # Create a CSV reader object
        reader = csv.DictReader(file)

        # Iterate over each row in the CSV file (assuming only one row)
        for row in reader:
            # Convert string boolean values to actual booleans
            for key, value in row.items():
                if value.lower() == 'true':
                    row[key] = True
                elif value.lower() == 'false':
                    row[key] = False

            # Convert budget, walking_distance, num_activities, and num_restaurants to integers
            row['budget'] = int(row['budget'])
            row['walking_distance'] = int(row['walking_distance'])
            row['num_activities'] = int(row['num_activities'])
            row['num_restaurants'] = int(row['num_restaurants'])

            # Parse cuisine_description list using ast.literal_eval
            row['cuisine_description'] = ast.literal_eval(row['cuisine_description'])

            # Store the row data in the data_dict
            data_dict = dict(row)

    return data_dict

def train_knn_model(feedback_data):
    feedback_data.fillna(int(round(feedback_data['feedback'].values.mean(),0)), inplace=True)
    X = feedback_data[['ID', 'longitude', 'latitude', 'ticket_price']].values  # Features
    y = feedback_data['feedback'].values  # Target

    knn_model = KNeighborsClassifier(n_neighbors=5)  # Initialize KNN model
    knn_model.fit(X, y)  # Train the model

    return knn_model


def generate_recommendations(user_id, user_preferences, feedback_data, knn_model, remaining_budget,
                             num_recommendations):
    recommendations = []
    # Prepare features for prediction
    X_pred = pd.DataFrame({
        'ID': [user_id] * len(feedback_data),  # Repeat user ID to match the number of rows in feedback_data
        'ticket_price': [user_preferences['budget']] * len(feedback_data),  # Use user's budget for all rows
        'longitude': [0] * len(feedback_data),  # Placeholder for longitude
        'latitude': [0] * len(feedback_data),  # Placeholder for latitude
    })

    # Encode categorical variables if present
    if 'Museum Type' in feedback_data.columns:
        feedback_data['Museum Type'] = LabelEncoder().fit_transform(feedback_data['Museum Type'])

    # Generate predictions using the trained KNN model
    predicted_feedback_proba = knn_model.predict_proba(X_pred)[:, 1]  # Probability of positive class

    # Combine predictions with activity names
    for index, row in feedback_data.iterrows():
        if row['ticket_price'] <= float(remaining_budget):
            recommendations.append({
                'name': row['name'],
                'predicted_feedback': predicted_feedback_proba[index],# Probability of user enjoying the activity
                'ticket_price':row['ticket_price']
            })

    # Sort recommendations by predicted feedback (descending order)
    recommendations.sort(key=lambda x: x['predicted_feedback'], reverse=True)

    # Limit the number of recommendations based on num_recommendations
    recommendations = recommendations[:num_recommendations]

    return recommendations

def activity_type_price(activity, museums, zoos, restaurants, parks):
    activity = activity.strip()
    if activity in museums['name'].str.strip().values:
        return "Museum:", museums.loc[museums['name'].str.strip() == activity, 'ticket_price'].iloc[0]
    elif activity in zoos['name'].str.strip().values:
        return "Zoo:", zoos.loc[zoos['name'].str.strip() == activity, 'ticket_price'].iloc[0]
    elif activity in restaurants['name'].str.strip().values:
        return "Restaurant:", restaurants.loc[restaurants['name'].str.strip() == activity, 'ticket_price'].iloc[0]
    else:
        return "Park:", parks.loc[parks['name'].str.strip() == activity, 'ticket_price'].iloc[0]

def add_ticket_price(dictionary,df):
    name = dictionary['name']
    ticket_price = df[df['name'] == name]['ticket_price'].values[0]
    dictionary['ticket_price'] = ticket_price
    return dictionary

def additional_activities(user_history,df,restaurants= False,preferred_cuisines=[]):
    filtered_df= df[~df['name'].isin(user_history)]
    if len(filtered_df)==0:
        filtered_df=df


    if restaurants:
        filtered_df = filtered_df[filtered_df['cuisine_description'].isin(preferred_cuisines)]
        if len(filtered_df)==0:
            filtered_df = df[~df['name'].isin(user_history)]

    filtered_df.reset_index(drop=True, inplace=True)
    activity = random.choice(filtered_df['name'])
    ticket_price= filtered_df.loc[filtered_df['name']==activity,'ticket_price']
    ticket_price = ticket_price.reset_index(drop=True)
    dict_activity = {'name': activity, 'ticket_price':ticket_price.iloc[0]}

    return dict_activity



def check_cuisine_pref(current_rest,rest_df,cuisine):
    new_rest=[]
    for rest in current_rest:
        rest_name = rest['name']
        val = rest_df.loc[rest_df['name'] == rest_name, 'cuisine_description']
        if val.iloc[0] in cuisine:
            new_rest.append(rest)

    return new_rest

def count_activities_per_day(history):
    counts = {'Museum': 0, 'Park': 0, 'Restaurant': 0, 'Zoo': 0}
    for item in history:
        for entity in counts.keys():
            if entity.lower() in str(item[0]).lower():
                counts[entity] += 1
    return counts

def adjust_day_plan(current_day_plan, prev_day, activity,user_history,df):
    activity_count=0
    for option in current_day_plan:

        if activity.lower() in str(option[0]).lower():
            activity_count+=1


    # Check if 'more_parks' is True and parks count in current day plan <= parks count in previous day
    if activity_count <= prev_day[activity]:
        # Remove activity that are not restaurants
        non_restaurant_activities = [(activity_type, price) for activity_type, price in current_day_plan if
                                     'Restaurant' not in activity_type and activity not in activity_type]

        if non_restaurant_activities:
            current_day_plan.remove(non_restaurant_activities[0])

            new_activity=additional_activities(user_history,df)
            price = new_activity['ticket_price']
            full_type_activity = activity+":" + new_activity['name']
            full_price = " ,Price:" + str(price)
            current_day_plan.append((full_type_activity, full_price))


    return current_day_plan

def adjusted_preferences_KNN(user_preferences, user_id, museums_df,zoos_df,restaurants_df,parks_df,current_day):
    museums_feedback = pd.read_csv("users_data/muesums_feedback.csv")
    parks_feedback  = pd.read_csv("users_data/parks_feedback.csv")
    restaurant_feedback  = pd.read_csv("users_data/restaurant_feedback.csv")
    zoos_feedback  = pd.read_csv("users_data/zoos_feedbacks.csv")
    users_data = pd.read_csv("users_data/user_feedback.csv")

    lst_user_history_extended=[]
    lst_user_history = []
    prev_day_history=[]
    for i in range(current_day-1):
        lst_user_history_extended.extend(read_csv_to_list(i+1))
        if i == current_day-2:
            prev_day_history.append(read_csv_to_list(i+1))

    activity_counts_prev_day= count_activities_per_day(prev_day_history[0])
    for i in range(len(lst_user_history_extended)):
        lst_user_history.append(remove_prefix(lst_user_history_extended[i][0]))


    museums_feedback['Museum Type'] = LabelEncoder().fit_transform(museums_feedback['Museum Type'])
    knn_models = {}
    flag_museums= False
    flag_zoos=False
    flag_parks=False
    if user_preferences['museums']:
        if 'more_museums' in user_preferences and user_preferences['more_museums']:
            flag_museums = True
        knn_models['museums'] = train_knn_model(museums_feedback)
    if user_preferences['parks']:
        if 'more_parks' in user_preferences and user_preferences['more_parks']:
            flag_parks = True
        knn_models['parks'] = train_knn_model(parks_feedback)
    if user_preferences['restaurants']:
        knn_models['restaurants'] = train_knn_model(restaurant_feedback)
    if user_preferences['zoos']:
        if 'more_zoos' in user_preferences and user_preferences['more_zoos']:
            flag_zoos = True
        knn_models['zoos'] = train_knn_model(zoos_feedback)

    all_recommendations = []
    remaining_budget = user_preferences['budget']
    total_recommendations = user_preferences['num_activities']

    restaurant_recommendations = []

    for activity_type, knn_model in knn_models.items():
        feedback_data = None
        if activity_type == 'museums':
            feedback_data = museums_feedback
        elif activity_type == 'parks':
            feedback_data = parks_feedback
        elif activity_type == 'restaurants':
            feedback_data = restaurant_feedback
        elif activity_type == 'zoos':
            feedback_data = zoos_feedback

        if feedback_data is not None:
            num_recommendations = total_recommendations
            if activity_type == 'restaurants':
                restaurant_recommendations = generate_recommendations(user_id, user_preferences, feedback_data,
                                                                      knn_model, remaining_budget, num_recommendations)
            else:
                all_recommendations.extend(generate_recommendations(user_id, user_preferences, feedback_data, knn_model, remaining_budget, num_recommendations))
        else:
            print(f"No feedback data available for {activity_type}")

    #restaurant_recommendations = [add_ticket_price(dictionary,restaurants_df) for dictionary in restaurant_recommendations]
    # Combine all recommendations
    random.shuffle(all_recommendations)
    selected_recommendations=[]
    selected_restaurants= []
    selected_activity_names = set()


    for recommendation in restaurant_recommendations:
        if recommendation['name'].strip() not in lst_user_history:
            if recommendation['ticket_price'] <= float(remaining_budget):
                if recommendation['name'].strip() not in selected_activity_names:
                    selected_restaurants.append(recommendation)
                    remaining_budget -= recommendation['ticket_price']
                    lst_user_history.append(recommendation['name'])
                    selected_activity_names.add(recommendation['name'])

                    if len(selected_restaurants) == user_preferences['num_restaurants']:
                        break

    selected_restaurants= check_cuisine_pref(selected_restaurants,restaurants_df,user_preferences['cuisine_description'])
    if len(selected_restaurants) < int(user_preferences['num_restaurants']):
        rest_left= user_preferences['num_restaurants']-len(selected_restaurants)
        for i in range(rest_left):
            new_rest = additional_activities(lst_user_history,restaurants_df,True,user_preferences['cuisine_description'])
            selected_restaurants.append(new_rest)
            lst_user_history.append(new_rest)

    for recommendation in all_recommendations:
        if recommendation['name'] not in lst_user_history:
            if recommendation['ticket_price'] <= float(remaining_budget):
                if recommendation['name'] not in selected_activity_names:
                    selected_recommendations.append(recommendation)
                    remaining_budget -= recommendation['ticket_price']
                    lst_user_history.append(recommendation['name'])
                    selected_activity_names.add(recommendation['name'])


                    if len(selected_recommendations) == total_recommendations :
                        break

    if len(selected_recommendations) < total_recommendations:
        activities_left = total_recommendations - len(selected_recommendations)
        for i in range(activities_left):
            if user_preferences['museums']:
                new_activity = additional_activities(lst_user_history, museums_df)
                selected_recommendations.append(new_activity)
                lst_user_history.append(new_activity)


                if len(selected_recommendations) == total_recommendations:
                    break

            if user_preferences['parks']:
                new_activity = additional_activities(lst_user_history, parks_df)
                selected_recommendations.append(new_activity)
                lst_user_history.append(new_activity)

                if len(selected_recommendations) == total_recommendations:
                    break

            if user_preferences['zoos']:
                new_activity = additional_activities(lst_user_history, zoos_df)
                selected_recommendations.append(new_activity)
                lst_user_history.append(new_activity)

                if len(selected_recommendations) == total_recommendations:
                    break

    selected_recommendations.extend(selected_restaurants)
    day_plan_titles=[]
    for activity in selected_recommendations:
        type ,price = activity_type_price(activity['name'],museums_df,zoos_df,restaurants_df,parks_df)
        full_type_activity = type + activity['name']
        full_price = " ,Price:"+ str(price)
        day_plan_titles.append((full_type_activity,full_price))


    if flag_museums:
        day_plan_titles=adjust_day_plan(day_plan_titles,activity_counts_prev_day,'Museum',
                        lst_user_history_extended,museums_df)
    if flag_parks:
        day_plan_titles=adjust_day_plan(day_plan_titles,activity_counts_prev_day,'Park',
                        lst_user_history_extended,parks_df)

    if flag_zoos:
        day_plan_titles=adjust_day_plan(day_plan_titles,activity_counts_prev_day,'Zoo',
                        lst_user_history_extended,zoos_df)


    return day_plan_titles

