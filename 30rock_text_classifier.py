import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd



# ===========================================================================================================  LOAD DATA
@st.cache_data
def get_data(filename):
    episodes_data = pd.read_csv(filename)
    return episodes_data

#Change data file path here
episodes_data = get_data('data/df_30_rock.csv')



# ==============================================================================================================  HEADER
st.title('Visualizing 30 Rock episode Data')
st.caption("Machine Learning Web App used to visualize data and predict who would have said your quote")
st.write('---')
st.subheader("What's Your Favorite Episode")



# =============================================================================================================  USER INPUT
#Lists for the select boxes
season_list = list(range(1, 8))
episode_list = list(range(1, 24))

#Creates two columns with two selectboxes
col1, col2 = st.columns(2)
with col1:
    seaon_selection = st.selectbox('Season', season_list, index=None, placeholder="Select the Season")
with col2:
    episode_selection = st.selectbox('Episode', episode_list, index=None, placeholder="Select the Episode")

#Gets data from the select boxes and stores it
selected_episode = episodes_data[(episodes_data['season'] == seaon_selection) & (episodes_data['episode_num_in_season'] == episode_selection)]

#If loops to get the rating and episode number
if not selected_episode.empty:
    user_y  = selected_episode['imdb_rating'].iloc[0]
    user_x = selected_episode['episode_num_overall'].iloc[0]
    #st.write(f'Episode Rating: {user_y}')
    #st.write(f'Overall Episode Number: {user_x}')
#else:
    #st.write("No data available for the selected episode.")


add_point = st.button("View Episode", use_container_width=True)



# =============================================================================================================  SCATTERPLOT
if add_point:
    new_data = {'x': user_x, 'y': user_y}
    df = pd.concat([episodes_data, pd.DataFrame([new_data])], ignore_index=True)
    #df = episodes_data.append(new_data, ignore_index=True)

    #Re-plot the scatterplot with the new point
    st.vega_lite_chart(
        df, 
        {
            "mark": "point",
            "encoding": 
            {
                "x": {"field": "x", "type": "quantitative", "scale":{"domain": [1,137]}, "title": "Episodes"},
                "y": {"field": "y", "type": "quantitative", "scale":{"domain": [7.2,9.2]}, "title": "IMDB Rating"},
                "color": {"field": "category", "type": "nominal", "title": "My Episode"},
                "size": {"value": 250}
            },
        },
        use_container_width=False,
        width=800,
        height=600
    )

else:
    st.vega_lite_chart(
        episodes_data,
        {
            "mark": {"type": "circle", "tooltip": True, "opacity": 0.75, "stroke": "black", "strokeWidth": 1},
            "encoding": 
            {
                "x": {"field": "episode_num_overall", "type": "quantitative", "scale":{"domain": [1,137]}, "title": "Episodes"},
                "y": {"field": "imdb_rating", "type": "quantitative", "scale":{"domain": [7.2,9.2]}, "title": "IMDB Rating"},
                "size": {"field": "us_viewers", "type": "quantitative", "scale": {"type": "threshold","domain": [5000000, 6000000], "range": [80, 200, 320]}, "legend":{"title": "# of Viewers"}},
                "color": {"field": "season", "legend":{"title": "Season"}},
                "shape": {"field": "season", "type": "quantitative"},
                "legend": {"title": "Title"}
                #"size": {"field": "us_viewers", "type": "quantitative", "scale": {"rangeMax": 1000}},
                #"size": {"field": "us_viewers", "type": "quantitative", "scale":{"domain": [3000000,9000000]}},
                #"size": {"field": "us_viewers", "type": "quantitative", "scale": {"type": "quantize"}},
                #"color": {"field": "season", "scale": {"type": "threshold","domain": [5000000, 6000000], "scheme": "viridis"}},
            },
        },
        use_container_width=False,
        width=800,
        height=600
    )  



# ====================================================================================================  DATAFRAME
filtered_df = episodes_data[(episodes_data['season'] == seaon_selection) & (episodes_data['episode_num_in_season'] == episode_selection)]

if not filtered_df.empty:
    user_favorite_director  = filtered_df['directed_by'].iloc[0]
    user_favorite_writer = filtered_df['written_by'].iloc[0]
    with st.expander("Episodes Suggestion"):
        st.write(f'Your Favorite Director is: {user_favorite_director}')
        st.write(f'Your Favorite Writer is: {user_favorite_writer}')
        #st.subheader("Episode Suggestions")
        new_df = episodes_data[(episodes_data['directed_by'] == user_favorite_director) & (episodes_data['written_by'] == user_favorite_writer)]
        st.dataframe(
            new_df, 
            hide_index=True, 
            column_order=("season", "episode_num_in_season", "title", "directed_by", "written_by", "imdb_rating"), 
            column_config={'season':'Season', 'episode_num_in_season':'Episode', 'title':'Title', 'directed_by':'Director', 'written_by':'Writer', 'imdb_rating':'IMDB Rating'}, 
            use_container_width=True)
        #st.dataframe(episodes_data, columns=("season", "episode_num_in_season", "title", "directed_by", "written_by") )
else:
    with st.expander("Similar Episodes Suggestion"):
        st.subheader("Episode Suggestions")
        st.dataframe(episodes_data, 
                     hide_index=True, 
                     column_order=("season", "episode_num_in_season", "title", "directed_by", "written_by", "imdb_rating"), 
                     column_config={'season':'Season', 'episode_num_in_season':'Episode', 'title':'Title', 'directed_by':'Director', 'written_by':'Writer', 'imdb_rating':'IMDB Rating'}, 
                     use_container_width=True)
        #st.dataframe(episodes_data, columns=("season", "episode_num_in_season", "title", "directed_by", "written_by") )



# ====================================================================================================  MACHINE LEARNING
# Sample data (quotes from the characters)
data = {
    "Liz Lemon": [
        "I believe that all anyone really wants in this life is to sit in peace and eat a sandwich.",
        "I want to go to there.",
        "Yes to love, yes to life, yes to staying in more!",
        "My mom used to send me articles about how older virgins are considered good luck in Mexico.",
        "Can I share with you my worldview? All of humankind has one thing in common: the sandwich. I believe that all anyone really wants in this life is to sit in peace and eat a sandwich.",
        "Why are my arms so weak? It's like I did that push-up last year for nothing!",
        "I already have a drink. Do you think he'd buy me mozzarella sticks?",
        "There ain’t no party like a Liz Lemon party 'cause a Liz Lemon party is MANDATORY!",
        "I’m feeling pretty drunk. Well, it's business drunk. It's like rich drunk. Either way, it's legal to drive.",
        "All of my summer dresses are getting weird.",
        "Jack, I spent two days making this. My wedding is my day! I get what I want!",
        "Who hasn't made mistakes? I once French-kissed a dog at a party to try to impress what turned out to be a very tall 12-year-old.",
        "I don't have any friends, I only have deal breakers.",
        "I wolfed my Teamster sub for you!",
        "I took one of those 'Which Gossip Girl are you?' quizzes, and it said I was the dad’s guitar.",
        "High-fiving a million angels!",
        "It’s not a Lemon party without old Dick!",
        "I'm not a creative type like you, with your work sneakers and left-handedness.",
        "I did Big Sister in college. That little girl taught me how to use tampons.",
        "I'm gonna win so hard for you, Hazel. So hard!",
        "You are my heroine! And by heroine, I mean lady hero. I don’t want to inject you and listen to jazz.",
        "Blammo! Another successful interaction with a man!",
        "The only things I like are dogs that fit into purses and weird cheeses.",
        "Ugh, I hate January. It’s dark and freezing and everyone’s wearing bulky coats, so you can do some serious subway flirting before you realize the guy is homeless.",
        "I believe that vampires are the world's greatest golfers but their curse is they never get to prove it.",
        "I don’t think it’s fair that women have an excuse once a month to act like complete idiots.",
        "Suck it, monkeys! I'm going corporate!",
        "I’m 37, please don’t make me go to Brooklyn.",
        "I will not calm down! Women are allowed to get angrier than men about double standards.",
        "Foot cycle to the face!"
    ],
    "Jack Donaghy": [
        "Lemon, we’d all like to flee to the Cleve and club-hop down at the Flats and have lunch with Little Richard, but we fight those urges.",
        "Every time I meet a new person, I figure out how I'm going to fight them.",
        "Ambition is the willingness to kill the things you love and eat them to stay alive. Haven’t you ever read my throw pillow?",
        "Money can't buy happiness. It is happiness.",
        "We all have ways of coping. I use sex and awesomeness.",
        "I only pass gas once a year, for an hour, atop a mountain in Switzerland.",
        "Your hair is your head-suit.",
        "Good God, Lemon, those jeans make you look like a Mexican sports reporter.",
        "The world is made by those who control their own destiny. It doesn't just happen, it's planned. The secret is preparation.",
        "The Italians have a saying, Lemon. 'Keep your friends close and your enemies closer.' And although they've never won a war or mass-produced a decent car, in this area they are correct.",
        "There are no bad ideas in brainstorming, Lemon, just like there are no statues of committees.",
        "If you want to succeed in business, you have to do one thing: Avoid shortcuts.",
        "Never go with a hippie to a second location.",
        "I am an extraordinary man, surrounded by ordinary people.",
        "Reaganing: The perfection of solving something from beginning to end without making a single mistake.",
        "I believe that when you have a problem, you talk it over with your priest, or your tailor, or the mute elevator porter at your men’s club, and you take that problem and crush it with your mind vice.",
        "You can't solve a problem with the people who created it.",
        "Lemon, having a pet is a responsibility. It's like having a baby that's also a grandma.",
        "You want to be mindful of your Ks. Avoid saying words with too many K sounds. It shows a lack of creativity.",
        "I like you. You have the boldness of a much younger woman.",
        "New York, third-wave feminist, college-educated, single and pretending to be happy about it, overscheduled, undersexed, you buy any magazine that says 'healthy body image' on the cover and every two years you take up knitting for... a week.",
        "The grown-up dating world is like your haircut. Sometimes, awkward triangles occur.",
        "The Irish have an expression, Lemon: 'May you die in bed at 95, shot by a jealous spouse.'",
        "Lemon, life is about minimizing regrets.",
        "Lemon, never follow a hippie to a second location. It's just common sense.",
        "What am I, a farmer?",
        "I do not get cute. I get drop-dead gorgeous.",
        "It's time for you to start dating up, and by 'up,' I mean in the corporate hierarchy.",
        "Business doesn't get me down, business gets me off.",
        "In five years, we'll all either be working for him... or dead by his hand."
    ],
    "Tracy Jordan": [
        "I am a Jedi! I am a Jedi!",
        "I love this cornbread so much, I want to take it behind a middle school and get it pregnant.",
        "Live every week like it's Shark Week.",
        "I'm whipped! Angie got me up at 7:30 today. Did you know in the morning they got food, TV, almost everything? It’s pretty good.",
        "You don’t have to thank me, Liz Lemon. We’re a team now. Like Batman and Robin. Like chicken, and a chicken container.",
        "Stop eating people's old French fries, pigeon. Have some self respect! Don't you know you can fly?",
        "Werewolf Bar Mitzvah, spooky scary! Boys becoming men, men becoming wolves!",
        "Dress every day like you're gonna get murdered in those clothes.",
        "I'm gonna make you a mixtape. You like Phil Collins?",
        "I watched Boston Legal nine times before I realized it wasn't a new Star Trek.",
        "Affirmative action was designed to keep women and minorities in competition with each other to distract us while white dudes inject AIDS into our chicken nuggets.",
        "I once saw a baby give another baby a tattoo! They were very drunk.",
        "Tell her you want your privates and your heart to snuggle.",
        "It’s after six. What am I, a farmer?",
        "The secret to a great marriage is separate bathrooms. After you've had a double soup bowel movement, you can walk out and look your spouse in the eye.",
        "I’m gonna make you a mixtape. You like Phil Collins? I got two ears and a heart, don’t I?",
        "I’m gonna eat you, foot!",
        "If you could combine any two animals, it would definitely be elephant and rhino. Think about it, Elephino.",
        "I believe that the moon does not exist. I believe that vampires are the world’s greatest golfers but their curse is that they’ll never get to prove it.",
        "Here’s some advice I wish I woulda got when I was your age: Live every week like it’s Shark Week.",
        "Who’s crazier, me or Ann Curry?",
        "I love cornbread so much, I want to take it behind the middle school and get it pregnant.",
        "Liz Lemon, you’re my heroine! And by heroine, I mean lady hero. I don’t want to inject you and listen to jazz.",
        "This is untoward! This is not toward!",
        "I once farted in the set of Blue Man Group.",
        "I'm like Pac-Man. I'm just like him. I love eatin' and runnin' and ghosts.",
        "Mankind has been using hallucinogens for thousands of years, and I am going to continue that proud tradition.",
        "In the future, computers will be twice as powerful, ten thousand times larger, and so expensive that only the five richest kings of Europe will own them.",
        "I've never met an idiot. Everyone has something to teach you if you're humble enough to learn.",
        "I'm an adult and I deserve an adult glass!"
    ],
    "Jenna Maroney": [
        "Listen up, fives, a ten is speaking!",
        "I'm not afraid of anyone in show business. I turned down intercourse with Harvey Weinstein on no less than three occasions... out of five.",
        "It's okay, you can say it. I'm superficial.",
        "The future is like a Japanese game show. You have no idea what's going on.",
        "I believe that vampires are the world’s greatest golfers, but their curse is they’ll never get to prove it.",
        "I don't really think about the ratings. That's just not who I am. Mostly because I don't understand how they work.",
        "Goodbye forever, you factory reject dildos!",
        "You can't put a price on the joy of not talking to people.",
        "My whole life is thunder!",
        "I only want you to star in my movie if you're damaged, because damaged people are dangerous because they know they can survive.",
        "Another successful interaction with a man!",
        "I've got a new life philosophy that I call 'Jenna's Me' Time'. I dictate my own life. I take what I want, when I want it, because I can.",
        "I'll do it, but only for the attention.",
        "I’ve been through so much, I’m basically a war hero. And we all know how America treats their war heroes.",
        "Oh, poor baby. Can't hack it in the big city? Gonna move to the Bay Area now and pretend that that was your dream the whole time? Have fun always carrying a light sweater.",
        "Jealousy is a powerful emotion. It's right up there with lust and respect.",
        "There are no bad ideas, Lemon, only great ideas that go horribly wrong.",
        "Am I trying to instigate fights by throwing wine at people just to get on camera, and maybe also promote my new lifestyle website, Jenna’s Side? Of course not.",
        "Me? Rural jury!",
        "They'll never cancel the Olympics. They're a billion-dollar industry!",
        "Why are you all acting like this is a big deal? I’ve been drinking red wine my entire life!",
        "Can I share with you my worldview? All of humankind has one thing in common: the sandwich. I believe that all anyone really wants in this life is to sit in peace and eat a sandwich.",
        "I already have a drink. Do you think he'd buy me mozzarella sticks?",
        "When I was a kid growing up in the Tampa area, we didn’t have a lot. But we had dreams. Dreams of getting out of there and doing a sky burial.",
        "It's not about money. It’s about starring.",
        "Don’t talk to me like I'm your wife!",
        "I want people to be afraid of how much they love me, and by 'people', I mean America.",
        "I’m going to turn this show into a show for women, by women, with jokes only women will get. Like: 'You’re in a shoe store. Two pair of shoes speak to you, but you can only buy one pair. Which do you choose?'",
        "You can’t have a Lemon party without old Dick!",
        "Do you know the number of men who have ruined my life? All of them! My dad, Conan O'Brien, Phil Spector, to name just a few."

    ]
}

# Flatten the data and separate labels and features
labels, texts = [], []
for character, quotes in data.items():
    labels.extend([character] * len(quotes))
    texts.extend(quotes)

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X, labels)

# Streamlit app
def main():
    st.write('---')
    st.title("30 Rock Character Speech Matcher")

    # User input
    user_input = st.text_area("Enter a quote to see which 30 Rock character might have said it:", value="", height=150)

    # Prediction and display
    if st.button("Predict", use_container_width=True):
        if user_input:
            # Transform and predict
            user_input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(user_input_vectorized)
            st.write(f"The quote sounds like something **{prediction[0]}** would say!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:

        st.image('images/LizLemon__image.png', caption='Liz Lemon')
    with col2:
        st.image('images/JackDonaghy_image.png', caption='Jack Donaghy')
    with col3:
        st.image('images/JennaMaroney_image.png', caption='Jenna Maroney')
    with col4:
        st.image('images/TracyJordan__image.png', caption='Tracy Jordan')



if __name__ == "__main__":
    main()