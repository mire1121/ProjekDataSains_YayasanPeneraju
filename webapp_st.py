import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import datetime
import isodate
import re
from googleapiclient.discovery import build
from textblob import TextBlob
from urllib.parse import urlparse, parse_qs
shap.initjs()

# Set up your YouTube API key
YOUTUBE_API_KEY = "AIzaSyCimtxP4XsX8d8h-mG7u-sU6OWBYq9sYTA"

# ---------------------------
# Helper Functions to Load Resources
# ---------------------------
@st.cache_resource
def load_model():
    """Load the pre-trained model from a pickle file."""
    with open("rf_opt_model2.pickle", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_metrics_trainval():
    """Load pre-computed evaluation metrics from a pickle file, if available."""
    try:
        with open("metrics_trainval.pickle", "rb") as f:
            metrics = pickle.load(f)
        return metrics
    except Exception as e:
        return None

def load_metrics_test():
    """Load pre-computed evaluation metrics from a pickle file, if available."""
    try:
        with open("metrics_test.pickle", "rb") as f:
            metrics = pickle.load(f)
        return metrics
    except Exception as e:
        return None

@st.cache_data
def load_data():
    """Load the dataset for insights and visualizations."""
    try:
        df = pd.read_csv('df4.csv')
        return df
    except Exception as e:
        st.error("Dataset not found. Please ensure 'df4.csv' is in the working directory.")
        return None

# Helper function to extract video ID from URL or direct input
def extract_video_id(url):
    if "youtu" in url:
        parsed_url = urlparse(url)
        if "youtube.com" in url:
            query = parse_qs(parsed_url.query)
            if "v" in query:
                return query["v"][0]
        elif "youtu.be" in url:
            return parsed_url.path.lstrip("/")
    return url

# Helper function to get video features from YouTube API
def get_video_features(video_id):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        video_response = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        ).execute()
        if not video_response["items"]:
            return None, None
        video = video_response["items"][0]
        snippet = video["snippet"]
        contentDetails = video["contentDetails"]
        statistics = video["statistics"]

        # Duration in seconds (convert ISO8601 to seconds)
        duration_iso = contentDetails["duration"]
        duration = int(isodate.parse_duration(duration_iso).total_seconds())

        # Title and Title Length
        title = snippet["title"]
        title_length = len(title)

        # Description and Hashtag Count
        description = snippet.get("description", "")
        hashtag_count = len(re.findall(r"#\w+", description))

        # Category from API (using categoryId mapping)
        category_id = snippet.get("categoryId", "")
        category_mapping = {
            "1": "Film & Animation",
            "2": "Autos & Vehicles",
            "10": "Music",
            "15": "Pets & Animals",
            "17": "Sports",
            "20": "Gaming",
            "22": "People & Blogs",
            "23": "Comedy",
            "24": "Entertainment",
            "25": "News & Politics",
            "26": "How-to & Style",
            "27": "Education",
            "28": "Science & Technology"
        }
        category = category_mapping.get(category_id, "Entertainment")

        # Likes, Comments, Views
        likes = int(statistics.get("likeCount", 0))
        comments = int(statistics.get("commentCount", 0))
        views = int(statistics.get("viewCount", 0))

        # Publish date/time and Video Age (in days)
        published_at = snippet["publishedAt"]
        published_date = datetime.datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        video_age = (datetime.datetime.utcnow() - published_date).days

        # Default Audio Language
        audio_language = snippet.get("defaultAudioLanguage", "English")

        # Get channel details
        channel_id = snippet["channelId"]
        channel_response = youtube.channels().list(
            part="snippet,statistics",
            id=channel_id
        ).execute()
        if not channel_response["items"]:
            return None, None
        channel = channel_response["items"][0]
        channel_snippet = channel["snippet"]
        channel_statistics = channel["statistics"]

        channel_subscribers = int(channel_statistics.get("subscriberCount", 0))
        channel_views = int(channel_statistics.get("viewCount", 0))
        total_videos_posted = int(channel_statistics.get("videoCount", 0))

        # Channel Age (in days)
        channel_published_at = channel_snippet["publishedAt"]
        channel_published_date = datetime.datetime.strptime(channel_published_at, "%Y-%m-%dT%H:%M:%SZ")
        channel_age = (datetime.datetime.utcnow() - channel_published_date).days

        # Average Video Views (approx.)
        if total_videos_posted > 0:
            average_video_views = channel_views / total_videos_posted
        else:
            average_video_views = 0

        # Channel Size based on subscribers
        if channel_subscribers < 100000:
            channel_size = "<100k"
        elif channel_subscribers < 1000000:
            channel_size = "<1M"
        elif channel_subscribers < 10000000:
            channel_size = "<10M"
        else:
            channel_size = "10M+"

        # Title Sentiment using TextBlob
        sentiment = TextBlob(title).sentiment.polarity

        # Engagement Rate: (Likes + Comments) / channel_subscribers
        if channel_subscribers > 0:
            engagement_rate = (likes + comments) / channel_subscribers
        else:
            engagement_rate = 0

        # Convert categorical features using the same mappings as in the Predictive Modelling page.
        category_options = {
            "Autos & Vehicles": 0,
            "Comedy": 1,
            "Education": 2,
            "Entertainment": 3,
            "Film & Animation": 4,
            "Gaming": 5,
            "How-to & Style": 6,
            "Music": 7,
            "News & Politics": 8,
            "Nonprofits & Activism": 9,
            "People & Blogs": 10,
            "Pets & Animals": 11,
            "Science & Technology": 12,
            "Sports": 13,
            "Travel & Events": 14
        }
        audio_language_options = {
            "Abkhazian": 0,
            "American Sign Language": 1,
            "Amharic": 2,
            "Arabic": 3,
            "Assamese": 4,
            "Bengali": 5,
            "Bhojpuri": 6,
            "Bosnian": 7,
            "Burmese": 8,
            "Chinese": 9,
            "Chinese (Hong Kong)": 10,
            "Chinese (Simplified Script)": 11,
            "Chinese (Simplified)": 12,
            "Chinese (Taiwan)": 13,
            "Chinese (Traditional Script)": 14,
            "Dutch": 15,
            "English": 16,
            "English (American)": 17,
            "English (Australian)": 18,
            "English (British)": 19,
            "English (Canadian)": 20,
            "English (Indian)": 21,
            "English (Irish)": 22,
            "Filipino": 23,
            "French": 24,
            "French (France)": 25,
            "German": 26,
            "German (Germany)": 27,
            "Greek": 28,
            "Gujarati": 29,
            "Hebrew": 30,
            "Hindi": 31,
            "Hindi (Latin Script)": 32,
            "Indonesian": 33,
            "Italian": 34,
            "Japanese": 35,
            "Khmer": 36,
            "Kinyarwanda": 37,
            "Korean": 38,
            "Malayalam": 39,
            "Marathi": 40,
            "Mongolian": 41,
            "Nepali": 42,
            "No Linguistic Content": 43,
            "Persian": 44,
            "Polish": 45,
            "Portuguese": 46,
            "Portuguese (Brazil)": 47,
            "Punjabi": 48,
            "Romanian": 49,
            "Russian": 50,
            "Serbian": 51,
            "Sinhala": 52,
            "Spanish": 53,
            "Spanish (Latin America)": 54,
            "Spanish (Spain)": 55,
            "Swahili": 56,
            "Tagalog": 57,
            "Tamil": 58,
            "Telugu": 59,
            "Thai": 60,
            "Turkish": 61,
            "Ukrainian": 62,
            "Unknown": 63,
            "Urdu": 64,
            "Uzbek": 65,
            "Vietnamese": 66,
            "Võro": 67,
            "Wolof": 68
        }
        channel_size_options = {
            "<100k": 1,
            "<1M": 2,
            "<10M": 3,
            "10M+": 4
        }
        category_num = category_options.get(category, 3)
        audio_language_num = audio_language_options.get(audio_language, 16)
        channel_size_num = channel_size_options.get(channel_size, 1)

        features = {
            "Duration": duration,
            "Likes": likes,
            "Comments": comments,
            "Category": category_num,
            "Title_Length": title_length,
            "Channel_Subscribers": channel_subscribers,
            "Channel_Views": channel_views,
            "Video_Age": video_age,
            "Average_Video_Views": average_video_views,
            "Hashtag_Count": hashtag_count,
            "Audio_Language": audio_language_num,
            "Total_Videos_Posted": total_videos_posted,
            "Channel_Size_Num": channel_size_num,
            "Channel_Age": channel_age,
            "Title_Sentiment": sentiment,
            "Engagement_Rate": engagement_rate
        }

        return features, views
    except Exception as e:
        print("Error in get_video_features:", e)
        return None, None

# ---------------------------
# Page 1: Project Documentation
# ---------------------------
def project_documentation():
    #st.title("Project Documentation")
    st.markdown(
        """
        # YouTube Video Views Prediction Project
        """
    )
    with st.expander("1. Background"):
        st.markdown(
            """
            YouTube is one of the most popular platforms in the world, with billions of people watching videos every day across countless topics. It has become a hub for content creators, marketers, and advertisers to engage with audiences, share ideas, and promote products. Despite its massive popularity, the platform’s dynamic nature makes it challenging to predict how well a video will perform. The number of views a video receives can depend on many factors, including its title, description, upload timing, and even current trends.
            """)
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("2. Problem"):
            st.markdown(
                """
                What makes one video get millions of views while another struggles to break a thousand? The factors influencing video popularity are complex and often unpredictable. For stakeholders like content creators and advertisers, understanding these factors is critical to maximizing impact and reach.
                """)
        with st.expander("4. Methodology"):
            st.markdown(
                """
                - **Data Collection:** Use the YouTube API to gather video data.
                - **Data Preprocessing:** Clean data, handle missing values, and encode categorical variables. dont forget feature engineering
                - **Model Training:** Train and compare Multiple Linear Regression, Decision Tree, Random Forest, XGBoost, and Support Vector Regressor.
                - **Evaluation:** Use MAE, RMSE, and R-squared metrics along with cross-validation.
                - **Deployment:** Deploy the best-performing model via a Streamlit web application.
                """)
        with st.expander("6. Outcomes/Benefit"):
            st.markdown(
                """
                - A robust predictive model for YouTube video views.
                - Insights into the most significant features affecting video popularity.
                - An interactive web application providing predictions and data insights.
                """)
    with col2:
        with st.expander("3. Objectives"):
            st.markdown(
                """
                - Develop a predictive model to estimate the number of views a YouTube video will receive based on its metadata.
                - Identify the most influential factors driving YouTube video viewership.
                - Assess the performance of different machine learning models for prediction accuracy and effectiveness.
                """)
        with st.expander("5. Challenges"):
            st.markdown(
                """
               - **Data Skewness:** Views are highly skewed; may require log-transformation.
               - **Dynamic Trends:** External factors can influence views.
               - **API Rate Limits:** Requires batching and caching strategies.
               - **Bias in Dataset:** Data may be dominated by large channels.
                """)
        with st.expander("7. Key Findings"):
            st.markdown(
                """
                - among 3 models tested, XGBoost was the best in terms of performance 
                - Feature1, Feature2, Feature3 are the top 3 features affecting the number of views (Y-variable)
                """)

    st.divider()
    st.markdown(
        """
        # YouTube Video Views Prediction Project

        ## 1. Background
        YouTube is one of the most popular platforms in the world, with billions of people watching videos every day across countless topics. It has become a hub for content creators, marketers, and advertisers to engage with audiences, share ideas, and promote products. Despite its massive popularity, the platform’s dynamic nature makes it challenging to predict how well a video will perform. The number of views a video receives can depend on many factors, including its title, description, upload timing, and even current trends.

        ## 2. Problem
        What makes one video get millions of views while another struggles to break a thousand? The factors influencing video popularity are complex and often unpredictable. For stakeholders like content creators and advertisers, understanding these factors is critical to maximizing impact and reach.

        ## 3. Objectives
        - Develop a predictive model to estimate the number of views a YouTube video will receive based on its metadata.
        - Identify the most influential factors driving YouTube video viewership.
        - Assess the performance of different machine learning models for prediction accuracy and effectiveness.

        ## 4. Methodology
        - **Data Collection:** Use the YouTube API to gather video data.
        - **Data Preprocessing:** Clean data, handle missing values, and encode categorical variables. dont forget feature engineering
        - **Model Training:** Train and compare Multiple Linear Regression, Decision Tree, Random Forest, XGBoost, and Support Vector Regressor.
        - **Evaluation:** Use MAE, RMSE, and R-squared metrics along with cross-validation.
        - **Deployment:** Deploy the best-performing model via a Streamlit web application.

        ## 5. Challenges
        - **Data Skewness:** Views are highly skewed; may require log-transformation.
        - **Dynamic Trends:** External factors can influence views.
        - **API Rate Limits:** Requires batching and caching strategies.
        - **Bias in Dataset:** Data may be dominated by large channels.

        ## 6. Outcomes/Benefits
        - A robust predictive model for YouTube video views.
        - Insights into the most significant features affecting video popularity.
        - An interactive web application providing predictions and data insights.

        ## 7. Key Findings
        - among 3 models tested, XGBoost was the best in terms of performance 
        - Feature1, Feature2, Feature3 are the top 3 features affecting the number of views (Y-variable)
        """
    )

# ---------------------------
# Page 2: Features Insights
# ---------------------------
def features_insights():
    st.title("Features Insights")
    st.subheader("1. About Dataset")
    model = load_model()
    df = load_data()
    
    with st.expander("About Dataset"):
        #st.subheader("About Dataset")
        st.markdown(
            """
            ### **Source**  
            YouTube Data API  
        
            ### **Number of Samples**  
            6,936 samples (a subset of the full project dataset)  
        
            ### **Attributes**  
            17 features, including video metadata, engagement metrics, and channel information.   
        
            ### **Data Preprocessing**  
            - Missing values handled using [mention method, e.g., imputation or removal].  
            - Categorical features encoded using [e.g., one-hot encoding, label encoding].  
            - Outliers detected and treated using [e.g., IQR, Z-score].  
        
            ### **Target Variable**  
            The model predicts **YouTube video views** based on available metadata.  
        
            ### **Limitations**  
            - Data represents only a subset of YouTube videos and may not be fully generalizable.  
            - External factors such as trends, promotions, or algorithm changes are not included.  
    
             Dive deep into the dataset’s features with detailed descriptions of each attribute. Understand what each feature represents and how it contributes to predicting YouTube video views. 
            """
        )
    with st.expander("Features Descriptions"):
        # New Section: Attributes and their Descriptions
        attributes = [
            "Video ID", "Title", "Description", "Category", "Duration", 
            "Publish Date", "Publish Time", "Likes", "Comments", 
            "Default Audio Language", "Channel Title", "Channel Subscribers", 
            "Total Videos Posted", "Channel Views", "Channel Creation Date", 
            "Channel Country", "Views"
        ]
        descriptions = [
            "Unique key for each video", "Title of the video", "Description of the video", 
            "Category of the video", "Duration of the video in seconds", 
            "Date the video was published", "Time in 24-hour format", 
            "Number of likes", "Number of comments", "Language used in the video", 
            "Name of the channel", "Number of subscribers for the channel", 
            "Total number of videos uploaded by the channel", 
            "Total number of views for the channel", "Date the channel was created", 
            "Country where the channel originated", "Number of views"
        ]
        
        attributes_df = pd.DataFrame({
            "Features": attributes,
            "Description": descriptions
        })
        
        st.table(attributes_df)

    with st.expander("Dataset Overview"):
        if df is not None:
            st.markdown(
                """
                Take a quick glance at the dataset with a snapshot view (first 5 rows). This preview highlights the structure and content of the data used for our analysis.
                """
            )
            st.write(df.head())
        else:
            st.error("Dataset could not be loaded.")

    # Sample DataFrames for Original and Feature Engineered Datasets
    original_data = pd.DataFrame({
        "Column": [
            "Video ID", "Title", "Description", "Category", "Duration (seconds)", "Publish Date",
            "Publish Time (24h)", "Views", "Likes", "Comments", "Default Audio Language",
            "Channel Title", "Channel Subscribers", "Total Videos Posted", "Channel Views",
            "Channel Creation Date", "Channel Country"
        ],
        "Dtype": [
            "object", "object", "object", "int64", "int64", "datetime64[ns]", "object", "int64", "int64", "int64", "object",
            "object", "int64", "int64", "int64", "datetime64[ns]", "object"
        ]
    })
    
    feature_engineered_data = pd.DataFrame({
        "Column": [
            "Views", "Duration", "Likes", "Comments", "Channel_Subscribers", "Total_Videos_Posted",
            "Channel_Views", "Title_Length", "Hashtag_Count", "Quarter", "Video_Age", "Engagement_Rate",
            "Channel_Title_Length", "Average_Video_Views", "Channel_Age", "Title_Sentiment", "Weekday",
            "Month", "Description_Present", "Duration_Category", "Channel_Size", "Publish_Hour_Time",
            "Weekly_Videos", "Monthly_Videos", "Category", "Audio_Language", "Channel_Country", "Sentiment_Category"
        ],
        "Dtype": [
            "int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64",
            "int64", "float64", "int64", "float64", "int64", "float64", "int64", "int64", "int64",
            "int64", "int64", "int32", "float64", "float64", "int32", "int32", "int32", "int32"
        ]
    })

    st.subheader("2. Exploratory Data Analysis (EDA)")
    tab1, tab2, tab3 = st.tabs(["Data Types", "Descriptive Statistics", "Feature Elements"])
    
    with tab1:
        # Layout for Side-by-Side DataFrames
        #st.title("EDA & Feature Engineering")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Dataset")
            st.table(original_data)
        
        with col2:
            st.subheader("Feature Engineered Dataset")
            st.table(feature_engineered_data)
    
    with tab2:
        # Descriptive Statistics for Numerical Features in Transformed Dataset
        #st.subheader("Descriptive Statistics (Transformed Dataset)")
        pd.set_option('display.float_format', '{:.2f}'.format)
        # Assuming 'transformed_df' is your processed dataset
        # Replace transformed_df with the actual dataset variable
        st.dataframe(df.describe())

    with tab3:
        # Feature Selection & Value Counts with Encoded Dictionary
        #st.subheader("Feature Distribution")
        feature_names = ['Channel_Country', 'Month', 'Publish_Hour_Time']
        
        Chosen_Features = st.selectbox(
            "Choose a feature to see its element count", 
            feature_names,
            help="Select a feature to display its unique elements and their respective counts, along with their encoded values. Useful for categorical and discrete numerical features."
        )
        
        # Assuming 'encoding_dict' holds the mapping from original categorical values to encoded values
        encoding_dict = {
            'Channel_Country':{0: 'Algeria', 1: 'Argentina', 2: 'Australia', 3: 'Austria', 4: 'Azerbaijan', 5: 'Bangladesh', 6: 'Belarus', 7: 'Belgium', 8: 'Brazil', 9: 'Bulgaria', 10: 'Cambodia', 11: 'Canada', 12: 'Chile', 13: 'China', 14: 'Colombia', 15: 'Costa Rica', 16: 'Croatia', 17: 'Cyprus', 18: 'Czech Republic', 19: 'Denmark', 20: 'Dominican Republic', 21: 'Ecuador', 22: 'Egypt', 23: 'Estonia', 24: 'Finland', 25: 'France', 26: 'Georgia', 27: 'Germany', 28: 'Greece', 29: 'Hong Kong', 30: 'Hungary', 31: 'Iceland', 32: 'India', 33: 'Indonesia', 34: 'Iraq', 35: 'Ireland', 36: 'Israel', 37: 'Italy', 38: 'Japan', 39: 'Jordan', 40: 'Kazakhstan', 41: 'Kenya', 42: 'Latvia', 43: 'Lithuania', 44: 'Luxembourg', 45: 'Malaysia', 46: 'Malta', 47: 'Mexico', 48: 'Montenegro', 49: 'Morocco', 50: 'Nepal', 51: 'Netherlands', 52: 'New Zealand', 53: 'Nicaragua', 54: 'Nigeria', 55: 'Norway', 56: 'Oman', 57: 'Pakistan', 58: 'Panama', 59: 'Peru', 60: 'Philippines', 61: 'Poland', 62: 'Portugal', 63: 'Qatar', 64: 'Romania', 65: 'Russia', 66: 'Saudi Arabia', 67: 'Senegal', 68: 'Serbia', 69: 'Singapore', 70: 'Slovakia', 71: 'Slovenia', 72: 'South Africa', 73: 'South Korea', 74: 'Spain', 75: 'Sri Lanka', 76: 'Sweden', 77: 'Switzerland', 78: 'Taiwan', 79: 'Tanzania', 80: 'Thailand', 81: 'Tunisia', 82: 'Turkey', 83: 'Ukraine', 84: 'United Arab Emirates', 85: 'United Kingdom', 86: 'United States', 87: 'Unknown', 88: 'Vietnam'},
            'Month': {1: 'April', 2: 'August', 3: 'December', 4: 'February', 5: 'January', 6: 'July', 7: 'June', 8: 'March', 9: 'May', 10: 'November', 11: 'October', 12: 'September'},
            'Publish_Hour_Time': {12: '20_Evening', 4: '13_Afternoon', 2: '11_Morning', 5: '14_Afternoon', 1: '10_Morning', 9: '18_Evening', 10: '19_Evening', 8: '17_Afternoon', 20: '6_Morning', 3: '12_Afternoon', 6: '15_Afternoon', 21: '7_Morning', 0: '0_Night', 17: '3_Night', 7: '16_Afternoon', 22: '8_Morning', 19: '5_Night', 14: '22_Night', 13: '21_Night', 16: '2_Night', 15: '23_Night', 23: '9_Morning', 11: '1_Night', 18: '4_Night'}
        }
        
        # Retrieve value counts
        value_counts = df[Chosen_Features].value_counts().reset_index()
        value_counts.columns = ['Element', 'Count']
        
        # Map encoded values back to original values
        value_counts['Original Element'] = value_counts['Element'].map(encoding_dict[Chosen_Features])
        
        # Rearranging columns
        value_counts = value_counts[['Element', 'Original Element', 'Count']]
        
        st.dataframe(value_counts)

    
    st.subheader("3. Explore The Relationships & Distributions")
    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Feature Distributions", "Scatter Plot"])
    with tab1:
        st.subheader("Correlation Heatmap")
        st.markdown(
            """
            Delve into the interrelationships between features using an interactive via the correlation heatmap. This visualization uncovers the strength and direction of associations among video metrics, helping you pinpoint which attributes most influence views and engagement.
            """
        )
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    with tab2:
        st.subheader("Feature Distributions")
        st.markdown(
            """
            Analyze how each feature is distributed across the dataset. Select a feature to view its histogram and gain insights into its impact on video performance.
            """
        )
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_feature = st.selectbox("Select a feature to view its distribution", numeric_columns, index=0)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_feature}")
        st.pyplot(fig)
    with tab3:
        st.subheader("Scatter Plot between Two Features")
        st.markdown(
            """
            Compare two features side by side using a scatter plot. Select features for the X and Y axes to uncover potential trends and patterns in video performance.
            """
        )
        col1, col2 = st.columns(2)
        x_feature = col1.selectbox("Select feature for X-axis", numeric_columns, index=0)
        y_feature = col2.selectbox("Select feature for Y-axis", numeric_columns, index=1)

        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=df[x_feature], y=df[y_feature], ax=ax3)
        ax3.set_title(f"Scatter Plot: {x_feature} vs {y_feature}")
        st.pyplot(fig3)
    
    

# ---------------------------
# Page 3: Features Importance (with SHAP Analysis)
# ---------------------------
def features_importance():
    st.title("Features Importance")

    # Load model and data
    model = load_model()
    df = load_data()
    
    if df is not None and model is not None:
        # Get feature names
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = list(df.drop(columns=['Views'], errors='ignore').columns)  # Ignore error if 'Views' is missing

        # Get feature importances
        if hasattr(model, "feature_importances_"):  # Ensure model supports feature importance
            importances = model.feature_importances_
            forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            
            # Convert to DataFrame
            importance_df = forest_importances.reset_index()
            importance_df.columns = ['Feature', 'Importance']

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x="Importance",
                y="Feature",
                data=importance_df,
                palette="viridis",
                ax=ax
            )
            ax.set_title("Feature Importances from Random Forest")
            ax.set_xlabel("Importance Score")
            ax.set_ylabel("Features")

            st.pyplot(fig)  # Display plot in Streamlit
        else:
            st.error("The selected model does not support feature importance.")
    else:
        st.error("Error: Model or dataset could not be loaded.")
# ---------------------------
# Page 4: Predictive Modelling (with Real-Time YouTube Prediction)
# ---------------------------
def predictive_modelling():
    st.title("Predictive Modelling - YouTube Views Prediction")
    st.markdown(
        """
        Using the selected video attributes, a regression model was built to estimate the potential number of views a YouTube video might receive. Choose the relevant features and click Submit to generate a prediction based on your video’s characteristics.
        """
    )

    # Load evaluation metrics if available and display them in the sidebar.
    metrics_trainval = load_metrics_trainval()
    if metrics_trainval:
        st.sidebar.header("Evaluation Metrics (Train + Validate Data)")
        st.sidebar.write("**R² Score:**", np.round(metrics_trainval.get("r2", 0), 3))
        st.sidebar.write("**Mean Absolute Error (MAE):**", np.round(metrics_trainval.get("mae", 0), 3))
        st.sidebar.write("**Mean Squared Error (MSE):**", np.round(metrics_trainval.get("mse", 0), 3))
    else:
        st.sidebar.info("Evaluation metrics not available.")

    # Load evaluation metrics if available and display them in the sidebar.
    metrics_test = load_metrics_test()
    if metrics_test:
        st.sidebar.header("Evaluation Metrics (Test Data)")
        st.sidebar.write("**R² Score:**", np.round(metrics_test.get("r2", 0), 3))
        st.sidebar.write("**Mean Absolute Error (MAE):**", np.round(metrics_test.get("mae", 0), 3))
        st.sidebar.write("**Mean Squared Error (MSE):**", np.round(metrics_test.get("mse", 0), 3))
    else:
        st.sidebar.info("Evaluation metrics not available.")

    st.subheader("Enter Video Details Manually")
    # Mappings for categorical features
    category_options = {
        "Autos & Vehicles": 0,
        "Comedy": 1,
        "Education": 2,
        "Entertainment": 3,
        "Film & Animation": 4,
        "Gaming": 5,
        "How-to & Style": 6,
        "Music": 7,
        "News & Politics": 8,
        "Nonprofits & Activism": 9,
        "People & Blogs": 10,
        "Pets & Animals": 11,
        "Science & Technology": 12,
        "Sports": 13,
        "Travel & Events": 14
    }

    audio_language_options = {
        "Abkhazian": 0,
        "American Sign Language": 1,
        "Amharic": 2,
        "Arabic": 3,
        "Assamese": 4,
        "Bengali": 5,
        "Bhojpuri": 6,
        "Bosnian": 7,
        "Burmese": 8,
        "Chinese": 9,
        "Chinese (Hong Kong)": 10,
        "Chinese (Simplified Script)": 11,
        "Chinese (Simplified)": 12,
        "Chinese (Taiwan)": 13,
        "Chinese (Traditional Script)": 14,
        "Dutch": 15,
        "English": 16,
        "English (American)": 17,
        "English (Australian)": 18,
        "English (British)": 19,
        "English (Canadian)": 20,
        "English (Indian)": 21,
        "English (Irish)": 22,
        "Filipino": 23,
        "French": 24,
        "French (France)": 25,
        "German": 26,
        "German (Germany)": 27,
        "Greek": 28,
        "Gujarati": 29,
        "Hebrew": 30,
        "Hindi": 31,
        "Hindi (Latin Script)": 32,
        "Indonesian": 33,
        "Italian": 34,
        "Japanese": 35,
        "Khmer": 36,
        "Kinyarwanda": 37,
        "Korean": 38,
        "Malayalam": 39,
        "Marathi": 40,
        "Mongolian": 41,
        "Nepali": 42,
        "No Linguistic Content": 43,
        "Persian": 44,
        "Polish": 45,
        "Portuguese": 46,
        "Portuguese (Brazil)": 47,
        "Punjabi": 48,
        "Romanian": 49,
        "Russian": 50,
        "Serbian": 51,
        "Sinhala": 52,
        "Spanish": 53,
        "Spanish (Latin America)": 54,
        "Spanish (Spain)": 55,
        "Swahili": 56,
        "Tagalog": 57,
        "Tamil": 58,
        "Telugu": 59,
        "Thai": 60,
        "Turkish": 61,
        "Ukrainian": 62,
        "Unknown": 63,
        "Urdu": 64,
        "Uzbek": 65,
        "Vietnamese": 66,
        "Võro": 67,
        "Wolof": 68
    }

    channel_size_options = {
        "<100k": 1,
        "<1M": 2,
        "<10M": 3,
        "10M+": 4
    }
    
    # Create a form for manual input.
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
    
        with col1:
            Duration = st.number_input(
                "Duration (seconds)", min_value=0, value=700,
                help="Enter the video duration in seconds."
            )
            Comments = st.number_input(
                "Comments", min_value=0, value=2500,
                help="Enter the number of comments on the video."
            )
            Title_Length = st.number_input(
                "Title Length", min_value=0, value=50,
                help="Enter the number of characters in the video title."
            )
            Channel_Views = st.number_input(
                "Channel Views", min_value=0, value=5000000,
                help="Enter the total number of views for the channel."
            )
            Average_Video_Views = st.number_input(
                "Average Video Views", min_value=0.0, value=1500000.0,
                help="Enter the average number of views for videos on this channel."
            )
            Audio_Language = st.selectbox(
                "Audio Language", list(audio_language_options.keys()),
                help="Select the default audio language of the video."
            )
            Channel_Size_Num = st.selectbox(
                "Channel Size", list(channel_size_options.keys()),
                help="Select the size category of the channel."
            )
        
        with col2:
            Likes = st.number_input(
                "Likes", min_value=0, value=150000,
                help="Enter the number of likes the video has received."
            )
            Category = st.selectbox(
                "Category", list(category_options.keys()),
                help="Select the video category."
            )
            Channel_Subscribers = st.number_input(
                "Channel Subscribers", min_value=0, value=1000000,
                help="Enter the number of subscribers of the channel."
            )
            Video_Age = st.number_input(
                "Video Age (days)", min_value=0, value=500,
                help="Enter how many days old the video is."
            )
            Hashtag_Count = st.number_input(
                "Hashtag Count", min_value=0, value=3,
                help="Enter the number of hashtags used in the video description."
            )
            Total_Videos_Posted = st.number_input(
                "Total Videos Posted", min_value=0, value=6000,
                help="Enter the total number of videos posted by the channel."
            )
            Channel_Age = st.number_input(
                "Channel Age (Days)", min_value=0, value=2500,
                help="Enter the age of the channel in days."
            )

        Title_Sentiment = st.slider("Title Sentiment", min_value=-1.00, max_value=1.00, value=0.00,
                                    help="Enter the sentiment score of the video title.")
    
        submit_button = st.form_submit_button(label="Predict Views")
    
        if submit_button:
            # Compute Engagement_Rate based on provided inputs.
            if Channel_Subscribers > 0:
                Engagement_Rate = (Likes + Comments) / Channel_Subscribers
            else:
                Engagement_Rate = 0
            
            input_data = {
                'Duration': Duration,
                'Likes': Likes,
                'Comments': Comments,
                'Category': category_options[Category],
                'Title_Length': Title_Length,
                'Channel_Subscribers': Channel_Subscribers,
                'Channel_Views': Channel_Views,
                'Video_Age': Video_Age,
                'Average_Video_Views': Average_Video_Views,
                'Hashtag_Count': Hashtag_Count,
                'Audio_Language': audio_language_options[Audio_Language],
                'Total_Videos_Posted': Total_Videos_Posted,
                'Channel_Size_Num': channel_size_options[Channel_Size_Num],
                'Channel_Age': Channel_Age,
                'Title_Sentiment': Title_Sentiment,
                'Engagement_Rate': Engagement_Rate
            }
            
            input_df = pd.DataFrame([input_data])
            model = load_model()  # Load the pre-trained model
            prediction = model.predict(input_df)
            
            st.success("Prediction Complete!")
            colA, colB = st.columns(2)
            colA.markdown("#### Predicted Number of Views:")
            colB.markdown("#### " + str(np.round(prediction[0], 0)))
    
    # ---------------------------
    # Real-Time YouTube Video Prediction Section
    # ---------------------------
    st.subheader("Real-Time YouTube Video Prediction")
    st.markdown(
        """
        Instantly analyze and predict the potential views of any YouTube video! Simply search for a YouTube video, and our model will evaluate its key attributes to estimate its future performance.
        """
    )
    
    # Main option: Search for a YouTube video by name
    search_query = st.text_input("Search for a YouTube video")
    
    if search_query:
        try:
            # Use the YouTube Data API to search for videos matching the query
            youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            search_response = youtube.search().list(
                q=search_query,
                part="snippet",
                type="video",
                maxResults=5
            ).execute()
    
            # Prepare a list of video titles and a mapping of title to video ID
            video_options = []
            video_mapping = {}
            for item in search_response.get("items", []):
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]
                video_options.append(title)
                video_mapping[title] = video_id
    
            if video_options:
                selected_video_title = st.selectbox("Select a video", video_options)
                if st.button("Predict Video Views"):
                    video_id = video_mapping[selected_video_title]
                    features, true_views = get_video_features(video_id)
                    if features is not None:
                        input_df_rt = pd.DataFrame([features])
                        model = load_model()
                        predicted_views = model.predict(input_df_rt)[0]
                        st.success("Real-Time Prediction Complete!")
                        st.write("#### Predicted Number of Views:", np.round(predicted_views, 0))
                        st.write("#### Actual Number of Views:", true_views)
                    else:
                        st.error("Could not retrieve video details. Please try another video.")
            else:
                st.info("No videos found. Please try a different search query.")
        except Exception as e:
            st.error("Error during video search: " + str(e))


# ---------------------------
# Sidebar Navigation
# ---------------------------
page = st.sidebar.selectbox("Navigation", 
    ["Project Documentation", "Features Insights", "Features Importance", "Predictive Modelling"]
)

if page == "Project Documentation":
    project_documentation()
elif page == "Features Insights":
    features_insights()
elif page == "Features Importance":
    features_importance()
elif page == "Predictive Modelling":
    predictive_modelling()
