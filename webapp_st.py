import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Helper Functions to Load Resources
# ---------------------------
@st.cache_resource
def load_model():
    """Load the pre-trained model from a pickle file."""
    with open("xgb_opt_model.pickle", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_metrics():
    """Load pre-computed evaluation metrics from a pickle file, if available."""
    try:
        with open("metrics.pickle", "rb") as f:
            metrics = pickle.load(f)
        return metrics
    except Exception as e:
        return None

@st.cache_data
def load_data():
    """Load the dataset for insights and visualizations."""
    try:
        df = pd.read_csv('df3.csv')
        return df
    except Exception as e:
        st.error("Dataset not found. Please ensure 'your_dataset.csv' is in the working directory.")
        return None

# ---------------------------
# Page 1: Project Documentation
# ---------------------------
def project_documentation():
    st.title("Project Documentation")
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

        ## 4. Data Collection
        - **Source:** YouTube Data API
        - **Number of Samples:** 6938 samples (a subset of the full project dataset)
        - **Attributes:** 17 attributes including video metadata, engagement metrics, and channel information.

        ## 5. Feature Engineering
        - **Text Features:** Title Length, Description Length, Keyword Presence, Sentiment Analysis.
        - **Engagement Features:** Engagement Ratio (Likes + Comments / Views).
        - **Temporal Features:** Day of the Week, Time of Day, Seasonality.
        - **Channel Features:** Subscriber Count, Channel Age, Number of Videos Posted.
        - **Hashtag Features:** Number of Hashtags.

        ## 6. Methodology
        - **Data Collection:** Use the YouTube API to gather video data.
        - **Data Preprocessing:** Clean data, handle missing values, and encode categorical variables.
        - **Model Training:** Train and compare Multiple Linear Regression, Decision Tree, Random Forest, XGBoost, and Support Vector Regressor.
        - **Evaluation:** Use MAE, RMSE, and R-squared metrics along with cross-validation.
        - **Deployment:** Deploy the best-performing model via a Streamlit web application.

        ## 7. Challenges
        - **Data Skewness:** Views are highly skewed; may require log-transformation.
        - **Dynamic Trends:** External factors can influence views.
        - **API Rate Limits:** Requires batching and caching strategies.
        - **Bias in Dataset:** Data may be dominated by large channels.

        ## 8. Expected Outcomes
        - A robust predictive model for YouTube video views.
        - Insights into the most significant features affecting video popularity.
        - An interactive web application providing predictions and data insights.
        """
    )

# ---------------------------
# Page 2: Features Insights
# ---------------------------
def features_insights():
    st.title("Features Insights")
    
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
        "Attributes": attributes,
        "Description": descriptions
    })
    
    st.subheader("About Dataset")
    st.table(attributes_df)
    
    # Load the dataset
    df = load_data()
    if df is not None:
        st.subheader("Dataset Overview")
        st.write(df.head())

        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

        st.subheader("Feature Distributions")
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_feature = st.selectbox("Select a feature to view its distribution", numeric_columns, index=0)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_feature}")
        st.pyplot(fig)
        
        # New Section: Scatter Plot between Two Features
        st.subheader("Scatter Plot between Two Features")
        col1, col2 = st.columns(2)
        x_feature = col1.selectbox("Select feature for X-axis", numeric_columns, index=0)
        y_feature = col2.selectbox("Select feature for Y-axis", numeric_columns, index=1)

        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=df[x_feature], y=df[y_feature], ax=ax3)
        ax3.set_title(f"Scatter Plot: {x_feature} vs {y_feature}")
        st.pyplot(fig3)
    
    else:
        st.error("Dataset could not be loaded.")

# ---------------------------
# Page 3: Features Importance
# ---------------------------
def features_importance():
    st.title("Features Importance")
    model = load_model()
    df = load_data()
    
    if df is not None:
        # Attempt to extract feature names from the model if available.
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            # Fallback: use dataset columns (assuming 'Views' is the target)
            if 'Views' in df.columns:
                feature_names = list(df.drop('Views', axis=1).columns)
            else:
                # As a last resort, use a hardcoded list.
                feature_names = [
                    'Duration', 'Likes', 'Comments', 'Category', 'Title_Length', 
                    'Channel_Subscribers', 'Channel_Views', 'Video_Age', 
                    'Average_Video_Views', 'Hashtag_Count', 'Audio_Language', 
                    'Total_Videos_Posted', 'Channel_Size_Num', 'Channel_Age', 
                    'Title_Sentiment'
                ]
        
        # Extract feature importance from the model.
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)  # Use absolute values for linear models.
        else:
            st.error("The loaded model does not support feature importance extraction.")
        
        if importances is not None:
            # Check if the number of features matches the importance values.
            if len(importances) != len(feature_names):
                st.error(
                    f"Mismatch detected: Number of features is {len(feature_names)} but "
                    f"number of importance values is {len(importances)}. "
                    "Please ensure that the features used for training match the ones provided here."
                )
            else:
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": np.round(importances, 3)
                }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
                
                # Format for display but keep the column numerical for plotting
                importance_df_display = importance_df.copy()
                importance_df_display["Importance"] = importance_df_display["Importance"].apply(lambda x: f"{x:.3f}")
                
                st.subheader("Feature Importance Ranking")
                st.table(importance_df_display)  # Show formatted table
                
                # Apply the dark theme
                st.markdown("<h1 style='text-align: center;'>Feature Importance Chart</h1>", unsafe_allow_html=True)
                plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
                
                # Create the figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Ensure Importance is numeric
                sns.barplot(x=importance_df["Importance"], y=importance_df["Feature"], ax=ax, palette="coolwarm")
                
                # Customize the plot
                ax.set_xlabel("Importance", fontsize=12)
                ax.set_ylabel("Feature", fontsize=12)
                ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
                ax.grid(True, linestyle="--", alpha=0.5)
                
                # Display the plot in Streamlit
                st.pyplot(fig)
                
        else:
            st.error("Dataset not loaded; cannot compute feature importances.")

# ---------------------------
# Page 4: Predictive Modelling
# ---------------------------
def predictive_modelling():
    st.title("Predictive Modelling - YouTube Views Prediction")

    # Load evaluation metrics if available and display them in the sidebar.
    metrics = load_metrics()
    if metrics:
        st.sidebar.header("Model Evaluation Metrics - XGBoost")
        st.sidebar.write("**R² Score:**", np.round(metrics.get("r2", 0), 3))
        st.sidebar.write("**Mean Absolute Error (MAE):**", np.round(metrics.get("mae", 0), 3))
        st.sidebar.write("**Root Mean Squared Error (RMSE):**", np.round(metrics.get("rmse", 0), 3))
    else:
        st.sidebar.info("Evaluation metrics not available.")

    st.subheader("Enter Video Details")
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
    
    # Create a form with dynamic validation and help texts.
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
            # For example, using (Likes + Comments) / Channel_Views.
            if Channel_Views > 0:
                Engagement_Rate = (Likes + Comments) / Channel_Subscribers
            else:
                Engagement_Rate = 0
            
            # Prepare the input data matching the model's expected features.
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
            
            # Convert the dictionary into a DataFrame (with one row)
            input_df = pd.DataFrame([input_data])
            
            model = load_model()  # Load the pre-trained model
            prediction = model.predict(input_df)
            
            st.success("Prediction Complete!")
            col1, col2 = st.columns(2)
            col1.markdown("#### Predicted Number of Views:")
            col2.markdown("#### " + str(np.round(prediction[0], 0)))

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
