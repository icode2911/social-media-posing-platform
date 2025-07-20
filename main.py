import streamlit as st
from dotenv import load_dotenv
import os
import json
import pytz
from datetime import datetime, timedelta, date
import random
import time

from utils import (
    save_user_settings,
    save_uploaded_file,
    load_user_settings,
    fetch_trending_topics_hardcoded,
    extract_text_from_pdf,
    extract_text_from_txt,
    chunk_text,
    embed_chunks,
    save_embeddings,
    load_embeddings,
    retrieve_relevant_chunks,
    generate_tweet_openai,
    post_tweet_to_x
)

# ---- Scheduler Setup ----
from apscheduler.schedulers.background import BackgroundScheduler

def scheduled_post_job(tweet, user_id, idx):
    ok, msg = post_tweet_to_x(tweet)
    log_entry = {
        "timestamp": str(datetime.now()),
        "tweet_index": idx,
        "tweet": tweet,
        "result": msg
    }
    with open(f"data/{user_id}_tweet_post_log.txt", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    # Mark as posted
    schedule_file = f"data/{user_id}_tweet_schedule.json"
    if os.path.exists(schedule_file):
        with open(schedule_file, "r") as f:
            tweets = json.load(f)
        for t in tweets:
            if t.get("tweet") == tweet and t.get("status") == "pending":
                t["status"] = "posted"
                t["posted_time"] = str(datetime.now())
        with open(schedule_file, "w") as f:
            json.dump(tweets, f, indent=2)

def setup_scheduler():
    if "scheduler" not in st.session_state:
        scheduler = BackgroundScheduler()
        scheduler.start()
        st.session_state.scheduler = scheduler
    return st.session_state.scheduler

load_dotenv()
st.set_page_config(page_title="Social Media Auto Posting Platform", layout="wide")
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Onboarding", "Dashboard"])
user_id = "default_user"

if app_mode == "Onboarding":
    st.title("🟦 Social Media Auto Posting Platform")
    st.write("Let's set up your account for automated tweeting.")

    saved_settings = load_user_settings(user_id)
    if saved_settings:
        st.success("Loaded your previous settings:")
        st.json(saved_settings)

    with st.form("onboarding_form"):
        st.subheader("1. Upload a Document (PDF or TXT)")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

        st.subheader("2. Enter Custom Instructions")
        custom_instructions = st.text_area("Write how you'd like your tweets to be generated:", height=100)

        st.subheader("3. (Optional) Provide a Website URL")
        website_url = st.text_input("Website URL")

        st.subheader("4. Configure Your Tweet Schedule")
        num_tweets = st.number_input("How many tweets per day?", min_value=1, max_value=10, value=3)
        tweet_times = st.text_input("Enter custom time slots (comma-separated, e.g., 10:00,14:30,18:00)")

        submitted = st.form_submit_button("Save Settings")

    if submitted:
        file_path = None
        if uploaded_file:
            file_path = save_uploaded_file(user_id, uploaded_file)

        settings = {
            "instructions": custom_instructions,
            "website_url": website_url,
            "num_tweets": num_tweets,
            "tweet_times": tweet_times,
            "uploaded_file": file_path
        }
        save_user_settings(user_id, settings)

        # Clear session state to avoid dashboard showing old info
        for key in ["scheduled_tweets", "selected_topics"]:
            if key in st.session_state:
                del st.session_state[key]

        st.success("Settings and file saved! Redirecting to Dashboard in 2 seconds...")
        time.sleep(2)
        st.rerun()

elif app_mode == "Dashboard":
    # Always reload settings for freshest data
    saved_settings = load_user_settings(user_id)
    onboarding_complete_path = f"data/{user_id}_onboarding_complete.txt"
    if not os.path.exists(onboarding_complete_path):
        st.warning("Please complete onboarding first!")
        st.info("Go to Onboarding from the sidebar.")
        st.stop()

    st.title("📋 Dashboard")
    st.info("Manage your topics, preview content, and prepare for tweet generation.")

    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

    # ----- Trending Topics Selection -----
    st.header("🌐 Trending Topics (Demo List)")
    num_topics = st.slider("How many trends to display?", 5, 20, 10)
    all_topics = fetch_trending_topics_hardcoded(num_topics)

    user_selected_topics_path = f"data/{user_id}_selected_topics.json"
    if os.path.exists(user_selected_topics_path):
        with open(user_selected_topics_path, "r") as f:
            prev_selected_topics = json.load(f)
    else:
        prev_selected_topics = all_topics[:3]

    selected_topics = st.multiselect(
        "Select the trending topics you want to use for tweet generation:",
        all_topics,
        default=prev_selected_topics
    )

    if st.button("Save Selected Topics"):
        st.success(f"You have selected: {selected_topics}")
        with open(user_selected_topics_path, "w") as f:
            json.dump(selected_topics, f)
        st.info("Selected topics saved and ready for tweet generation!")

    # ----- Extracted Content Preview -----
    st.header("📑 Extracted Content from Your Upload")
    uploaded_file_path = None
    if saved_settings and saved_settings.get("uploaded_file"):
        uploaded_file_path = saved_settings["uploaded_file"]

    extracted_text = ""
    if uploaded_file_path:
        ext = os.path.splitext(uploaded_file_path)[1].lower()
        if ext == ".pdf":
            extracted_text = extract_text_from_pdf(uploaded_file_path)
        elif ext == ".txt":
            extracted_text = extract_text_from_txt(uploaded_file_path)
        else:
            extracted_text = "Unsupported file format."
    else:
        extracted_text = "No file uploaded."

    st.text_area("Extracted Content (auto)", extracted_text[:2000], height=300)

    # ----- Chunk, Embed, and Save Embeddings (RAG indexing) -----
    if uploaded_file_path and extracted_text and not extracted_text.startswith("Error"):
        emb_path = f"data/{user_id}_embeddings.npy"
        chunk_path = f"data/{user_id}_chunks.json"
        if not (os.path.exists(emb_path) and os.path.exists(chunk_path)):
            with st.spinner("Chunking and embedding your document (first time only)..."):
                chunks = chunk_text(extracted_text, chunk_size=400, overlap=50)
                embeddings = embed_chunks(chunks)
                save_embeddings(user_id, embeddings, chunks)
            st.success("Document indexed for knowledge retrieval!")

    # ----- Generate and Schedule Unique Tweets -----
    st.header("✍️ Generate & Schedule Unique Tweets")

    if st.button("Generate Unique Scheduled Tweets"):
        num_tweets = int(saved_settings.get("num_tweets", 3))
        tweet_times = saved_settings.get("tweet_times", "10:00,14:30,18:00")
        time_slots = [t.strip() for t in tweet_times.split(",") if t.strip()]
        today = date.today()

        if len(time_slots) < num_tweets:
            st.error("Not enough time slots for the number of tweets. Please update your schedule in onboarding.")
            st.stop()

        topics_for_today = random.sample(selected_topics, min(num_tweets, len(selected_topics)))
        if len(topics_for_today) < num_tweets:
            topics_for_today = (topics_for_today * (num_tweets // len(topics_for_today) + 1))[:num_tweets]

        scheduled_tweets = []
        for i in range(num_tweets):
            topic = topics_for_today[i]
            scheduled_time = time_slots[i % len(time_slots)]
            dt_str = f"{today} {scheduled_time}"
            dt_local = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
            user_tz = pytz.timezone("Asia/Kolkata")
            dt_utc = user_tz.localize(dt_local).astimezone(pytz.utc)
            relevant_chunks = retrieve_relevant_chunks(user_id, topic, top_k=(i+1))
            context_snippet = relevant_chunks[-1] if relevant_chunks else ""
            tweet = generate_tweet_openai(
                topic=topic,
                content_snippet=context_snippet,
                custom_instructions=saved_settings.get("instructions", "")
            )
            scheduled_tweets.append({
                "tweet": tweet,
                "datetime_utc": dt_utc.isoformat(),
                "topic": topic,
                "status": "pending"
            })

        with open(f"data/{user_id}_tweet_schedule.json", "w") as f:
            json.dump(scheduled_tweets, f, indent=2)

        st.success("Unique tweets generated and scheduled for posting!")
        st.rerun()  # Refresh to show the schedule immediately

    # ---- Show Current Schedule ----
    schedule_file = f"data/{user_id}_tweet_schedule.json"
    if os.path.exists(schedule_file):
        with open(schedule_file, "r") as f:
            tweet_schedule_list = json.load(f)
        st.header("📆 Scheduled Tweets")
        for i, entry in enumerate(tweet_schedule_list):
            dt_utc = datetime.fromisoformat(entry['datetime_utc'])
            dt_local = dt_utc.astimezone(pytz.timezone("Asia/Kolkata"))
            st.markdown(f"**Tweet #{i+1} ({entry['topic']})**")
            st.write("Scheduled for (IST):", dt_local.strftime("%Y-%m-%d %H:%M:%S %Z (%z)"))
            st.write("Scheduled for (UTC):", dt_utc.strftime("%Y-%m-%d %H:%M:%S %Z (%z)"))
            st.code(entry["tweet"])
            st.write(f"Status: {entry.get('status', 'pending')}")
        st.info("These tweets will be posted automatically at the scheduled times.")

    # ---- APScheduler Background Posting ----
    scheduler = setup_scheduler()
    if os.path.exists(schedule_file):
        with open(schedule_file) as f:
            tweet_schedule_list = json.load(f)
        for idx, entry in enumerate(tweet_schedule_list):
            tweet = entry["tweet"]
            dt_utc = datetime.fromisoformat(entry["datetime_utc"])
            job_id = f"{user_id}_{hash(tweet+str(dt_utc))}"
            if entry.get("status", "pending") == "pending" and not scheduler.get_job(job_id):
                scheduler.add_job(
                    scheduled_post_job,
                    'date',
                    run_date=dt_utc,
                    args=[tweet, user_id, idx],
                    id=job_id
                )
        st.success("Tweets will be posted automatically at scheduled times (as long as this app is running).")

    st.info("Keep this app running for scheduled tweets to post. Check your X/Twitter account for live results!")
