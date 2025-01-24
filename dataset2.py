import random
import csv
from datetime import datetime, timedelta
import numpy as np

def generate_phone_number():
    return f"{random.randint(6000000000, 9999999999)}"

def generate_spam_classification():
    # Introduce some randomness in classification
    return random.choice(["Spam", "Not Spam", "Not Spam", "Not Spam"])  # Make spam less frequent

def generate_caller_type(classification):
    if classification == "Spam":
        return random.choice(["Telemarketer", "Scammer", "Survey", "Fraudulent Service", "Unknown"])
    return random.choice(["None", "Business", "Personal", "Service"])

def generate_location():
    # Add more variety in locations and unknown cases
    locations = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Unknown", "Unknown", 
                "Pune", "Hyderabad", "Ahmedabad", "Unknown", "Jaipur"]
    return random.choice(locations)

def generate_time_of_call():
    return random.choice(["Morning", "Afternoon", "Evening", "Night"])

def generate_duration():
    # More varied duration with some outliers
    if random.random() < 0.1:  # 10% chance of outlier
        return random.randint(600, 1200)
    return random.randint(10, 600)

def generate_conversation_text(classification, caller_type):
    # Add more variety and some ambiguous cases
    spam_conversations = {
        "Telemarketer": [
            "Hello, we're offering a great deal on credit cards with zero annual fees!",
            "We have an exclusive offer for you today.",
            "Are you interested in saving money on your utility bills?",
            "Special discount available only today!"
        ],
        "Scammer": [
            "This is regarding your car's extended warranty.",
            "Your bank account needs verification.",
            "You've won a prize in our lucky draw!",
            "Important notice about your account security."
        ],
        "Survey": [
            "Would you like to participate in a quick survey?",
            "We're gathering customer feedback.",
            "Can you spare 5 minutes for a survey?",
            "Your opinion matters to us."
        ],
        "Fraudulent Service": [
            "Your computer might be infected.",
            "Your social security number needs verification.",
            "We detected suspicious activity.",
            "Urgent action required on your account."
        ]
    }
    
    legitimate_conversations = [
        "Hi, this is regarding your job application.",
        "Hello, calling to confirm your appointment.",
        "This is your delivery service.",
        "Your requested callback about the service issue.",
        "Following up on your inquiry.",
        "This is about your recent purchase.",
        "Calling from your doctor's office.",
        "About your recent service request.",
        "This is your bank calling to verify a transaction."
    ]

    # Add some ambiguous cases
    ambiguous_conversations = [
        "Hello, this is about your subscription.",
        "We're calling about your account.",
        "This is a follow-up call.",
        "Important information about your service.",
        "Calling regarding your recent inquiry."
    ]

    if random.random() < 0.1:  # 10% chance of ambiguous message
        return random.choice(ambiguous_conversations)
    
    if classification == "Spam":
        return random.choice(spam_conversations.get(caller_type, spam_conversations["Telemarketer"]))
    return random.choice(legitimate_conversations)

def generate_confidence_score(classification):
    # More realistic confidence scores with noise
    base_score = random.randint(55, 85) if classification == "Spam" else random.randint(15, 45)
    noise = random.randint(-10, 10)
    return max(0, min(100, base_score + noise))

def generate_verified_user_reported():
    # Make verified reports less common
    return random.random() < 0.3  # 30% chance of being verified

def generate_dataset(num_entries):
    dataset = []
    for _ in range(num_entries):
        classification = generate_spam_classification()
        caller_type = generate_caller_type(classification)
        
        # Add noise to reports and call frequency
        reports = 0
        if classification == "Spam":
            reports = int(np.random.gamma(2, 3))  # More realistic distribution
        
        call_frequency = 0
        if classification == "Spam":
            call_frequency = int(np.random.gamma(1.5, 2))
        else:
            call_frequency = int(np.random.gamma(0.5, 1))

        entry = [
            generate_phone_number(),
            classification,
            caller_type,
            reports,
            call_frequency,
            generate_location(),
            generate_time_of_call(),
            generate_duration(),
            generate_user_review(classification),
            generate_confidence_score(classification),
            generate_last_report_date(),
            generate_verified_user_reported(),
            generate_conversation_text(classification, caller_type)
        ]
        dataset.append(entry)
    return dataset

def generate_user_review(classification):
    if classification == "Spam":
        return random.choice([
            "Annoying call", "Scam alert", "Unwanted sales", "Telemarketing spam",
            "Not sure", "Possible spam", "Suspicious call"
        ])
    return random.choice([
        "Helpful service", "Genuine call", "Important call",
        "Normal call", "Business call", "OK", "Expected call"
    ])

def generate_last_report_date():
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    return (start_date + (end_date - start_date) * random.random()).strftime("%Y-%m-%d")

def save_to_csv(dataset, filename="spam_call_dataset.csv"):
    headers = [
        "Phone Number", "Spam Classification", "Caller Type", "Number of Reports", 
        "Call Frequency", "Location", "Time of Call", "Duration (seconds)", 
        "User Review", "Confidence Score (%)", "Last Report Date", 
        "Reported by Verified Users", "Conversation Text"
    ]
    
    with open(filename, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(dataset)
    print(f"Dataset with {len(dataset)} entries saved to '{filename}'.")

if __name__ == "__main__":
    num_entries = 10000
    dataset = generate_dataset(num_entries)
    save_to_csv(dataset)