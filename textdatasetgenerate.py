import pandas as pd
import random

# Departments
departments = ['electricity', 'water', 'roads', 'garbage', 'streetlight', 'health']
samples_per_dept = 200

# Variations for locations, time expressions, severity, synonyms
locations = ['MG Road', 'Park Street', 'Main Street', 'Central Avenue', 'School Lane', 'Market Road', 'Hospital Road', 'Colony A', 'Block C', 'Near the temple']
time_expr = ['since yesterday', 'for 2 days', 'from last week', 'all week', 'since morning', 'since last night']
severity = ['urgent', 'critical', 'major', 'small', 'huge', 'serious']

# Synonyms for each department
dept_synonyms = {
    'electricity': ['No electricity', 'Power outage', 'Electricity supply disrupted'],
    'water': ['No water', 'Water supply disrupted', 'Water not available'],
    'roads': ['Potholes', 'Broken road', 'Cracks on the road', 'Damaged street'],
    'garbage': ['Garbage overflowing', 'Trash piling up', 'Waste not collected', 'Rubbish accumulation'],
    'streetlight': ['Streetlight not working', 'Lamp broken', 'Light is out', 'Dark street'],
    'health': ['Clinic understaffed', 'Hospital not equipped', 'Medical services unavailable', 'Health center issue']
}

def generate_complaints(label, n):
    complaints = []
    for _ in range(n):
        loc = random.choice(locations)
        time = random.choice(time_expr)
        sev = random.choice(severity)
        phrase = random.choice(dept_synonyms[label])
        
        # Combine variations to create diverse complaint
        if label in ['electricity', 'water']:
            complaint = f"{phrase} in {loc} {time}; {sev} attention needed."
        elif label == 'roads':
            complaint = f"{phrase} near {loc} {time}, causing {sev} inconvenience to commuters."
        elif label == 'garbage':
            complaint = f"{phrase} around {loc} {time}, spreading {sev} smell and unhygienic conditions."
        elif label == 'streetlight':
            complaint = f"{phrase} at {loc} {time}; area is {sev} dark at night."
        elif label == 'health':
            complaint = f"{phrase} at {loc} {time}; {sev} impact on residents."
        
        complaints.append(complaint)
    return complaints

# Generate dataset
all_complaints = []
for dept in departments:
    texts = generate_complaints(dept, samples_per_dept)
    for t in texts:
        all_complaints.append({'complaint_text': t, 'label': dept})

# Shuffle the dataset
random.shuffle(all_complaints)

# Create DataFrame
df = pd.DataFrame(all_complaints)

# Save CSV
df.to_csv('balanced_varied_text_dataset.csv', index=False)

print("CSV file 'balanced_varied_text_dataset.csv' created with 1200 diverse samples.")
