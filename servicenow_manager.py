import requests
import json

def submit_to_servicenow(category, description, urgency="2"):
    # Replace with your actual details
    instance_url = "https://YOUR_INSTANCE.service-now.com/api/now/table/incident"
    user = "admin"
    password = "your_password"

    payload = {
        "short_description": f"AI Classified: {category}",
        "description": description,
        "urgency": urgency,  # 1: High, 2: Medium, 3: Low
        "assignment_group": "Software" if category == "Technical Issue" else "Customer Support"
    }

    response = requests.post(
        instance_url,
        auth=(user, password),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 201:
        result = response.json()
        return True, result['result']['number'] # Returns the Ticket Number (e.g. INC0010001)
    else:
        return False, response.text