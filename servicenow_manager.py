import pysnow

def submit_to_servicenow_pysnow(category, description):
    instance = 'your_instance_name' # Just the name, not the full URL
    user = 'admin'
    password = 'your_password'

    s = pysnow.Client(instance=instance, user=user, password=password)

    incident = s.resource(api_path='/table/incident')

    new_record = {
        'short_description': f"AI Classified: {category}",
        'description': description,
        'urgency': 2, # Medium
        'impact': 2,
        'comments': "Ticket generated automatically by AI Classifier."
    }

    try:
        result = incident.create(payload=new_record)
        return True, result['number']
    except Exception as e:
        return False, str(e)
