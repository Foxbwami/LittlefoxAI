#!/usr/bin/env python3
import requests
import json
import sys
import time

# Test message
test_message = '''John Locke stated "Business Finances are like our shoes; if too small, they gall and pinch us; but if too large, they cause us to stumble and to trip." Explain the above statement with reference to the various sources of Business Finances'''

# Make the request
try:
    start_time = time.time()
    response = requests.post(
        'http://127.0.0.1:5000/chat',
        json={
            'message': test_message,
            'user_id': 'test_user',
            'academic_mode': True
        },
        timeout=30
    )
    elapsed = time.time() - start_time
    
    # Pretty print the response
    result = response.json()
    print("\n" + "="*80)
    print(f"RESPONSE TIME: {elapsed:.2f} seconds")
    print("="*80)
    print("\nMODEL RESPONSE:")
    print(json.dumps(result, indent=2))
    print("="*80 + "\n")
    
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to Flask app. Is it running on http://127.0.0.1:5000?")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {str(e)}")
    sys.exit(1)
