#!/usr/bin/env python3
"""
Comprehensive System Dry Run Test
Tests all major components and API endpoints
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://127.0.0.1:5000"

class DryRunTester:
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        
    def log(self, test_name, status, details=""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "time": timestamp
        }
        self.results.append(result)
        
        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{icon} [{timestamp}] {test_name}: {status}")
        if details:
            print(f"    → {details}")
    
    def test_connection(self):
        """Test if Flask app is running"""
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            if response.status_code == 200:
                self.log("Connection", "PASS", "Flask app is running")
                return True
            else:
                self.log("Connection", "FAIL", f"Unexpected status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.log("Connection", "FAIL", "Cannot connect to Flask app on http://127.0.0.1:5000")
            return False
        except Exception as e:
            self.log("Connection", "FAIL", str(e))
            return False
    
    def test_chat_basic(self):
        """Test basic chat functionality"""
        try:
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/chat",
                json={
                    "message": "Hello, how are you?",
                    "user_id": "test_user"
                },
                timeout=15
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                if "reply" in data and len(data["reply"]) > 5:
                    self.log("Chat Basic", "PASS", f"Response: '{data['reply'][:50]}...' ({elapsed:.2f}s)")
                    return True
                else:
                    self.log("Chat Basic", "FAIL", "No valid reply in response")
                    return False
            else:
                self.log("Chat Basic", "FAIL", f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log("Chat Basic", "FAIL", str(e))
            return False
    
    def test_chat_academic(self):
        """Test academic mode chat"""
        try:
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/chat",
                json={
                    "message": "Explain photosynthesis",
                    "user_id": "test_user",
                    "academic_mode": True
                },
                timeout=15
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                if "reply" in data and len(data["reply"]) > 10:
                    self.log("Chat Academic", "PASS", f"Response received ({elapsed:.2f}s)")
                    return True
                else:
                    self.log("Chat Academic", "FAIL", "Invalid response")
                    return False
            else:
                self.log("Chat Academic", "FAIL", f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log("Chat Academic", "FAIL", str(e))
            return False
    
    def test_chat_complex_query(self):
        """Test complex business finance query"""
        try:
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/chat",
                json={
                    "message": 'John Locke stated "Business Finances are like our shoes; if too small, they gall and pinch us; but if too large, they cause us to stumble and to trip." Explain this statement with reference to sources of business finance.',
                    "user_id": "test_user"
                },
                timeout=20
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                reply = data.get("reply", "")
                
                # Check response quality
                has_content = len(reply) > 50
                has_coherence = not reply.count("grants2grants") > 0  # Check for tokenization errors
                
                if has_content and has_coherence:
                    self.log("Chat Complex", "PASS", f"Response received ({elapsed:.2f}s)")
                    return True
                elif has_content:
                    self.log("Chat Complex", "WARN", f"Response has content but quality issues ({elapsed:.2f}s)")
                    return True
                else:
                    self.log("Chat Complex", "FAIL", "Response too short or incoherent")
                    return False
            else:
                self.log("Chat Complex", "FAIL", f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log("Chat Complex", "FAIL", str(e))
            return False
    
    def test_profile_endpoint(self):
        """Test profile management endpoints"""
        try:
            # Save profile
            save_response = requests.post(
                f"{BASE_URL}/profile",
                json={
                    "user_id": "test_user",
                    "name": "Test User",
                    "personality": "curious, analytical"
                },
                timeout=5
            )
            
            # Get profile
            get_response = requests.get(
                f"{BASE_URL}/profile?user_id=test_user",
                timeout=5
            )
            
            if save_response.status_code == 200 and get_response.status_code == 200:
                data = get_response.json()
                if data.get("name") == "Test User":
                    self.log("Profile Endpoint", "PASS", "Save and retrieve working")
                    return True
                else:
                    self.log("Profile Endpoint", "FAIL", "Profile data mismatch")
                    return False
            else:
                self.log("Profile Endpoint", "FAIL", f"Status codes: {save_response.status_code}, {get_response.status_code}")
                return False
        except Exception as e:
            self.log("Profile Endpoint", "FAIL", str(e))
            return False
    
    def test_feedback_endpoint(self):
        """Test feedback endpoint"""
        try:
            response = requests.post(
                f"{BASE_URL}/feedback",
                json={
                    "response": "This is a test response",
                    "rating": "good"
                },
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    self.log("Feedback Endpoint", "PASS", "Feedback recorded")
                    return True
                else:
                    self.log("Feedback Endpoint", "FAIL", "Invalid response")
                    return False
            else:
                self.log("Feedback Endpoint", "FAIL", f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log("Feedback Endpoint", "FAIL", str(e))
            return False
    
    def test_response_speed(self):
        """Test response speed benchmark"""
        try:
            times = []
            for i in range(3):
                start = time.time()
                response = requests.post(
                    f"{BASE_URL}/chat",
                    json={
                        "message": f"What is test {i}?",
                        "user_id": "test_user"
                    },
                    timeout=15
                )
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    times.append(elapsed)
            
            if len(times) == 3:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                if avg_time < 5:
                    status = "PASS"
                elif avg_time < 8:
                    status = "WARN"
                else:
                    status = "FAIL"
                
                details = f"Min: {min_time:.2f}s, Avg: {avg_time:.2f}s, Max: {max_time:.2f}s"
                self.log("Response Speed", status, details)
                return status in ["PASS", "WARN"]
            else:
                self.log("Response Speed", "FAIL", f"Only {len(times)}/3 requests succeeded")
                return False
        except Exception as e:
            self.log("Response Speed", "FAIL", str(e))
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("LITTLEFOXAI SYSTEM DRY RUN TEST")
        print("="*80)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Core tests
        if not self.test_connection():
            print("\n❌ Cannot connect to Flask app. Please start it first.")
            return False
        
        print("\n--- Testing Chat Functionality ---")
        self.test_chat_basic()
        self.test_chat_academic()
        self.test_chat_complex_query()
        
        print("\n--- Testing Endpoints ---")
        self.test_profile_endpoint()
        self.test_feedback_endpoint()
        
        print("\n--- Testing Performance ---")
        self.test_response_speed()
        
        # Summary
        print("\n" + "="*80)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        warned = sum(1 for r in self.results if r["status"] == "WARN")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        total = len(self.results)
        
        print(f"\nRESULTS: {passed} PASS | {warned} WARN | {failed} FAIL (out of {total})")
        
        if failed == 0:
            print("✅ All tests passed! System is ready.")
        elif failed <= 2:
            print("⚠️  Some tests had issues. Review above.")
        else:
            print("❌ Multiple failures detected. Check configuration and logs.")
        
        print("="*80 + "\n")
        
        return failed == 0

if __name__ == "__main__":
    tester = DryRunTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
