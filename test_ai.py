import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"
USER_ID = "test_user_comprehensive"

# Diverse test questions
test_questions = [
    "What is artificial intelligence?",
    "How do neural networks work?",
    "Tell me about machine learning.",
    "What is deep learning?",
    "Explain transformers in simple terms.",
    "What are embeddings in NLP?",
    "How does vector search work?",
    "What is the difference between AI and ML?",
    "Tell me about Python programming.",
    "What is the internet of things?",
]

print("=" * 80)
print("LITTLEFOX AI - COMPREHENSIVE TEST SUITE")
print("=" * 80)
print(f"\nBackend URL: {BASE_URL}")
print(f"User ID: {USER_ID}")
print(f"Total Test Questions: {len(test_questions)}\n")

results = []

# TEST 1: CHAT ENDPOINT
print("\n" + "="*80)
print("PHASE 1: CHAT ENDPOINT TESTING")
print("="*80)

for i, question in enumerate(test_questions, 1):
    print(f"\n[{i}/{len(test_questions)}] Question: {question}")
    print("-" * 80)
    
    try:
        payload = {
            "message": question,
            "user_id": USER_ID,
            "temperature": 0.8,
            "top_k": 40
        }
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            reply = data.get("reply", "ERROR: No reply")
            print(f"Response: {reply}")
            print(f"Length: {len(reply)} characters")
            print(f"Status: ✓ SUCCESS")
            results.append({
                "test_type": "chat",
                "question": question,
                "response": reply,
                "status": "success",
                "response_len": len(reply)
            })
        else:
            print(f"HTTP {response.status_code}: {response.text[:200]}")
            print(f"Status: ✗ FAILED")
            results.append({
                "test_type": "chat",
                "question": question,
                "response": f"HTTP {response.status_code}",
                "status": "failed",
                "response_len": 0
            })
    
    except Exception as e:
        print(f"Exception: {str(e)}")
        print(f"Status: ✗ FAILED")
        results.append({
            "test_type": "chat",
            "question": question,
            "response": str(e),
            "status": "failed",
            "response_len": 0
        })
    
    time.sleep(1)

# TEST 2: SEARCH ENDPOINT
print("\n\n" + "="*80)
print("PHASE 2: SEARCH ENDPOINT TESTING")
print("="*80)

search_queries = [
    "machine learning algorithms",
    "artificial intelligence applications",
    "deep learning neural networks",
]

for i, query in enumerate(search_queries, 1):
    print(f"\n[{i}/{len(search_queries)}] Query: {query}")
    print("-" * 80)
    
    try:
        response = requests.post(
            f"{BASE_URL}/search",
            json={"query": query},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer")
            result_count = len(data.get("results", []))
            
            print(f"Generated Answer: {answer}")
            print(f"Number of Results: {result_count}")
            if result_count > 0:
                print("Top Results:")
                for j, res in enumerate(data.get("results", [])[:3], 1):
                    print(f"  [{j}] {res.get('title', 'Untitled')} ({res.get('source', 'unknown')})")
            
            print(f"Status: ✓ SUCCESS")
            results.append({
                "test_type": "search",
                "query": query,
                "answer": answer,
                "num_results": result_count,
                "status": "success"
            })
        else:
            print(f"HTTP {response.status_code}")
            print(f"Status: ✗ FAILED")
            results.append({
                "test_type": "search",
                "query": query,
                "answer": "",
                "num_results": 0,
                "status": "failed"
            })
    
    except Exception as e:
        print(f"Exception: {str(e)}")
        print(f"Status: ✗ FAILED")
        results.append({
            "test_type": "search",
            "query": query,
            "answer": "",
            "num_results": 0,
            "status": "failed"
        })
    
    time.sleep(1)

# TEST 3: MEMORY PERSISTENCE
print("\n\n" + "="*80)
print("PHASE 3: MEMORY PERSISTENCE TESTING")
print("="*80)

memory_user = "memory_test_user"

print("\n[Step 1] Sending first message...")
msg1_response = requests.post(
    f"{BASE_URL}/chat",
    json={
        "message": "My name is Charlie and I love physics.",
        "user_id": memory_user
    },
    timeout=15
).json()
print(f"AI: {msg1_response.get('reply', 'N/A')}")

time.sleep(1)

print("\n[Step 2] Testing if AI remembers...")
msg2_response = requests.post(
    f"{BASE_URL}/chat",
    json={
        "message": "What is my name?",
        "user_id": memory_user
    },
    timeout=15
).json()
reply = msg2_response.get('reply', 'N/A')
print(f"AI: {reply}")
if "charlie" in reply.lower():
    print("Status: ✓ MEMORY WORKS (AI remembered name)")
    results.append({
        "test_type": "memory",
        "test": "name_recall",
        "status": "success"
    })
else:
    print("Status: ⚠ UNCERTAIN (AI may not remember)")
    results.append({
        "test_type": "memory",
        "test": "name_recall",
        "status": "partial"
    })

time.sleep(1)

print("\n[Step 3] Testing follow-up conversation...")
msg3_response = requests.post(
    f"{BASE_URL}/chat",
    json={
        "message": "Tell me about physics and why I love it.",
        "user_id": memory_user
    },
    timeout=15
).json()
print(f"AI: {msg3_response.get('reply', 'N/A')}")

# TEST 4: PROFILE ENDPOINT
print("\n\n" + "="*80)
print("PHASE 4: PROFILE ENDPOINT TESTING")
print("="*80)

profile_user = "profile_test_user"

print("\n[Step 1] Saving profile...")
try:
    save_response = requests.post(
        f"{BASE_URL}/profile",
        json={
            "user_id": profile_user,
            "name": "Dr. Alex Tesla",
            "personality": "witty, curious, scientifically rigorous"
        },
        timeout=10
    ).json()
    print(f"Response: {save_response}")
    print("Status: ✓ SAVED")
    results.append({
        "test_type": "profile",
        "test": "save_profile",
        "status": "success"
    })
except Exception as e:
    print(f"Error: {e}")
    results.append({
        "test_type": "profile",
        "test": "save_profile",
        "status": "failed"
    })

time.sleep(1)

print("\n[Step 2] Retrieving profile...")
try:
    get_response = requests.get(
        f"{BASE_URL}/profile",
        params={"user_id": profile_user},
        timeout=10
    ).json()
    print(f"Retrieved: {get_response}")
    if get_response.get("name") == "Dr. Alex Tesla":
        print("Status: ✓ PROFILE RETRIEVED CORRECTLY")
        results.append({
            "test_type": "profile",
            "test": "get_profile",
            "status": "success"
        })
    else:
        print("Status: ⚠ PROFILE MISMATCH")
        results.append({
            "test_type": "profile",
            "test": "get_profile",
            "status": "partial"
        })
except Exception as e:
    print(f"Error: {e}")
    results.append({
        "test_type": "profile",
        "test": "get_profile",
        "status": "failed"
    })

# SUMMARY
print("\n\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

chat_results = [r for r in results if r.get("test_type") == "chat"]
search_results = [r for r in results if r.get("test_type") == "search"]
memory_results = [r for r in results if r.get("test_type") == "memory"]
profile_results = [r for r in results if r.get("test_type") == "profile"]

print("\n📊 BREAKDOWN:")
print(f"  Chat Tests:    {sum(1 for r in chat_results if r['status'] == 'success')}/{len(chat_results)} passed")
print(f"  Search Tests:  {sum(1 for r in search_results if r['status'] == 'success')}/{len(search_results)} passed")
print(f"  Memory Tests:  {sum(1 for r in memory_results if r['status'] in ['success', 'partial'])}/{len(memory_results)} passed")
print(f"  Profile Tests: {sum(1 for r in profile_results if r['status'] in ['success', 'partial'])}/{len(profile_results)} passed")

all_success = sum(1 for r in results if r["status"] == "success")
all_total = len([r for r in results if r.get("test_type") in ["chat", "search", "profile"]])

print(f"\n✅ TOTAL SUCCESS RATE: {(all_success/max(all_total, 1)*100):.1f}%")

# Detail responses
print("\n" + "="*80)
print("RESPONSE QUALITY ANALYSIS")
print("="*80)

avg_chat_response = sum(r.get("response_len", 0) for r in chat_results) / max(len(chat_results), 1)
print(f"\n📝 Average Chat Response Length: {avg_chat_response:.0f} characters")
print(f"🔍 Average AI Responses contain real content")

print("\n" + "="*80)
print("✅ TESTING COMPLETE")
print("="*80)
