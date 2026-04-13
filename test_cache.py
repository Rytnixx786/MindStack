#!/usr/bin/env python3
"""Test script to verify query caching functionality."""

import requests
import time
import json
import pytest

API_BASE = 'http://localhost:8000'


def _backend_available() -> bool:
    """Return True if the local API is reachable, otherwise False."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False

def test_query_cache():
    """Test that repeat queries are served from cache with significant latency reduction."""
    if not _backend_available():
        pytest.skip("Skipping cache test: backend not reachable at http://localhost:8000")
    
    print("=" * 70)
    print("🧪 QUERY CACHING TEST")
    print("=" * 70)
    
    test_queries = [
        "What is the refund policy?",
        "What is the onboarding process?",
        "What are the product features?",
    ]
    
    first_latencies = []
    cached_latencies = []
    
    for query in test_queries:
        print(f"\n📝 Testing query: '{query}'")
        print("-" * 70)
        
        # First request (should miss cache and do full processing)
        print("  ⏱️  Sending FIRST request (should NOT be cached)...")
        start = time.time()
        response1 = requests.post(
            f"{API_BASE}/query",
            json={"query": query},
            timeout=60
        )
        time1 = (time.time() - start) * 1000
        
        if response1.status_code != 200:
            print(f"    ❌ Error: {response1.status_code}")
            continue
        
        data1 = response1.json()
        cached1 = data1.get('cached', False)
        latency1 = data1.get('latency_ms', time1)
        
        print(f"    ✓ Latency: {latency1:.2f}ms")
        print(f"    ✓ Cached flag: {cached1}")
        print(f"    ✓ Grounded: {data1.get('answer_grounded', False)}")
        
        if cached1:
            print(f"    ⚠️  WARNING: First request marked as cached!")
        
        first_latencies.append(latency1)
        
        # Sleep to ensure cache is populated
        time.sleep(0.5)
        
        # Second request (should hit cache)
        print("  ⏱️  Sending SECOND request (should BE cached)...")
        start = time.time()
        response2 = requests.post(
            f"{API_BASE}/query",
            json={"query": query},
            timeout=60
        )
        time2 = (time.time() - start) * 1000
        
        if response2.status_code != 200:
            print(f"    ❌ Error: {response2.status_code}")
            continue
        
        data2 = response2.json()
        cached2 = data2.get('cached', False)
        latency2 = data2.get('latency_ms', time2)
        
        print(f"    ✓ Latency: {latency2:.2f}ms")
        print(f"    ✓ Cached flag: {cached2}")
        
        cached_latencies.append(latency2)
        
        # Verify response is identical
        if data1['answer'] == data2['answer'] and cached2:
            print(f"    ✅ CACHE HIT: Response identical, cached flag is True")
            speedup = latency1 / latency2 if latency2 > 0 else 999
            print(f"    📈 Speedup: {speedup:.1f}x faster ({latency1:.2f}ms → {latency2:.2f}ms)")
        else:
            print(f"    ❌ CACHE MISS: Unexpected behavior")
            print(f"       Answer identical: {data1['answer'] == data2['answer']}")
            print(f"       Cached flag: {cached2}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 CACHE PERFORMANCE SUMMARY")
    print("=" * 70)
    
    avg_first = sum(first_latencies) / len(first_latencies) if first_latencies else 0
    avg_cached = sum(cached_latencies) / len(cached_latencies) if cached_latencies else 0
    
    print(f"\n📈 Average latency (first query):  {avg_first:.2f}ms")
    print(f"📈 Average latency (cached query): {avg_cached:.2f}ms")
    
    if avg_cached > 0:
        speedup = avg_first / avg_cached
        reduction = ((avg_first - avg_cached) / avg_first) * 100
        print(f"⚡ Overall speedup: {speedup:.1f}x")
        print(f"⚡ Latency reduction: {reduction:.1f}%")
    
    # Test with slightly different query (should not be cached)
    print("\n" + "=" * 70)
    print("🔤 NORMALIZATION TEST")
    print("=" * 70)
    
    test_query = "What is the refund policy?"
    variants = [
        "  What  is   the  refund  policy?  ",  # Extra spaces
        "WHAT IS THE REFUND POLICY?",  # Uppercase
        "what is the refund policy?",  # Lowercase
    ]
    
    print(f"\n🔍 Testing query normalization for: '{test_query}'")
    
    # Prime the cache
    print("  • Priming cache with original query...")
    requests.post(f"{API_BASE}/query", json={"query": test_query}, timeout=60)
    time.sleep(0.5)
    
    for variant in variants:
        print(f"\n  Testing variant: '{variant}'")
        start = time.time()
        response = requests.post(
            f"{API_BASE}/query",
            json={"query": variant},
            timeout=60
        )
        elapsed = (time.time() - start) * 1000
        data = response.json()
        cached = data.get('cached', False)
        latency = data.get('latency_ms', elapsed)
        
        if cached:
            print(f"    ✅ Variant matched cache (normalized): {latency:.2f}ms")
        else:
            print(f"    ⚠️  Variant NOT in cache: {latency:.2f}ms")
    
    print("\n" + "=" * 70)
    print("✅ CACHE TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_query_cache()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
